import json
from dataclasses import dataclass, field

import carla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from bev import SceneBEV
from lane_detection import LaneDetection, Spline
from lateral_control import LateralController
from object_detection import ObjectDetection
from sim_session import FIXED_DELTA_SECONDS, CarlaSession, build_arg_parser, sample_spline
from waypoint_prediction import target_speed_prediction, waypoint_prediction

# Half the width of the Lincoln MKZ ego vehicle; the car has left its lane once its centre is further than
# lane_width / 2 - EGO_HALF_WIDTH_M from the lane centre.
EGO_HALF_WIDTH_M = 1.05


@dataclass
class LaneKeepingLog:
    """Per-step ground truth and commands for judging the lateral controller."""

    steer: list[float] = field(default_factory=list)
    lane_offset_m: list[float] = field(default_factory=list)
    heading_error_deg: list[float] = field(default_factory=list)
    in_lane: list[bool] = field(default_factory=list)
    in_junction: list[bool] = field(default_factory=list)
    distance_m: list[float] = field(default_factory=list)

    def record(self, world_map: carla.Map, vehicle: carla.Actor, steer: float, speed_mps: float) -> None:
        """Compare the vehicle pose with the centre of the lane it is in.

        Parameters
        ----------
        world_map : carla.Map
            Map of the current town, for the ground-truth lane.
        vehicle : carla.Actor
            Ego vehicle.
        steer : float
            Steer command applied this step.
        speed_mps : float
            Vehicle speed in m/s.
        """
        transform = vehicle.get_transform()
        waypoint = world_map.get_waypoint(transform.location)
        offset = transform.location - waypoint.transform.location
        right = waypoint.transform.get_right_vector()
        lane_offset = offset.x * right.x + offset.y * right.y
        heading_error = (transform.rotation.yaw - waypoint.transform.rotation.yaw + 180.0) % 360.0 - 180.0

        self.steer.append(steer)
        self.lane_offset_m.append(lane_offset)
        self.heading_error_deg.append(heading_error)
        # get_waypoint snaps to the nearest lane, which is the oncoming one once the car has crossed over.
        right_way = abs(heading_error) < 90.0
        self.in_lane.append(right_way and abs(lane_offset) <= waypoint.lane_width / 2 - EGO_HALF_WIDTH_M)
        self.in_junction.append(waypoint.is_junction)
        self.distance_m.append((self.distance_m[-1] if self.distance_m else 0.0) + speed_mps * FIXED_DELTA_SECONDS)

    def summary(self) -> dict[str, float]:
        """Lane keeping statistics outside junctions, where the map's lane centre is well defined.

        Returns
        -------
        dict[str, float]
            Empty before the first step. distance_m (total), distance_to_departure_m (driven before the car first left its lane outside a
            junction), steps, in_lane_fraction, mean_abs_offset_m and rms_heading_error_deg (both over the
            in-lane steps) and mean_abs_steer_change (steering jitter per step, in CARLA steer units).
        """
        outside = ~np.array(self.in_junction, dtype=bool)
        in_lane = np.array(self.in_lane, dtype=bool)
        distance = np.array(self.distance_m)
        departures = np.flatnonzero(outside & ~in_lane)
        tracked = outside & in_lane
        offset = np.array(self.lane_offset_m)[tracked]
        heading = np.array(self.heading_error_deg)[tracked]
        if distance.size == 0:
            return {}
        return {
            'distance_m': round(float(distance[-1]), 1),
            'distance_to_departure_m': round(float(distance[departures[0]] if departures.size else distance[-1]), 1),
            'steps': int(outside.sum()),
            'in_lane_fraction': round(float(np.mean(in_lane[outside])), 3) if outside.any() else float('nan'),
            'mean_abs_offset_m': round(float(np.mean(np.abs(offset))), 3) if offset.size else float('nan'),
            'rms_heading_error_deg': round(float(np.sqrt(np.mean(heading ** 2))), 2) if heading.size else float('nan'),
            'mean_abs_steer_change': round(float(np.mean(np.abs(np.diff(self.steer)))), 4),
        }


def ground_truth_waypoints(
    world_map: carla.Map,
    vehicle: carla.Actor,
    LD_module: LaneDetection,
    num_waypoints: int = 6,
) -> npt.NDArray[np.float64]:
    """Centre of the ego lane from the CARLA map, over the same range ahead as the detected waypoints.

    Parameters
    ----------
    world_map : carla.Map
        Map of the current town.
    vehicle : carla.Actor
        Ego vehicle.
    LD_module : LaneDetection
        Lane detector, whose cut coordinates the waypoints are returned in.
    num_waypoints : int
        Number of waypoints, evenly spaced along the lane over the cut's forward range.

    Returns
    -------
    (2 x NumWaypoints) array
        (x, y) waypoints in cut_gray coordinates, nearest first. At a junction the first of the map's branches is
        followed.
    """
    transform = vehicle.get_transform()
    waypoint = world_map.get_waypoint(transform.location)
    rows_m = LD_module.bev_metres_per_pixel()[0]
    near_m = LD_module.bev_forward_range_m[0]
    world_to_vehicle = np.array(transform.get_inverse_matrix())

    points = np.ones((4, num_waypoints))
    for i, distance in enumerate(np.linspace(near_m, near_m + LD_module.cut_size * rows_m, num_waypoints)):
        ahead = waypoint.next(distance)
        location = (ahead[0] if ahead else waypoint).transform.location
        points[:3, i] = location.x, location.y, location.z
    vehicle_points = (world_to_vehicle @ points)[:2]
    return LD_module.bev_to_cut(LD_module.vehicle_to_bev_pixels(vehicle_points))


def plot_step(
    axes: npt.NDArray[np.object_],
    LD_module: LaneDetection,
    front: npt.NDArray[np.uint8],
    scene: npt.NDArray[np.uint8],
    splines: tuple[Spline | None, Spline | None],
    waypoints: npt.NDArray[np.float64] | None,
    log: LaneKeepingLog,
    title: str,
) -> None:
    """Draw the front view, the scene BEV with the path, and the steering and lane offset history.

    Parameters
    ----------
    axes : (3,) array of matplotlib Axes
        Axes for the front view, the BEV and the history.
    LD_module : LaneDetection
        Lane detector, used for the coordinate conversions.
    front : (ImageHeight x ImageWidth x 3) array
        RGB front camera image.
    scene : (ImageHeight x ImageWidth x 3) array
        Scene BEV from SceneBEV.render.
    splines : tuple[Spline or None, Spline or None]
        Left and right lane boundaries.
    waypoints : (2 x NumWaypoints) array or None
        Waypoints the controller followed, in cut_gray coordinates; None while no path is known.
    log : LaneKeepingLog
        History drawn in the third axis.
    title : str
        Title of the front view.
    """
    for ax in axes[:2]:
        ax.clear()
        ax.axis('off')
    height, width = front.shape[:2]

    axes[0].imshow(front)
    axes[1].imshow(scene)
    for to_pixels, ax in ((LD_module.cut_to_front_view, axes[0]), (LD_module.cut_to_bev_pixels, axes[1])):
        for spline in splines:
            u, v = to_pixels(sample_spline(spline))
            ax.plot(u, v, color='orange', linewidth=2)
        if waypoints is not None:
            u, v = to_pixels(waypoints)
            ax.plot(u, v, color='lime', linewidth=1, marker='o', markersize=4)
        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)
    axes[0].set_title(title, fontsize=9)
    axes[1].set_title('scene BEV: lane splines + waypoints', fontsize=9)

    axes[2].clear()
    axes[2].plot(log.steer, label='steer command')
    axes[2].plot(log.lane_offset_m, label='lane offset [m] (ground truth)')
    axes[2].axhline(0.0, color='grey', linewidth=0.5)
    axes[2].set_xlabel('step')
    axes[2].legend(loc='upper left', fontsize=7)

    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def main() -> None:
    """Drive with the Stanley controller at constant gas and log lane keeping.

    The controller follows the detected path by default; --path ground-truth follows the map's lane centre instead,
    which separates the controller's behaviour from perception errors.
    """
    parser = build_arg_parser('Live test of the lateral (Stanley) controller in CARLA.', 'lateral_control_snapshots')
    parser.add_argument('--gain', type=float, default=None, help='Stanley gain k; default: the controller default.')
    parser.add_argument('--damping', type=float, default=None,
                        help='Damping constant D; default: the controller default.')
    parser.add_argument('--way-type', default='smooth', choices=('center', 'smooth'))
    parser.add_argument('--path', default='detected', choices=('detected', 'ground-truth'),
                        help='Follow the waypoints from the perception pipeline or the lane centre from the map.')
    parser.add_argument('--throttle', type=float, default=0.6,
                        help='Constant gas while below --max-speed; at 0.4 the car needs ~10 s to reach 10 km/h.')
    parser.add_argument('--max-speed', type=float, default=30.0,
                        help='Gas is released (no braking) above this speed in km/h; Town01 has a 30 km/h limit.')
    parser.add_argument('--log', type=str, default=None, help='Write the lane keeping summary to this JSON file.')
    # Without longitudinal control the car neither brakes for traffic nor stops at red lights.
    parser.set_defaults(npcs=0)
    args = parser.parse_args()

    with CarlaSession(args, autopilot=False) as session:
        LD_module = LaneDetection()
        obj_detection = ObjectDetection()
        scene_bev = SceneBEV(LD_module)
        LatC_kwargs: dict[str, float] = {}
        if args.gain is not None:
            LatC_kwargs['gain_constant'] = args.gain
        if args.damping is not None:
            LatC_kwargs['damping_constant'] = args.damping
        LatC_module = LateralController(car_position=tuple(LD_module.car_position.astype(float)),
                                        metres_per_pixel=LD_module.bev_metres_per_pixel()[1], **LatC_kwargs)
        world_map = session.world.get_map()
        log = LaneKeepingLog()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        session.attach_figure(fig)
        plt.ion()
        plt.show()

        control = carla.VehicleControl()
        step = 0
        while session.running(step):
            front = session.tick(None if args.manual else control)
            speed = session.speed_mps
            detections = obj_detection.detect(front)

            # lane detection
            lane1, lane2 = LD_module.lane_detection(front, detections.obstacle_mask())

            # waypoint and target_speed prediction
            waypoints: npt.NDArray[np.float64] | None = None
            target_speed = float('nan')
            if args.path == 'ground-truth':
                waypoints = ground_truth_waypoints(world_map, session.vehicle, LD_module)
            elif lane1 is not None and lane2 is not None:
                waypoints = waypoint_prediction(lane1, lane2, way_type=args.way_type)
            if waypoints is not None:
                target_speed = target_speed_prediction(waypoints)

            # control with constant gas and no braking
            steer = 0.0 if waypoints is None else LatC_module.stanley(waypoints, speed)
            throttle = args.throttle if 3.6 * speed < args.max_speed else 0.0
            control = carla.VehicleControl(throttle=throttle, steer=steer)
            log.record(world_map, session.vehicle, steer, speed)

            if session.should_draw(step):
                print(f'step {step}: steer {steer:+0.2f} throttle {throttle:0.2f} speed {3.6 * speed:4.1f} km/h '
                      f'target speed {target_speed:4.1f} km/h')
                scene = scene_bev.render(front, detections, scene_bev.localize(detections))
                title = (f'step {step}, {3.6 * speed:.0f} km/h, steer {steer:+.2f}, '
                         f'k={LatC_module.gain_constant:g}, D={LatC_module.damping_constant:g}, {args.path} path')
                plot_step(axes, LD_module, front, scene, (lane1, lane2), waypoints, log, title)
                session.after_draw(step)
            step += 1

        session.save_final()
        summary = log.summary()
        print('lane keeping (outside junctions):', json.dumps(summary))
        if args.log is not None:
            with open(args.log, 'w') as f:
                json.dump({'args': {k: str(v) for k, v in vars(args).items()}, **summary}, f, indent=1)


if __name__ == '__main__':
    main()
