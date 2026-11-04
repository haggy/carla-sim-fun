import json
from dataclasses import dataclass, field

import carla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lane_detection import LaneDetection, Spline
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
from object_detection import ObjectDetection
from sim_session import FIXED_DELTA_SECONDS, CarlaSession, build_arg_parser, sample_spline
from test_lateral_control import LaneKeepingLog, ground_truth_waypoints
from waypoint_prediction import target_speed_prediction, waypoint_prediction

# Speed must be within this band of the target to count as settled.
SETTLE_BAND_KMH = 2.0
# The ego vehicle is spawned slightly above the road and bounces for about a second, which shows up as speed spikes.
SPAWN_SETTLE_STEPS = 20


@dataclass
class SpeedLog:
    """Per-step speeds and pedal commands for judging the longitudinal controller."""

    speed_kmh: list[float] = field(default_factory=list)
    target_kmh: list[float] = field(default_factory=list)
    gas: list[float] = field(default_factory=list)
    brake: list[float] = field(default_factory=list)

    def record(self, speed_kmh: float, target_kmh: float, gas: float, brake: float) -> None:
        """Append one step.

        Parameters
        ----------
        speed_kmh : float
            Vehicle speed in km/h.
        target_kmh : float
            Target speed in km/h.
        gas : float
            Gas command.
        brake : float
            Brake command.
        """
        self.speed_kmh.append(speed_kmh)
        self.target_kmh.append(target_kmh)
        self.gas.append(gas)
        self.brake.append(brake)

    def summary(self) -> dict[str, float]:
        """Speed tracking statistics.

        Every change of the target starts a segment that is judged as a step response: overshoot beyond the new
        target (in the direction of the change, as a percentage of the step) and the time until the speed stays
        within SETTLE_BAND_KMH of the target. Steps smaller than SETTLE_BAND_KMH are ignored, so with the predicted
        target speed only its larger jumps count. The first SPAWN_SETTLE_STEPS steps are left out.

        Returns
        -------
        dict[str, float]
            rms_error_kmh, mean_abs_error_kmh, mean_overshoot_pct, mean_settling_s (NaN when a segment never
            settles), unsettled_fraction (segments that never settled), max_accel_mps2 and max_decel_mps2
            (from the speed change between steps) and pedal_switches (changes between gas and brake).
        """
        speed = np.array(self.speed_kmh[SPAWN_SETTLE_STEPS:])
        target = np.array(self.target_kmh[SPAWN_SETTLE_STEPS:])
        if speed.size < 2:
            return {}
        error = target - speed
        accel = np.diff(speed) / 3.6 / FIXED_DELTA_SECONDS

        overshoots: list[float] = []
        settling: list[float] = []
        changes = np.flatnonzero(np.abs(np.diff(target)) >= SETTLE_BAND_KMH) + 1
        bounds = np.concatenate([[0], changes, [speed.size]])
        for start, end in zip(bounds[:-1], bounds[1:]):
            step = target[start] - speed[start]
            if abs(step) < SETTLE_BAND_KMH:
                continue
            beyond = np.sign(step) * (speed[start:end] - target[start:end])
            overshoots.append(100.0 * max(float(beyond.max()), 0.0) / abs(step))
            outside = np.flatnonzero(np.abs(error[start:end]) > SETTLE_BAND_KMH)
            settled = outside.size == 0 or outside[-1] < end - start - 1
            settling.append((outside[-1] + 1 if outside.size else 0) * FIXED_DELTA_SECONDS if settled else np.nan)

        pedal = np.sign(np.array(self.gas[SPAWN_SETTLE_STEPS:]) - np.array(self.brake[SPAWN_SETTLE_STEPS:]))
        pedal = pedal[pedal != 0]
        settling_arr = np.array(settling)
        return {
            'rms_error_kmh': round(float(np.sqrt(np.mean(error ** 2))), 2),
            'mean_abs_error_kmh': round(float(np.mean(np.abs(error))), 2),
            'mean_overshoot_pct': round(float(np.mean(overshoots)), 1) if overshoots else float('nan'),
            'mean_settling_s': round(float(np.nanmean(settling_arr)), 2)
            if settling and not np.all(np.isnan(settling_arr)) else float('nan'),
            'unsettled_fraction': round(float(np.mean(np.isnan(settling_arr))), 2) if settling else float('nan'),
            'max_accel_mps2': round(float(accel.max()), 2),
            'max_decel_mps2': round(float(-accel.min()), 2),
            'pedal_switches': int(np.count_nonzero(np.diff(pedal))),
        }


def step_target(step: int, levels_kmh: list[float], period_steps: int) -> float:
    """Piecewise constant target speed for tuning: cycles through levels_kmh, holding each for period_steps.

    Parameters
    ----------
    step : int
        Simulation step.
    levels_kmh : list[float]
        Target speeds in km/h.
    period_steps : int
        Steps each level is held.

    Returns
    -------
    float
        Target speed in km/h.
    """
    return levels_kmh[(step // period_steps) % len(levels_kmh)]


def plot_step(
    axes: npt.NDArray[np.object_],
    LD_module: LaneDetection,
    LongC_module: LongitudinalController,
    front: npt.NDArray[np.uint8],
    splines: tuple[Spline | None, Spline | None],
    waypoints: npt.NDArray[np.float64] | None,
    speed_kmh: float,
    target_kmh: float,
    step: int,
    title: str,
) -> None:
    """Draw the front view with the path, and the speed and target speed history.

    Parameters
    ----------
    axes : (2,) array of matplotlib Axes
        Axes for the front view and the speed history.
    LD_module : LaneDetection
        Lane detector, used for the coordinate conversions.
    LongC_module : LongitudinalController
        Controller that keeps and draws the speed history.
    front : (ImageHeight x ImageWidth x 3) array
        RGB front camera image.
    splines : tuple[Spline or None, Spline or None]
        Left and right lane boundaries.
    waypoints : (2 x NumWaypoints) array or None
        Waypoints the controllers followed, in cut_gray coordinates; None while no path is known.
    speed_kmh : float
        Current speed in km/h.
    target_kmh : float
        Current target speed in km/h.
    step : int
        Simulation step.
    title : str
        Title of the front view.
    """
    height, width = front.shape[:2]
    axes[0].clear()
    axes[0].axis('off')
    axes[0].imshow(front)
    for spline in splines:
        u, v = LD_module.cut_to_front_view(sample_spline(spline))
        axes[0].plot(u, v, color='orange', linewidth=2)
    if waypoints is not None:
        u, v = LD_module.cut_to_front_view(waypoints)
        axes[0].plot(u, v, color='lime', linewidth=1, marker='o', markersize=4)
    axes[0].set_xlim(0, width - 1)
    axes[0].set_ylim(height - 1, 0)
    axes[0].set_title(title, fontsize=9)

    LongC_module.plot_speed(speed_kmh, target_kmh, step, axes[1])
    axes[1].set_title('speed vs. target speed', fontsize=9)

    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def main() -> None:
    """Drive the full modular pipeline: Stanley steering and PID gas/brake on the target speed."""
    parser = build_arg_parser('Live test of the longitudinal (PID) controller in CARLA.',
                              'longitudinal_control_snapshots')
    parser.add_argument('--kp', type=float, default=None, help='K_p; default: the controller default.')
    parser.add_argument('--ki', type=float, default=None, help='K_i; default: the controller default.')
    parser.add_argument('--kd', type=float, default=None, help='K_d; default: the controller default.')
    parser.add_argument('--max-error-sum', type=float, default=None,
                        help='Anti-windup bound on the error sum; default: the controller default.')
    parser.add_argument('--path', default='detected', choices=('detected', 'ground-truth'),
                        help='Follow the waypoints from the perception pipeline or the lane centre from the map.')
    parser.add_argument('--target', default='predicted', choices=('predicted', 'steps'),
                        help='Target speed from target_speed_prediction, or the --step-levels profile for tuning.')
    parser.add_argument('--max-speed', type=float, default=60.0, help='v_max of target_speed_prediction, km/h.')
    parser.add_argument('--step-levels', type=float, nargs='+', default=[20.0, 40.0, 15.0, 35.0, 25.0],
                        help='Target speeds in km/h cycled by --target steps.')
    parser.add_argument('--step-period', type=int, default=160, help='Steps each --step-levels value is held.')
    parser.add_argument('--log', type=str, default=None, help='Write the tracking summary to this JSON file.')
    # Without object-aware planning the car would not brake for traffic or red lights.
    parser.set_defaults(npcs=0)
    args = parser.parse_args()

    with CarlaSession(args, autopilot=False) as session:
        LD_module = LaneDetection()
        obj_detection = ObjectDetection()
        LatC_module = LateralController(car_position=tuple(LD_module.car_position.astype(float)),
                                        metres_per_pixel=LD_module.bev_metres_per_pixel()[1])
        gains = {name: value for name, value in (('KP', args.kp), ('KI', args.ki), ('KD', args.kd),
                                                 ('max_error_sum', args.max_error_sum)) if value is not None}
        LongC_module = LongitudinalController(**gains)
        world_map = session.world.get_map()
        lane_log = LaneKeepingLog()
        speed_log = SpeedLog()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        session.attach_figure(fig)
        plt.ion()
        plt.show()

        control = carla.VehicleControl()
        step = 0
        while session.running(step):
            front = session.tick(None if args.manual else control)
            speed = session.speed_mps
            speed_kmh = 3.6 * speed
            detections = obj_detection.detect(front)

            # lane detection
            lane1, lane2 = LD_module.lane_detection(front, detections.obstacle_mask())

            # waypoint and target_speed prediction
            waypoints: npt.NDArray[np.float64] | None = None
            if args.path == 'ground-truth':
                waypoints = ground_truth_waypoints(world_map, session.vehicle, LD_module)
            elif lane1 is not None and lane2 is not None:
                waypoints = waypoint_prediction(lane1, lane2)
            if args.target == 'steps':
                target_speed = step_target(step, args.step_levels, args.step_period)
            elif waypoints is not None:
                target_speed = target_speed_prediction(waypoints, max_speed=args.max_speed, exp_constant=4.5)
            else:
                target_speed = 0.0

            # control
            steer = 0.0 if waypoints is None else LatC_module.stanley(waypoints, speed)
            gas, brake = LongC_module.control(speed_kmh, target_speed)
            control = carla.VehicleControl(throttle=gas, steer=steer, brake=brake)
            lane_log.record(world_map, session.vehicle, steer, speed)
            speed_log.record(speed_kmh, target_speed, gas, brake)

            # output and plot figure
            if session.should_draw(step):
                print(f'step {step}: steer {steer:+0.2f} gas {gas:0.2f} brake {brake:0.2f} '
                      f'speed {speed_kmh:4.1f} km/h target speed {target_speed:4.1f} km/h')
                title = (f'step {step}, {args.path} path, Kp={LongC_module.KP:g} Ki={LongC_module.KI:g} '
                         f'Kd={LongC_module.KD:g}')
                plot_step(axes, LD_module, LongC_module, front, (lane1, lane2), waypoints, speed_kmh,
                          target_speed, step, title)
                session.after_draw(step)
            step += 1

        session.save_final()
        summary = {**speed_log.summary(), **{f'lane_{k}': v for k, v in lane_log.summary().items()}}
        print('speed tracking:', json.dumps(summary))
        if args.log is not None:
            with open(args.log, 'w') as f:
                json.dump({'args': {k: str(v) for k, v in vars(args).items()}, **summary,
                           'speed_kmh': speed_log.speed_kmh, 'target_kmh': speed_log.target_kmh}, f, indent=1)


if __name__ == '__main__':
    main()
