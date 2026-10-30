"""
Closed-loop evaluation of the imitation learning policies (unit 4C) on Town01, next to the modular pipeline.

Each policy drives the same spawn points (seeds below the data collection seeds) with no traffic and lights ignored,
the setting the expert data was collected in. Lane keeping is judged against the map as in test_lateral_control.py,
and collisions are counted with a collision sensor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import carla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

from common import MODEL_DIR, NUM_WAYPOINTS, WAYPOINT_STEP_INTERVAL, LaneDetection, network_input
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
from network import DrivingPolicy
from object_detection import ObjectDetection
from sim_session import FIXED_DELTA_SECONDS, CarlaSession, build_arg_parser
from test_lateral_control import LaneKeepingLog
from waypoint_prediction import target_speed_prediction, waypoint_prediction

# The run ends early once the car has stood still this long after the start, e.g. against a wall.
STUCK_STEPS = 200
# Speed of the modular pipeline's straight-road target; Town01's limit, which the expert data was also driven at.
MODULAR_MAX_SPEED_KMH = 30.0


class Agent:
    """One driving policy: an imitation learning network or the modular pipeline."""

    def __init__(self, name: str, lane_detector: LaneDetection, device: torch.device) -> None:
        """Load the policy.

        Parameters
        ----------
        name : str
            "modular" for the modular pipeline, otherwise a model name "<input>_<output>" in MODEL_DIR.
        lane_detector : LaneDetection
            Shared lane detector, for the BEV input and the waypoint coordinate conversions.
        device : torch.device
            Device for the network.
        """
        self.name = name
        self.lane_detector = lane_detector
        self.device = device
        self.lateral = LateralController(car_position=tuple(lane_detector.car_position.astype(float)),
                                         metres_per_pixel=lane_detector.bev_metres_per_pixel()[1])
        self.longitudinal = LongitudinalController()
        self.model: DrivingPolicy | None = None
        self.input_type = 'rgb'
        if name == 'modular':
            self.object_detector = ObjectDetection()
        else:
            self.model, self.input_type = DrivingPolicy.load(MODEL_DIR / f'{name}.pth', device)

    def act(self, front: npt.NDArray[np.uint8], speed_mps: float) -> tuple[carla.VehicleControl, npt.NDArray[np.float64] | None]:
        """Control for one frame.

        Parameters
        ----------
        front : (ImageHeight x ImageWidth x 3) array
            RGB front camera frame.
        speed_mps : float
            Vehicle speed in m/s.

        Returns
        -------
        tuple[carla.VehicleControl, (2 x NumWaypoints) array or None]
            The command, and the waypoints it followed in cut_gray coordinates (None for end-to-end control or when
            the modular pipeline has no path).
        """
        if self.model is None:
            return self._act_modular(front, speed_mps)

        image = network_input(front, self.input_type, self.lane_detector)
        with torch.no_grad():
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(self.device)
            speed_tensor = torch.tensor([speed_mps], dtype=torch.float32, device=self.device)
            prediction = self.model(image_tensor, speed_tensor)[0].float().cpu().numpy()

        if self.model.output_type == 'control':
            steer, throttle, brake = (float(v) for v in prediction)
            # Regression blurs "gas" and "brake" into small values of both; the stronger one wins.
            if brake > throttle:
                throttle = 0.0
            else:
                brake = 0.0
            return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake), None

        # Predicted (forward, right) metres -> the lane detector's cut coordinates, where Stanley works.
        waypoints = self.lane_detector.bev_to_cut(self.lane_detector.vehicle_to_bev_pixels(prediction.T.astype(np.float64)))
        horizon_s = NUM_WAYPOINTS * WAYPOINT_STEP_INTERVAL * FIXED_DELTA_SECONDS
        path_length_m = float(np.sum(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], prediction]), axis=0), axis=1)))
        target_kmh = 3.6 * path_length_m / horizon_s
        return self._command(waypoints, speed_mps, target_kmh), waypoints

    def _act_modular(self, front: npt.NDArray[np.uint8], speed_mps: float) -> tuple[carla.VehicleControl, npt.NDArray[np.float64] | None]:
        """Control from the modular pipeline (lane detection, waypoints, target speed, Stanley, PID).

        Parameters
        ----------
        front : (ImageHeight x ImageWidth x 3) array
            RGB front camera frame.
        speed_mps : float
            Vehicle speed in m/s.

        Returns
        -------
        tuple[carla.VehicleControl, (2 x NumWaypoints) array or None]
            See act().
        """
        detections = self.object_detector.detect(front)
        lane1, lane2 = self.lane_detector.lane_detection(front, detections.obstacle_mask())
        if lane1 is None or lane2 is None:
            return carla.VehicleControl(), None
        waypoints = waypoint_prediction(lane1, lane2)
        target_kmh = target_speed_prediction(waypoints, max_speed=MODULAR_MAX_SPEED_KMH,
                                             offset_speed=MODULAR_MAX_SPEED_KMH / 2)
        return self._command(waypoints, speed_mps, target_kmh), waypoints

    def _command(self, waypoints: npt.NDArray[np.float64], speed_mps: float, target_kmh: float) -> carla.VehicleControl:
        """Stanley steering and PID gas/brake towards a path and a target speed.

        Parameters
        ----------
        waypoints : (2 x NumWaypoints) array
            Path in cut_gray coordinates.
        speed_mps : float
            Vehicle speed in m/s.
        target_kmh : float
            Target speed in km/h.

        Returns
        -------
        carla.VehicleControl
            Steer, throttle and brake.
        """
        steer = self.lateral.stanley(waypoints, speed_mps)
        gas, brake = self.longitudinal.control(3.6 * speed_mps, target_kmh)
        return carla.VehicleControl(steer=steer, throttle=gas, brake=brake)


def plot_step(axes: npt.NDArray[np.object_], agent: Agent, front: npt.NDArray[np.uint8],
              waypoints: npt.NDArray[np.float64] | None, title: str) -> None:
    """Draw the camera frame with the followed path, and the network input.

    Parameters
    ----------
    axes : (2,) array of matplotlib Axes
        Axes for the camera frame and the network input.
    agent : Agent
        Policy being evaluated.
    front : (ImageHeight x ImageWidth x 3) array
        RGB front camera frame.
    waypoints : (2 x NumWaypoints) array or None
        Followed path in cut_gray coordinates.
    title : str
        Title of the camera frame.
    """
    height, width = front.shape[:2]
    for ax in axes:
        ax.clear()
        ax.axis('off')
    axes[0].imshow(front)
    axes[1].imshow(network_input(front, agent.input_type, agent.lane_detector))
    if waypoints is not None:
        for ax, to_pixels in ((axes[0], agent.lane_detector.cut_to_front_view),
                              (axes[1], agent.lane_detector.cut_to_front_view if agent.input_type == 'rgb'
                               else agent.lane_detector.cut_to_bev_pixels)):
            u, v = to_pixels(waypoints)
            ax.plot(u, v, color='lime', marker='o', markersize=4, linewidth=1)
            ax.set_xlim(0, width - 1)
            ax.set_ylim(height - 1, 0)
    axes[0].set_title(title, fontsize=9)
    axes[1].set_title(f'network input: {agent.input_type}', fontsize=9)
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def evaluate_run(args: argparse.Namespace, agent: Agent, seed: int) -> dict[str, float]:
    """Drive one episode with an agent.

    Parameters
    ----------
    args : argparse.Namespace
        Options from build_arg_parser plus out; the final figure and --snapshot-every snapshots go to
        <out folder>/closed_loop/<agent>/.
    agent : Agent
        Policy to evaluate.
    seed : int
        Spawn point seed.

    Returns
    -------
    dict[str, float]
        Lane keeping summary (LaneKeepingLog.summary), plus collisions, mean_speed_kmh and stuck (1 if the run ended
        early because the car stood still).
    """
    args.seed = seed
    run_dir = args.out.parent / 'closed_loop' / agent.name
    args.save = run_dir / f'seed{seed}_final.png'
    args.snapshot_dir = run_dir / f'seed{seed}'
    log = LaneKeepingLog()
    collisions: list[carla.CollisionEvent] = []
    speeds: list[float] = []
    stuck = 0
    with CarlaSession(args, autopilot=False) as session:
        blueprint = session.world.get_blueprint_library().find('sensor.other.collision')
        sensor = session.world.spawn_actor(blueprint, carla.Transform(), attach_to=session.vehicle)
        sensor.listen(collisions.append)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            session.attach_figure(fig)
            plt.ion()
            plt.show()
            world_map = session.world.get_map()
            agent.lateral.previous_steering_angle = 0.0
            agent.longitudinal = LongitudinalController()

            control = carla.VehicleControl()
            standing = 0
            step = 0
            while session.running(step):
                front = session.tick(control)
                speed = session.speed_mps
                control, waypoints = agent.act(front, speed)
                log.record(world_map, session.vehicle, control.steer, speed)
                speeds.append(speed)
                standing = standing + 1 if speed < 0.1 else 0
                if step > 100 and standing >= STUCK_STEPS:
                    stuck = 1
                    break
                if session.should_draw(step):
                    plot_step(axes, agent, front, waypoints,
                              f'{agent.name}, seed {seed}, step {step}, {3.6 * speed:.0f} km/h, '
                              f'collisions {len(collisions)}')
                    session.after_draw(step)
                step += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            session.save_final()
            plt.close(fig)
        finally:
            sensor.stop()
            sensor.destroy()
    # One impact fires an event every tick while in contact; count impacts at least 1 s apart.
    frames = sorted(event.frame for event in collisions)
    impacts = sum(1 for i, frame in enumerate(frames) if i == 0 or frame - frames[i - 1] > 20)
    return {**log.summary(), 'collisions': impacts, 'mean_speed_kmh': round(3.6 * float(np.mean(speeds)), 1),
            'stuck': stuck}


def main() -> None:
    """Evaluate the requested agents on the same spawn points and write a JSON summary."""
    parser = build_arg_parser('Closed-loop evaluation of the imitation learning policies.', 'unused')
    parser.add_argument('--agents', nargs='+',
                        default=['rgb_control', 'bev_control', 'rgb_waypoints', 'bev_waypoints', 'modular'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', type=Path, default=MODEL_DIR / 'closed_loop.json')
    parser.set_defaults(npcs=0, steps=1000, snapshot_every=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lane_detector = LaneDetection()
    results: dict[str, dict[str, dict[str, float]]] = {}
    if args.out.exists():
        results = json.loads(args.out.read_text())
    for name in args.agents:
        agent = Agent(name, lane_detector, device)
        results.setdefault(name, {})
        for seed in args.seeds:
            results[name][str(seed)] = evaluate_run(args, agent, seed)
            print(name, seed, json.dumps(results[name][str(seed)]), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
