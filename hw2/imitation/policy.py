"""
Driving policies that turn one front camera frame and the current speed into a CARLA control: an imitation learning
network (unit 4C) or the modular pipeline (lane detection, waypoints, target speed, Stanley, PID). Shared by the
closed-loop evaluation and the leaderboard agent, so both drive with the same code.
"""
from __future__ import annotations

from pathlib import Path

import carla
import numpy as np
import numpy.typing as npt
import torch

from common import MODEL_DIR, NUM_WAYPOINTS, SIMULATION_STEP_S, WAYPOINT_STEP_INTERVAL, LaneDetection, network_input
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
from network import DrivingPolicy
from waypoint_prediction import target_speed_prediction, waypoint_prediction

# Speed of the modular pipeline's straight-road target; Town01's limit, which the expert data was also driven at.
MODULAR_MAX_SPEED_KMH = 30.0


class Agent:
    """One driving policy: an imitation learning network or the modular pipeline."""

    def __init__(self, name: str, lane_detector: LaneDetection, device: torch.device,
                 model_path: Path | None = None) -> None:
        """Load the policy.

        Parameters
        ----------
        name : str
            "modular" for the modular pipeline, otherwise a model name such as "bev_control_weather".
        lane_detector : LaneDetection
            Shared lane detector, for the BEV input and the waypoint coordinate conversions.
        device : torch.device
            Device for the network.
        model_path : Path or None
            Weights of a network policy; None for <name>.pth in MODEL_DIR.
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
            # Imported here because only this policy needs ultralytics, which does not support Python 3.7.
            from object_detection import ObjectDetection
            self.object_detector = ObjectDetection()
        else:
            self.model, self.input_type = DrivingPolicy.load(model_path or MODEL_DIR / f'{name}.pth', device)

    def reset(self) -> None:
        """Clear the controller state before a new drive."""
        self.lateral.previous_steering_angle = 0.0
        self.longitudinal = LongitudinalController()

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
        horizon_s = NUM_WAYPOINTS * WAYPOINT_STEP_INTERVAL * SIMULATION_STEP_S
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
