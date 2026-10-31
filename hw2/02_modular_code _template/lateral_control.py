# The leaderboard kit may run on Python 3.7, where annotations must not be evaluated at definition time.
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class LateralController:
    """Lateral control with the Stanley controller (assignment eq. 3) and steering damping (eq. 4).

    Waypoints are in the lane detector's cut coordinates: x in BEV pixels (right positive), y in rows ahead of the
    car, with the car at car_position heading +y. Steering angles are in radians, positive to the right, which
    matches CARLA's steer sign.
    """

    def __init__(
        self,
        gain_constant: float = 3,
        damping_constant: float = 0.6,
        car_position: tuple[float, float] = (160.0, 0.0),
        metres_per_pixel: float = 0.075,
        max_steering_angle: float = 0.4,
        full_lock_angle: float = np.radians(70.0),
    ) -> None:
        """Set the controller parameters.

        Parameters
        ----------
        gain_constant : float
            k in eq. 3, in 1/s: how strongly a cross-track error of 1 m is corrected at 1 m/s. Tuned in unit 3B
            on Town01 with the ground-truth path.
        damping_constant : float
            D in eq. 4, in [0, 1): the fraction of the change from the previous steering angle that is held back.
            0 disables damping. Tuned in unit 3C on Town01.
        car_position : tuple[float, float]
            (x, y) of the car in cut coordinates, LaneDetection.car_position.
        metres_per_pixel : float
            Ground size of a cut/BEV pixel (LaneDetection.bev_metres_per_pixel), so the cross-track error is in
            metres to match the speed in m/s.
        max_steering_angle : float
            Largest wheel angle the controller commands, in radians.
        full_lock_angle : float
            Wheel angle at CARLA steer = 1, in radians (70 deg for the Lincoln MKZ ego vehicle).
        """
        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.car_position = np.asarray(car_position, dtype=np.float64)
        self.metres_per_pixel = metres_per_pixel
        self.max_steering_angle = max_steering_angle
        self.full_lock_angle = full_lock_angle
        self.previous_steering_angle = 0.0

    def stanley(self, waypoints: npt.NDArray[np.float64], speed: float) -> float:
        """One step of the Stanley controller with damping.

        Parameters
        ----------
        waypoints : (2 x NumWaypoints) array
            (x, y) waypoints from waypoint_prediction, nearest first; at least two.
        speed : float
            Vehicle speed in m/s.

        Returns
        -------
        float
            CARLA steer command in [-1, 1]: the wheel angle, clipped to max_steering_angle, as a fraction of
            full_lock_angle.
        """
        # Orientation error: heading of the first path segment relative to the car's +y axis.
        segment = waypoints[:, 1] - waypoints[:, 0]
        orientation_error = np.arctan2(segment[0], segment[1])

        # Cross-track error at the reference point, positive when the path lies to the right of the car.
        cross_track_error = (waypoints[0, 0] - self.car_position[0]) * self.metres_per_pixel

        stanley_angle = orientation_error + np.arctan(self.gain_constant * cross_track_error / (speed + 1e-3))

        # Eq. 4 is a first-order low-pass filter: delta(t) = (1 - D) delta_SC(t) + D delta(t - 1), so each step moves
        # only 1 - D of the way to the new Stanley angle, with a time constant of -dt / ln(D).
        steering_angle = stanley_angle - self.damping_constant * (stanley_angle - self.previous_steering_angle)

        steering_angle = float(np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle))
        self.previous_steering_angle = steering_angle
        return steering_angle / self.full_lock_angle
