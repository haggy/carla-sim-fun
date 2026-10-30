import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev
from scipy.optimize import minimize

from lane_detection import Spline


def normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Scale every column vector to unit length.

    Parameters
    ----------
    v : (2 x NumVectors) array
        Column vectors.

    Returns
    -------
    (2 x NumVectors) array
        Unit column vectors; the small epsilon keeps zero-length vectors finite.
    """
    norm = np.linalg.norm(v, axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])


def curvature(waypoints: npt.NDArray[np.float64]) -> float:
    """Second term of the smoothing objective: summed cosines of the turning angles along the path.

    For consecutive path segments d_n = x_{n+1} - x_n, the normalised dot product d_n . d_{n-1} / (|d_n| |d_{n-1}|)
    is the cosine of the angle the path turns at waypoint n: 1 when it continues straight, 0 at a right angle and
    -1 when it doubles back. The sum therefore peaks at N - 2 for a straight line, and maximising it penalises
    every bend, whatever the segment lengths.

    Parameters
    ----------
    waypoints : (2 x NumWaypoints) array
        (x, y) waypoints, ordered along the path.

    Returns
    -------
    float
        Sum over the NumWaypoints - 2 interior waypoints of the cosine of the turning angle, in [-(N-2), N-2].
    """
    segments = normalize(np.diff(waypoints, axis=1))
    return float(np.sum(segments[:, 1:] * segments[:, :-1]))


def smoothing_objective(
    waypoints: npt.NDArray[np.float64],
    waypoints_center: npt.NDArray[np.float64],
    weight_curvature: float = 40,
) -> float:
    """Objective for path smoothing: stay close to the centre waypoints, but turn as little as possible.

    The template uses the mean instead of the assignment's sum of squared distances, so with 2 x NumWaypoints
    coordinates weight_curvature corresponds to beta / (2 * NumWaypoints) in the assignment's formula.

    Parameters
    ----------
    waypoints : (2 * NumWaypoints,) array
        Candidate waypoints, flattened as [x_1..x_N, y_1..y_N].
    waypoints_center : (2 * NumWaypoints,) array
        Centre waypoints from the "center" prediction, flattened the same way.
    weight_curvature : float
        Weight of the curvature term.

    Returns
    -------
    float
        Objective value to minimise.
    """
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(
    roadside1_spline: Spline,
    roadside2_spline: Spline,
    num_waypoints: int = 6,
    way_type: str = "smooth",
) -> npt.NDArray[np.float64]:
    """Predict waypoints along the ego lane from its two lane boundary splines.

    Both boundaries are sampled at the same equidistant spline parameters, starting at 0 (the boundary point nearest
    the car). Each spline is parametrised by its own normalised chord length, so equal parameters mean equal
    fractions of each boundary's detected length, not equal distances ahead of the car.

    Parameters
    ----------
    roadside1_spline : Spline
        Left lane boundary from LaneDetection.lane_detection.
    roadside2_spline : Spline
        Right lane boundary from LaneDetection.lane_detection.
    num_waypoints : int
        Number of waypoints, including the one at spline parameter 0.
    way_type : str
        "center" for the midpoints between the boundaries, "smooth" for waypoints that minimise
        smoothing_objective starting from the centre waypoints.

    Returns
    -------
    (2 x NumWaypoints) array
        (x, y) waypoints in cut_gray coordinates (x in BEV pixels, y in rows ahead of the car), nearest first.
    """
    t = np.linspace(0.0, 1.0, num_waypoints)
    roadside1_points = np.array(splev(t, roadside1_spline))
    roadside2_points = np.array(splev(t, roadside2_spline))
    way_points_center: npt.NDArray[np.float64] = (roadside1_points + roadside2_points) / 2.0

    if way_type == "center":
        return way_points_center

    elif way_type == "smooth":
        # optimization
        way_points = minimize(smoothing_objective,
                              way_points_center.flatten(),
                              args=(way_points_center.flatten(),))["x"]
        return way_points.reshape(2, -1)

    raise ValueError(f'unknown way_type {way_type!r}; expected "center" or "smooth"')


def target_speed_prediction(
    waypoints: npt.NDArray[np.float64],
    num_waypoints_used: int = 5,
    max_speed: float = 60,
    exp_constant: float = 4.5,
    offset_speed: float = 30,
) -> float:
    """Target speed for the predicted path: max_speed on a straight path, decaying towards offset_speed in bends.

    Implements assignment eq. 2, v = (v_max - v_min) * exp(-K_v * |N - 2 - curvature|) + v_min. curvature() sums
    the cosines of the N - 2 turning angles, so N - 2 - curvature is 0 for a straight path and grows with every
    bend; for small angles each turn of theta contributes 1 - cos(theta), about theta^2 / 2.

    Parameters
    ----------
    waypoints : (2 x NumWaypoints) array
        (x, y) waypoints from waypoint_prediction, nearest first.
    num_waypoints_used : int
        N, the number of leading waypoints the curvature is measured over.
    max_speed : float
        v_max, the target speed on a straight path, in km/h.
    exp_constant : float
        K_v, how quickly the target speed drops with curvature.
    offset_speed : float
        v_min, the speed approached in the sharpest bends, in km/h.

    Returns
    -------
    float
        Target speed in km/h, in (offset_speed, max_speed].
    """
    used = waypoints[:, :num_waypoints_used]
    bend = abs(used.shape[1] - 2 - curvature(used))
    return float((max_speed - offset_speed) * np.exp(-exp_constant * bend) + offset_speed)
