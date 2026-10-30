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
    """
    ##### TODO ##### (unit 2B)
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args:
        waypoints [2, num_waypoints] !!!!!
    """
    raise NotImplementedError('curvature is implemented in unit 2B')


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
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
        "center" for the midpoints between the boundaries, "smooth" for the path-smoothed waypoints (unit 2B).

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
        ##### TODO ##### (unit 2B)
        raise NotImplementedError('path smoothing is implemented in unit 2B')

    raise ValueError(f'unknown way_type {way_type!r}; expected "center" or "smooth"')


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO ##### (unit 2C)
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)

    output:
        target_speed (float)
    '''
    raise NotImplementedError('target speed prediction is implemented in unit 2C')
