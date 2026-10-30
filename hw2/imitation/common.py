"""
Shared constants and helpers for the imitation learning experiments (unit 4C): where the data lives, how far ahead
the waypoint labels reach, and how a camera frame becomes the RGB or BEV network input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

# The modular pipeline lives in a folder with a space in its name, so it cannot be a package; its modules are imported
# by putting the folder on the path, as its own tests do.
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / '02_modular_code _template'
sys.path.insert(0, str(TEMPLATE_DIR))

from lane_detection import LaneDetection  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'hw2_imitation'
MODEL_DIR = DATA_DIR / 'models'

# Future waypoints: the ego position every WAYPOINT_STEP_INTERVAL simulation steps (0.5 s at 20 Hz), NUM_WAYPOINTS
# times, i.e. a 2 s horizon.
WAYPOINT_STEP_INTERVAL = 10
NUM_WAYPOINTS = 4

INPUT_TYPES = ('rgb', 'bev')
OUTPUT_TYPES = ('control', 'waypoints')


def network_input(front_rgb: npt.NDArray[np.uint8], input_type: str, lane_detector: LaneDetection) -> npt.NDArray[np.uint8]:
    """Turn a front camera frame into the image the network sees.

    Parameters
    ----------
    front_rgb : (ImageHeight x ImageWidth x 3) array
        RGB front camera frame, 240 x 320.
    input_type : str
        "rgb" for the camera frame itself, "bev" for its bird's-eye view from LaneDetection.front2bev.
    lane_detector : LaneDetection
        Provides the homography for the BEV.

    Returns
    -------
    (ImageHeight x ImageWidth x 3) array
        RGB network input; the BEV has the same 240 x 320 size as the camera frame.
    """
    if input_type == 'rgb':
        return front_rgb
    if input_type == 'bev':
        return lane_detector.front2bev(front_rgb)
    raise ValueError(f'unknown input type {input_type!r}; expected one of {INPUT_TYPES}')


def future_waypoints(poses: npt.NDArray[np.float64], step: int) -> npt.NDArray[np.float64]:
    """Future ego positions in the vehicle frame at `step`.

    Parameters
    ----------
    poses : (NumSteps x 3) array
        World (x, y, yaw in degrees) of the ego vehicle at every simulation step of an episode.
    step : int
        Step whose vehicle frame the waypoints are expressed in; step + NUM_WAYPOINTS * WAYPOINT_STEP_INTERVAL must
        be a valid index.

    Returns
    -------
    (NumWaypoints x 2) array
        (forward, right) offsets in metres of the positions WAYPOINT_STEP_INTERVAL, 2 * WAYPOINT_STEP_INTERVAL, ...
        steps later.
    """
    x, y, yaw_deg = poses[step]
    yaw = np.radians(yaw_deg)
    ahead = poses[step + WAYPOINT_STEP_INTERVAL * np.arange(1, NUM_WAYPOINTS + 1), :2] - (x, y)
    # CARLA's world frame is left-handed (z up, y to the right of x), so rotating by -yaw gives forward/right directly.
    forward = ahead[:, 0] * np.cos(yaw) + ahead[:, 1] * np.sin(yaw)
    right = -ahead[:, 0] * np.sin(yaw) + ahead[:, 1] * np.cos(yaw)
    return np.stack([forward, right], axis=1)
