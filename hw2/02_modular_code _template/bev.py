"""
Scene bird's-eye view built from the front camera: homography for the ground, YOLO26 masks for obstacles.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev

from lane_detection import LaneDetection, Spline
from object_detection import DetectionResult

# Typical footprint length along the line of sight, in metres, per COCO class id. The camera only sees an obstacle's
# near face, so its extent away from the camera has to be assumed.
DEFAULT_CLASS_LENGTHS_M: dict[int, float] = {0: 0.6, 1: 1.8, 2: 4.5, 3: 2.0, 5: 11.0, 7: 7.0}

# Distance, in metres, from where an obstacle's lowest visible pixels touch the road to its near face. For a car seen
# from behind, the lowest pixels are the rear tyres, which sit behind the overhanging bumper; 0.9 m was measured
# against CARLA ground truth in Town01 (median near-range distance bias). People and bikes touch the road at their
# near face.
DEFAULT_CONTACT_SETBACK_M: dict[int, float] = {2: 0.9, 5: 0.9, 7: 0.9}

# RGB colours per COCO class id for the rendered footprints.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 60, 60), 1: (255, 160, 0), 2: (0, 200, 255), 3: (255, 160, 0), 5: (160, 90, 255), 7: (120, 120, 255),
}


@dataclass
class BEVObstacle:
    """One detected obstacle placed on the ground plane.

    Attributes
    ----------
    class_id : int
        COCO class id.
    class_name : str
        COCO class name.
    confidence : float
        Detection confidence in [0, 1].
    near_distance_m : float
        Forward distance from the vehicle origin to the obstacle's nearest visible ground contact, in metres.
    lateral_extent_m : tuple[float, float]
        (left, right) lateral extent of the ground contact, in metres (CARLA frame: y right).
    length_m : float
        Assumed extent away from the camera, from the class.
    truncated : bool
        True if the mask touches the bottom image edge, so the true contact is below the frame and closer than
        near_distance_m.
    """
    class_id: int
    class_name: str
    confidence: float
    near_distance_m: float
    lateral_extent_m: tuple[float, float]
    length_m: float
    truncated: bool

    def footprint_m(self) -> npt.NDArray[np.float64]:
        """Corners of the assumed ground footprint.

        Returns
        -------
        (2 x 4) array
            (x forward, y right) corners in metres: near-left, near-right, far-right, far-left.
        """
        x0, x1 = self.near_distance_m, self.near_distance_m + self.length_m
        y0, y1 = self.lateral_extent_m
        return np.array([[x0, x0, x1, x1], [y0, y1, y1, y0]])


class SceneBEV:
    """Builds a metric bird's-eye view of the scene from one front camera frame.

    The ground is warped with the flat-ground homography. Obstacles are not on the ground, so the same warp
    stretches them into long streaks; instead, each obstacle is localised from the one part of it that does lie
    on the ground, the bottom of its segmentation mask, and drawn as a footprint. The streak region is ground
    hidden behind the obstacle and is shown as unknown.
    """

    def __init__(
        self,
        lane_detector: LaneDetection,
        class_lengths_m: dict[int, float] | None = None,
        contact_setback_m: dict[int, float] | None = None,
        contact_band_px: int = 4,
    ) -> None:
        """Share the camera model and BEV grid of a lane detector.

        Parameters
        ----------
        lane_detector : LaneDetection
            Provides the homography, BEV range and coordinate conversions.
        class_lengths_m : dict[int, float] or None
            Footprint length per COCO class id; DEFAULT_CLASS_LENGTHS_M if None.
        contact_setback_m : dict[int, float] or None
            Ground-contact-to-near-face distance per COCO class id; DEFAULT_CONTACT_SETBACK_M if None.
        contact_band_px : int
            Height, in front-view rows, of the mask's bottom band used to measure the contact width.
        """
        self.lane_detector = lane_detector
        self.class_lengths_m = class_lengths_m if class_lengths_m is not None else DEFAULT_CLASS_LENGTHS_M
        self.contact_setback_m = contact_setback_m if contact_setback_m is not None else DEFAULT_CONTACT_SETBACK_M
        self.contact_band_px = contact_band_px

    def localize(self, detections: DetectionResult) -> list[BEVObstacle]:
        """Place every detection on the ground plane.

        Parameters
        ----------
        detections : DetectionResult
            Output of ObjectDetection.detect for the frame.

        Returns
        -------
        list[BEVObstacle]
            One entry per detection whose ground contact is below the horizon.
        """
        obstacles: list[BEVObstacle] = []
        for mask, class_id, confidence in zip(detections.masks, detections.class_ids, detections.confidences):
            rows = np.flatnonzero(mask.any(axis=1))
            if rows.size == 0:
                continue
            bottom = int(rows[-1])
            band_cols = np.flatnonzero(mask[max(bottom - self.contact_band_px, 0):bottom + 1].any(axis=0))

            # The lowest mask pixels are where the obstacle meets the road (tyres, feet), the only part of it that
            # satisfies the flat-ground assumption. The bottom pixel edge (+0.5 rows) and the outer pixel edges
            # (+-0.5 columns) are the actual contact boundary.
            contact_uv = np.array([[band_cols[0] - 0.5, band_cols[-1] + 0.5], [bottom + 0.5, bottom + 0.5]])
            contact_m = self.lane_detector.image_to_vehicle(contact_uv)
            if np.isnan(contact_m).any():
                continue

            obstacles.append(BEVObstacle(
                class_id=int(class_id),
                class_name=detections.class_names.get(int(class_id), str(class_id)),
                confidence=float(confidence),
                near_distance_m=float(contact_m[0].mean()) - self.contact_setback_m.get(int(class_id), 0.0),
                lateral_extent_m=(float(contact_m[1].min()), float(contact_m[1].max())),
                length_m=self.class_lengths_m.get(int(class_id), 2.0),
                truncated=bottom >= mask.shape[0] - 1,
            ))
        return obstacles

    def render(
        self,
        front_view_image: npt.NDArray[np.uint8],
        detections: DetectionResult,
        obstacles: list[BEVObstacle],
        splines: tuple[Spline | None, Spline | None] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Draw the scene BEV: warped ground, occluded ground, obstacle footprints and optional lane splines.

        Parameters
        ----------
        front_view_image : (ImageHeight x ImageWidth x 3) array
            RGB front camera image.
        detections : DetectionResult
            Detections of this frame; their masks mark the ground hidden behind obstacles.
        obstacles : list[BEVObstacle]
            Output of localize for the same detections.
        splines : tuple[Spline or None, Spline or None] or None
            Lane boundaries from LaneDetection.lane_detection, drawn in orange.

        Returns
        -------
        (ImageHeight x ImageWidth x 3) array
            RGB scene BEV on the LaneDetection BEV grid.
        """
        ld = self.lane_detector
        bev: npt.NDArray[np.uint8] = ld.front2bev(front_view_image)

        # Where the warp lands obstacle pixels, the ground itself is hidden from the camera.
        occluded = ld.obstacle_mask_to_bev(detections.obstacle_mask())
        bev[occluded] = (0.35 * bev[occluded] + 0.65 * np.array([60, 60, 60])).astype(np.uint8)

        for obstacle in obstacles:
            corners = ld.vehicle_to_bev_pixels(obstacle.footprint_m()).T.round().astype(np.int32)
            color = CLASS_COLORS.get(obstacle.class_id, (255, 255, 255))
            cv2.fillPoly(bev, [corners], color)
            cv2.polylines(bev, [corners], True, (0, 0, 0), 1)

        if splines is not None:
            t = np.linspace(0.0, 1.0, 40)
            for spline in splines:
                if spline is not None:
                    uv = ld.cut_to_bev_pixels(np.array(splev(t, spline)))
                    cv2.polylines(bev, [uv.T.round().astype(np.int32)], False, (255, 140, 0), 2)

        return bev
