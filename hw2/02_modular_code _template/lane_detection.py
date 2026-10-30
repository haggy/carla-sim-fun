from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.figure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

# B-spline as returned by scipy.interpolate.splprep: (knots, [x coefficients, y coefficients], degree).
Spline = tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]], int]


@dataclass
class LaneDetectionFrame:
    """Intermediate results of one LaneDetection.lane_detection call, kept for visualisation and debugging.

    Attributes
    ----------
    bev : (ImageHeight x ImageWidth x 3) array
        RGB BEV image from front2bev.
    obstacle_mask_bev : (ImageHeight x ImageWidth) array or None
        BEV obstacle mask from obstacle_mask_to_bev, None when no obstacle mask was given.
    gradient_sum : (CutSize x ImageWidth) array
        Thresholded gradients from edge_detection, row 0 nearest the car.
    maxima : (2 x NumMaxima) array
        (x, y) maxima from find_maxima_gradient_rowwise.
    lane_boundary1_points : (NumBoundary1Points x 2) array
        (x, y) maxima assigned to the left lane boundary.
    lane_boundary2_points : (NumBoundary2Points x 2) array
        (x, y) maxima assigned to the right lane boundary.
    lane_boundary1_fitted : bool
        True if the left spline was fitted on this frame, False if the preceding spline was reused.
    lane_boundary2_fitted : bool
        True if the right spline was fitted on this frame, False if the preceding spline was reused.
    """
    bev: npt.NDArray[np.uint8]
    obstacle_mask_bev: npt.NDArray[np.bool_] | None
    gradient_sum: npt.NDArray[np.float64]
    maxima: npt.NDArray[np.int64]
    lane_boundary1_points: npt.NDArray[np.int64]
    lane_boundary2_points: npt.NDArray[np.int64]
    lane_boundary1_fitted: bool
    lane_boundary2_fitted: bool


class LaneDetection:
    """Lane detection module: BEV projection, edge detection, edge-to-boundary assignment and B-spline fitting.

    See __init__ for the parameters.
    """

    def __init__(
        self,
        cut_size: int = 120,
        spline_smoothness: float = 100,
        gradient_threshold: float = 10,
        distance_maxima_gradient: int = 3,
        image_size: tuple[int, int] = (240, 320),
        camera_fov_deg: float = 90.0,
        camera_xyz: tuple[float, float, float] = (0.0, 0.0, 2.0),
        bev_forward_range_m: tuple[float, float] = (4.5, 22.5),
        bev_lateral_range_m: tuple[float, float] = (-12.0, 12.0),
        obstacle_margin_px: int = 3,
        max_assignment_distance_px: float = 100.0,
        lateral_distance_weight: float = 3.0,
        slope_lookback_points: int = 10,
        min_lane_width_px: float = 20.0,
        min_boundary_points: int = 10,
        max_start_attempts: int = 5,
        max_lateral_jump_px: float = 4.0,
        max_lateral_slope: float = 0.3,
    ) -> None:
        """Configure the lane detector and precompute the front-view to BEV homography.

        Camera defaults mirror the 'rgb_front' sensor in performance_benchmark/leaderboard/team_code/test_agent.py.
        The default BEV ranges give square 7.5 cm pixels.

        Parameters
        ----------
        cut_size : int
            Number of BEV rows, counted from the car, that are kept for lane detection.
        spline_smoothness : float
            Smoothing factor `s` passed to scipy.interpolate.splprep: an upper bound on the summed squared pixel
            distance between the boundary points and the spline. Tuned in unit 1F: accuracy is insensitive to it,
            but below ~30 the spline follows the 1 px quantisation of the maxima and turns by 50-400 deg over 9 m;
            from ~100 (about the number of points per boundary) it stops tracking that noise.
        gradient_threshold : float
            Absolute gradients below this value are zeroed during edge detection. Tuned in unit 1F against map
            ground truth: Town01's faint, wet markings need <= 10, while Town10's high-contrast scenes do best at
            12-14; 10 balances the two.
        distance_maxima_gradient : int
            Minimum pixel distance between neighbouring row-wise gradient maxima.
        image_size : tuple[int, int]
            (ImageHeight, ImageWidth) of the front-view camera image, in pixels. The BEV image has the same size.
        camera_fov_deg : float
            Horizontal field of view of the camera, in degrees.
        camera_xyz : tuple[float, float, float]
            Camera position relative to the vehicle origin, in metres (CARLA frame: x forward, y right, z up).
            The camera is assumed to have zero roll, pitch and yaw.
        bev_forward_range_m : tuple[float, float]
            (near, far) distance ahead of the vehicle covered by the BEV image, in metres. The default camera's
            view of the ground is blocked by the ego hood up to ~3.8 m ahead, so the near limit sits past it;
            otherwise the hood's outline is the first edge the lane search meets.
        bev_lateral_range_m : tuple[float, float]
            (left, right) lateral extent covered by the BEV image, in metres. Left is negative.
        obstacle_margin_px : int
            Front-view pixels the obstacle masks are grown by before they are warped to the BEV.
        max_assignment_distance_px : float
            A lane boundary stops growing once its nearest candidate maximum is at least this far away, measured
            with the weighted distance described in _trace_lane_boundary.
        lateral_distance_weight : float
            Factor applied to the sideways offset of a candidate maximum when tracing a lane boundary.
        slope_lookback_points : int
            Number of most recent boundary points used to estimate the boundary's direction while tracing.
        min_lane_width_px : float
            The right lane boundary only takes maxima at least this far right of the left boundary in the same
            row.
        min_boundary_points : int
            A traced boundary with fewer points than this is treated as a false start and retraced from the next
            start candidate.
        max_start_attempts : int
            Maximum number of start points tried per lane boundary.
        max_lateral_jump_px : float
            While tracing, a candidate maximum is rejected outright if its sideways offset from the boundary's
            extrapolated direction exceeds max_lateral_jump_px + max_lateral_slope * (its distance ahead of the
            last boundary point).
        max_lateral_slope : float
            See max_lateral_jump_px.
        """
        # The car sits at the bottom centre of the BEV, which is row 0 after cut_gray flips the rows.
        self.car_position: npt.NDArray[np.int64] = np.array([image_size[1] // 2, 0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old: Spline | None = None
        self.lane_boundary2_old: Spline | None = None
        self.last_frame: LaneDetectionFrame | None = None
        self.image_size = image_size
        self.obstacle_margin_px = obstacle_margin_px
        self.max_assignment_distance_px = max_assignment_distance_px
        self.lateral_distance_weight = lateral_distance_weight
        self.slope_lookback_points = slope_lookback_points
        self.min_lane_width_px = min_lane_width_px
        self.min_boundary_points = min_boundary_points
        self.max_start_attempts = max_start_attempts
        self.max_lateral_jump_px = max_lateral_jump_px
        self.max_lateral_slope = max_lateral_slope
        self.H_bev2im: npt.NDArray[np.float64] = self._bev_to_image_homography(
            image_size, camera_fov_deg, camera_xyz, bev_forward_range_m, bev_lateral_range_m)

        # Warping an all-ones image marks which BEV pixels the camera actually sees. The mask is eroded so
        # pixels next to the black border wedges are also excluded, since their gradients belong to the border.
        seen: npt.NDArray[np.uint8] = self.front2bev(np.ones((*image_size, 3), dtype=np.uint8))[:, :, 0]
        self.bev_valid_mask: npt.NDArray[np.bool_] = cv2.erode(seen, np.ones((5, 5), np.uint8)) > 0


    @staticmethod
    def _bev_to_image_homography(
        image_size: tuple[int, int],
        camera_fov_deg: float,
        camera_xyz: tuple[float, float, float],
        bev_forward_range_m: tuple[float, float],
        bev_lateral_range_m: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Build the homography that maps homogeneous BEV pixels to homogeneous front-view pixels.

        Every point on a flat ground plane (z=0) satisfies p_im ~ K [r1 r2 t] (x, y, 1)^T, so the
        3D projection reduces to a plane-to-plane homography. It is composed as
        BEV pixel -> ground metres -> camera frame -> image pixel.

        Parameters
        ----------
        image_size : tuple[int, int]
            (ImageHeight, ImageWidth) shared by the front-view and BEV images, in pixels.
        camera_fov_deg : float
            Horizontal field of view of the camera, in degrees.
        camera_xyz : tuple[float, float, float]
            Camera position relative to the vehicle origin, in metres (CARLA frame: x forward, y right, z up).
        bev_forward_range_m : tuple[float, float]
            (near, far) distance ahead of the vehicle covered by the BEV image, in metres.
        bev_lateral_range_m : tuple[float, float]
            (left, right) lateral extent covered by the BEV image, in metres.

        Returns
        -------
        (3 x 3) matrix
            Homography H_bev2im such that (u_im, v_im, 1) ~ H_bev2im @ (u_bev, v_bev, 1).
        """
        height, width = image_size
        x_near, x_far = bev_forward_range_m
        y_left, y_right = bev_lateral_range_m
        cam_x, cam_y, cam_z = camera_xyz

        # CARLA cameras are ideal pinholes: no skew or lens distortion, so K is fully set by the FOV.
        focal = width / (2.0 * np.tan(np.deg2rad(camera_fov_deg) / 2.0))
        K = np.array([
            [focal, 0.0,   width / 2.0],
            [0.0,   focal, height / 2.0],
            [0.0,   0.0,   1.0],
        ])

        # Ground (x, y, 1) -> camera (right, down, forward) for a camera with zero roll/pitch/yaw.
        # The constant down component is the camera height, because the ground lies at z=0 below it.
        H_g2cam = np.array([
            [0.0, 1.0, -cam_y],
            [0.0, 0.0,  cam_z],
            [1.0, 0.0, -cam_x],
        ])

        # BEV pixel centres -> ground metres. The top row is the far edge and the bottom row the near edge,
        # so the vehicle sits at the bottom centre the way car_position expects.
        dx = (x_far - x_near) / height
        dy = (y_right - y_left) / width
        H_bev2g = np.array([
            [0.0, -dx, x_far - 0.5 * dx],
            [dy,  0.0, y_left + 0.5 * dy],
            [0.0, 0.0, 1.0],
        ])

        return K @ H_g2cam @ H_bev2g


    def front2bev(self, front_view_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Warp the front-view image onto the ground plane to get a bird's-eye view, assuming the road is flat.

        Parameters
        ----------
        front_view_image : (ImageHeight x ImageWidth x 3) array
            Front camera image; the default camera produces (240 x 320 x 3).

        Returns
        -------
        (ImageHeight x ImageWidth x 3) array
            BEV image with the vehicle at the bottom centre and forward pointing up. Ground outside the
            camera's field of view is black, which makes wedges in the near corners.
        """
        bev_image: npt.NDArray[np.uint8] = self._warp_to_bev(front_view_image, cv2.INTER_LINEAR)
        return bev_image


    def _warp_to_bev(self, front_view: npt.NDArray[np.uint8], interpolation: int) -> npt.NDArray[np.uint8]:
        """Apply the ground-plane homography to any front-view-aligned array.

        Parameters
        ----------
        front_view : (ImageHeight x ImageWidth x ...) array
            Front-view image or mask.
        interpolation : int
            cv2 interpolation flag, e.g. cv2.INTER_LINEAR for images or cv2.INTER_NEAREST for masks.

        Returns
        -------
        (ImageHeight x ImageWidth x ...) array
            BEV-aligned array; pixels outside the camera's field of view are 0.
        """
        height, width = self.image_size

        # H_bev2im already maps destination -> source, so pass WARP_INVERSE_MAP to skip a matrix inversion.
        warped: npt.NDArray[np.uint8] = cv2.warpPerspective(
            front_view,
            self.H_bev2im,
            (width, height),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped


    def obstacle_mask_to_bev(self, obstacle_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Project a front-view obstacle segmentation mask into the BEV.

        Obstacles are not on the ground plane, so the homography smears them away from the camera into long
        streaks. Warping the mask with the same homography as the image produces exactly the same streak, so the
        BEV mask covers every BEV pixel the obstacle's image pixels landed on.

        Parameters
        ----------
        obstacle_mask : (ImageHeight x ImageWidth) array
            True where an obstacle covers the front-view pixel, e.g. DetectionResult.obstacle_mask().

        Returns
        -------
        (ImageHeight x ImageWidth) array
            True where the BEV pixel shows an obstacle.
        """
        mask_u8: npt.NDArray[np.uint8] = obstacle_mask.astype(np.uint8)

        # Segmentation masks stop slightly inside the object outline, yet the strongest gradients sit exactly on
        # that outline. Growing the mask in the front view keeps the margin a constant size in the camera image,
        # so after the warp it scales with distance the same way the obstacle does.
        if self.obstacle_margin_px > 0:
            kernel_size = 2 * self.obstacle_margin_px + 1
            mask_u8 = cv2.dilate(mask_u8, np.ones((kernel_size, kernel_size), np.uint8))

        # Nearest neighbour keeps the mask binary.
        obstacle_mask_bev: npt.NDArray[np.bool_] = self._warp_to_bev(mask_u8, cv2.INTER_NEAREST) > 0
        return obstacle_mask_bev


    def _cut_rows(self, image: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
        """Keep the cut_size BEV rows closest to the car and reorder them so row 0 is nearest the car.

        Parameters
        ----------
        image : (ImageHeight x ImageWidth x ...) array
            Any BEV-aligned array, e.g. the BEV image or the BEV validity mask.

        Returns
        -------
        (CutSize x ImageWidth x ...) array
            Rows nearest the car, in order of increasing distance.
        """
        return image[-self.cut_size:][::-1]


    def cut_gray(self, state_image_full: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        """Crop the BEV image to the region just ahead of the car and convert it to grey scale.

        Parameters
        ----------
        state_image_full : (ImageHeight x ImageWidth x 3) array
            RGB BEV image from front2bev.

        Returns
        -------
        (CutSize x ImageWidth) array
            Grey scale intensities in [0, 255]. Row 0 is the row nearest the car, so the row index grows
            with distance ahead, as find_first_lane_point expects.
        """
        # luma weights, the same ones cv2.COLOR_RGB2GRAY uses
        luma_weights = np.array([0.299, 0.587, 0.114])
        gray_state_image: npt.NDArray[np.float64] = self._cut_rows(state_image_full) @ luma_weights
        return gray_state_image


    def edge_detection(
        self,
        gray_image: npt.NDArray[np.float64],
        obstacle_mask_bev: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute the thresholded absolute gradient magnitude of the grey scale image.

        Parameters
        ----------
        gray_image : (CutSize x ImageWidth) array
            Grey scale image from cut_gray.
        obstacle_mask_bev : (ImageHeight x ImageWidth) array or None
            Full BEV obstacle mask from obstacle_mask_to_bev. Obstacle outlines are strong edges that are not
            lane markings, so gradients under the mask are discarded. None skips obstacle suppression.

        Returns
        -------
        (CutSize x ImageWidth) array
            |dI/dx| + |dI/dy| per pixel. Values below self.gradient_threshold, pixels on or outside the border
            of the camera's field of view in the BEV, and pixels covered by obstacles are set to zero.
        """

        # First array is the gradient vertically (changes vertically), second is horizontal
        grad_rows, grad_cols = np.gradient(gray_image)

        # The L1 norm approach
        gradient_sum: npt.NDArray[np.float64] = np.abs(grad_rows) + np.abs(grad_cols)

        # Deactivate any pixel where the gradient doesn't meet our edge threshold
        gradient_sum[gradient_sum < self.gradient_threshold] = 0.0
        gradient_sum[~self._cut_rows(self.bev_valid_mask)] = 0.0
        if obstacle_mask_bev is not None:
            gradient_sum[self._cut_rows(obstacle_mask_bev)] = 0.0
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Find the local maxima of the gradient image in every row.

        Parameters
        ----------
        gradient_sum : (CutSize x ImageWidth) array
            Thresholded gradient image from edge_detection.

        Returns
        -------
        (2 x NumMaxima) array
            Pixel coordinates of every maximum: row 0 holds the column (x) and row 1 holds the row (y), the
            same (x, y) order used for the lane boundary points.
        """
        cols_per_row: list[npt.NDArray[np.int64]] = [
            find_peaks(row, distance=self.distance_maxima_gradient)[0] for row in gradient_sum
        ]
        rows_per_row: list[npt.NDArray[np.int64]] = [
            np.full(cols.shape[0], row_idx, dtype=np.int64) for row_idx, cols in enumerate(cols_per_row)
        ]
        argmaxima: npt.NDArray[np.int64] = np.stack([np.concatenate(cols_per_row), np.concatenate(rows_per_row)])
        return argmaxima


    def find_first_lane_point(
        self,
        gradient_sum: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], bool]:
        """Find the start points of the left and right lane boundaries in the rows closest to the car.

        Rows are scanned outward from the car until one has a maximum on each side of the car. The innermost
        maximum on each side is taken, so the two boundaries of the ego lane are chosen rather than two edges on
        the same side (e.g. both edges of one painted line, or a lane line and the curb beyond it).

        Parameters
        ----------
        gradient_sum : (CutSize x ImageWidth) array
            Thresholded gradient image from edge_detection, row 0 nearest the car.

        Returns
        -------
        lane_boundary1_startpoint : (1 x 2) array
            (x, y) start of the left lane boundary.
        lane_boundary2_startpoint : (1 x 2) array
            (x, y) start of the right lane boundary.
        lanes_found : bool
            False if no row holds any maximum; both start points are then (0, 0).
        """
        car_x = int(self.car_position[0])
        width = gradient_sum.shape[1]
        one_sided: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]] | None = None

        for row in range(gradient_sum.shape[0]):
            argmaxima = find_peaks(gradient_sum[row], distance=self.distance_maxima_gradient)[0]
            left = argmaxima[argmaxima < car_x]
            right = argmaxima[argmaxima >= car_x]

            if left.size and right.size:
                return np.array([[left.max(), row]]), np.array([[right.min(), row]]), True

            # Keep the nearest row that only shows one side, in case the other boundary never appears. The missing
            # boundary is then parked on the image border, as in the original template.
            if one_sided is None and left.size:
                one_sided = np.array([[left.max(), row]]), np.array([[width - 1, row]])
            elif one_sided is None and right.size:
                one_sided = np.array([[0, row]]), np.array([[right.min(), row]])

        if one_sided is not None:
            return one_sided[0], one_sided[1], True
        return np.array([[0, 0]]), np.array([[0, 0]]), False


    def _trace_lane_boundary(
        self,
        start_point: npt.NDArray[np.int64],
        maxima: npt.NDArray[np.int64],
        available: npt.NDArray[np.bool_],
        left_boundary: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Grow one lane boundary away from the car by repeatedly appending its nearest unassigned maximum.

        Only maxima further from the car than the last boundary point are candidates, so the boundary cannot fold
        back on itself. Distances are measured relative to where the boundary is heading: the lateral offset is
        taken from the straight-line extrapolation of the recent boundary points and weighted by
        lateral_distance_weight. In the BEV a dashed line's next dash is ~50-80 px ahead while the neighbouring
        lane line is ~47 px to the side, so with an unweighted distance the gap between dashes would make the
        boundary jump lanes.

        Parameters
        ----------
        start_point : (1 x 2) array
            (x, y) start point from find_first_lane_point.
        maxima : (2 x NumMaxima) array
            (x, y) maxima from find_maxima_gradient_rowwise.
        available : (NumMaxima,) array
            True for maxima not yet assigned to a boundary. Updated in place, so a maximum is never assigned to
            both boundaries.
        left_boundary : (NumLeftPoints x 2) array or None
            Already traced left boundary when tracing the right one. Within the left boundary's rows, the right
            boundary only takes maxima at least min_lane_width_px to its right. Painted lines are often doubled
            (e.g. the yellow centre line) and each paint stripe gives its own maxima, and diagonal markings lead
            across the lane, so without this the right boundary can cross over or ride the left one's line.

        Returns
        -------
        (NumBoundaryPoints x 2) array
            (x, y) points of the boundary, starting with start_point and ordered away from the car.
        """
        points: list[npt.NDArray[np.int64]] = [start_point[0]]
        available &= ~((maxima[0] == start_point[0, 0]) & (maxima[1] == start_point[0, 1]))

        allowed: npt.NDArray[np.bool_] = available.copy()
        if left_boundary is not None and left_boundary.shape[0] > 1:
            # Traced boundaries advance strictly row by row, so their rows are sorted and np.interp is valid.
            left_x = np.interp(maxima[1], left_boundary[:, 1], left_boundary[:, 0])
            in_span = (maxima[1] >= left_boundary[0, 1]) & (maxima[1] <= left_boundary[-1, 1])
            allowed &= ~(in_span & (maxima[0] - left_x < self.min_lane_width_px))

        while True:
            last = points[-1]
            candidates = np.flatnonzero(available & allowed & (maxima[1] > last[1]))
            if candidates.size == 0:
                break

            anchor = points[max(len(points) - self.slope_lookback_points, 0)]
            slope = (last[0] - anchor[0]) / (last[1] - anchor[1]) if last[1] != anchor[1] else 0.0

            forward = maxima[1, candidates] - last[1]
            lateral = maxima[0, candidates] - (last[0] + slope * forward)
            distance = np.hypot(self.lateral_distance_weight * lateral, forward)

            # Lane lines change direction slowly, while texture, puddle and shadow edges scatter in every direction.
            # The weighted distance alone still allows a sideways hop when the candidate sits in the next row, so
            # offsets beyond a cone around the extrapolated direction are excluded outright.
            distance[np.abs(lateral) > self.max_lateral_jump_px + self.max_lateral_slope * forward] = np.inf

            nearest = int(np.argmin(distance))
            if distance[nearest] >= self.max_assignment_distance_px:
                break

            points.append(maxima[:, candidates[nearest]])
            available[candidates[nearest]] = False

        return np.array(points, dtype=np.int64)


    def assign_lane_boundary_points(
        self,
        gradient_sum: npt.NDArray[np.float64],
        maxima: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], bool]:
        """Assign the gradient maxima to the left and right lane boundaries of the ego lane.

        Parameters
        ----------
        gradient_sum : (CutSize x ImageWidth) array
            Thresholded gradient image from edge_detection, row 0 nearest the car.
        maxima : (2 x NumMaxima) array
            (x, y) maxima of gradient_sum from find_maxima_gradient_rowwise.

        Returns
        -------
        lane_boundary1_points : (NumBoundary1Points x 2) array
            (x, y) points of the left lane boundary, ordered away from the car.
        lane_boundary2_points : (NumBoundary2Points x 2) array
            (x, y) points of the right lane boundary, ordered away from the car.
        lanes_found : bool
            False if find_first_lane_point found no start point; both boundaries then hold only (0, 0).
        """
        lane_boundary1_points, lane_boundary2_points, lanes_found = self.find_first_lane_point(gradient_sum)
        if not lanes_found:
            return lane_boundary1_points, lane_boundary2_points, False

        car_x = int(self.car_position[0])
        available: npt.NDArray[np.bool_] = np.ones(maxima.shape[1], dtype=bool)
        lane_boundary1_points = self._trace_with_retries(
            lane_boundary1_points, maxima[0] < car_x, maxima, available)
        lane_boundary2_points = self._trace_with_retries(
            lane_boundary2_points, maxima[0] >= car_x, maxima, available, left_boundary=lane_boundary1_points)
        return lane_boundary1_points, lane_boundary2_points, True


    def _trace_with_retries(
        self,
        first_start: npt.NDArray[np.int64],
        on_side: npt.NDArray[np.bool_],
        maxima: npt.NDArray[np.int64],
        available: npt.NDArray[np.bool_],
        left_boundary: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Trace a lane boundary, retrying from other start points when the trace is too short to be a lane line.

        The rows nearest the car only see a few metres to each side, so the innermost maximum there is often a
        short road marking or the end of a dash rather than the lane line itself. Further candidates are the
        side's maxima nearest the car first and innermost within a row; maxima consumed by an earlier attempt are
        skipped because they would only reproduce the same short trace.

        Parameters
        ----------
        first_start : (1 x 2) array
            (x, y) start point from find_first_lane_point, tried first.
        on_side : (NumMaxima,) array
            True for maxima on this boundary's side of the car; these are the retry start candidates.
        maxima : (2 x NumMaxima) array
            (x, y) maxima from find_maxima_gradient_rowwise.
        available : (NumMaxima,) array
            True for maxima not yet assigned to a boundary. Only the kept trace's maxima are marked as used.
        left_boundary : (NumLeftPoints x 2) array or None
            Left boundary to stay clear of when tracing the right boundary, see _trace_lane_boundary.

        Returns
        -------
        (NumBoundaryPoints x 2) array
            (x, y) points of the first trace with at least min_boundary_points points, else the longest trace.
        """
        car_x = int(self.car_position[0])
        candidates = np.flatnonzero(on_side & available)
        candidates = candidates[np.lexsort((np.abs(maxima[0, candidates] - car_x), maxima[1, candidates]))]

        best_points: npt.NDArray[np.int64] = first_start
        best_available: npt.NDArray[np.bool_] = available.copy()
        tried: npt.NDArray[np.bool_] = np.zeros_like(available)
        start = first_start
        for _ in range(self.max_start_attempts):
            trial_available = available.copy()
            points = self._trace_lane_boundary(start, maxima, trial_available, left_boundary)
            tried |= available & ~trial_available
            if points.shape[0] > best_points.shape[0]:
                best_points, best_available = points, trial_available
            if points.shape[0] >= self.min_boundary_points:
                break

            remaining = candidates[~tried[candidates]]
            if remaining.size == 0:
                break
            tried[remaining[0]] = True
            start = maxima[:, remaining[0]][None]

        available[:] = best_available
        return best_points


    def _fit_spline(self, points: npt.NDArray[np.int64]) -> Spline | None:
        """Fit a smoothing parametric cubic B-spline through the points of one lane boundary.

        splprep parametrises the curve by normalised cumulative chord length, so the spline parameter runs from
        0 at the point nearest the car to 1 at the farthest one, roughly proportional to the length travelled
        along the boundary. smoothing s bounds the sum of squared distances between the points and the curve.

        Parameters
        ----------
        points : (NumBoundaryPoints x 2) array
            (x, y) lane boundary points ordered away from the car.

        Returns
        -------
        Spline or None
            (knots, coefficients, degree) for scipy.interpolate.splev, or None if the points cannot support a
            cubic spline.
        """
        # Repeated consecutive points give a zero chord length, which splprep cannot parametrise.
        keep: npt.NDArray[np.bool_] = np.ones(points.shape[0], dtype=bool)
        keep[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
        unique_points = points[keep].astype(np.float64)

        # A cubic spline needs more points than its degree.
        if unique_points.shape[0] <= 3:
            return None
        try:
            tck, _ = splprep([unique_points[:, 0], unique_points[:, 1]], s=self.spline_smoothness, k=3)
        except ValueError:
            return None
        return tck


    def lane_detection(
        self,
        state_image_full: npt.NDArray[np.uint8],
        obstacle_mask: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[Spline | None, Spline | None]:
        """Detect the two lane boundaries of the ego lane.

        Each boundary whose points cannot be fitted on this frame falls back to its spline from the preceding
        frame. The intermediate results are stored in self.last_frame.

        Parameters
        ----------
        state_image_full : (ImageHeight x ImageWidth x 3) array
            RGB front camera image.
        obstacle_mask : (ImageHeight x ImageWidth) array or None
            Front-view obstacle mask, e.g. ObjectDetection.detect(...).obstacle_mask(). Edges on obstacles are
            ignored. None disables obstacle suppression.

        Returns
        -------
        lane_boundary1 : Spline or None
            Left lane boundary in cut_gray coordinates (x in pixels, y in rows ahead of the car). None until a
            boundary has been fitted once.
        lane_boundary2 : Spline or None
            Right lane boundary, same convention.
        """
        bev = self.front2bev(state_image_full)
        obstacle_mask_bev = self.obstacle_mask_to_bev(obstacle_mask) if obstacle_mask is not None else None

        # to gray
        gray_state = self.cut_gray(bev)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state, obstacle_mask_bev)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        lane_boundary1_points, lane_boundary2_points, lane_found = self.assign_lane_boundary_points(
            gradient_sum, maxima)

        # The template's "more points than spline parameters" check is part of _fit_spline.
        lane_boundary1 = self._fit_spline(lane_boundary1_points) if lane_found else None
        lane_boundary2 = self._fit_spline(lane_boundary2_points) if lane_found else None

        self.last_frame = LaneDetectionFrame(
            bev=bev,
            obstacle_mask_bev=obstacle_mask_bev,
            gradient_sum=gradient_sum,
            maxima=maxima,
            lane_boundary1_points=lane_boundary1_points,
            lane_boundary2_points=lane_boundary2_points,
            lane_boundary1_fitted=lane_boundary1 is not None,
            lane_boundary2_fitted=lane_boundary2 is not None,
        )

        if lane_boundary1 is not None:
            self.lane_boundary1_old = lane_boundary1
        if lane_boundary2 is not None:
            self.lane_boundary2_old = lane_boundary2

        return self.lane_boundary1_old, self.lane_boundary2_old


    def cut_to_bev_pixels(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert points from cut_gray coordinates to BEV image pixel coordinates.

        Parameters
        ----------
        points : (2 x NumPoints) array
            (x, y) points with y counted in rows ahead of the car, e.g. a sampled spline or waypoints.

        Returns
        -------
        (2 x NumPoints) array
            (u, v) pixel coordinates in the BEV image, v counted down from the top row.
        """
        height = self.image_size[0]
        return np.stack([points[0], height - 1 - points[1]])


    def cut_to_front_view(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project points from cut_gray coordinates into the front camera image.

        Parameters
        ----------
        points : (2 x NumPoints) array
            (x, y) points with y counted in rows ahead of the car.

        Returns
        -------
        (2 x NumPoints) array
            (u, v) pixel coordinates in the front camera image.
        """
        bev_pixels = self.cut_to_bev_pixels(points)
        homogeneous = self.H_bev2im @ np.vstack([bev_pixels, np.ones(bev_pixels.shape[1])])
        return homogeneous[:2] / homogeneous[2]


    def plot_state_lane(
        self,
        state_image_full: npt.NDArray[np.uint8],
        steps: int,
        fig: matplotlib.figure.Figure,
        waypoints: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Plot the BEV of the current frame with the latest lane boundary splines and optional waypoints.

        Parameters
        ----------
        state_image_full : (ImageHeight x ImageWidth x 3) array
            RGB front camera image.
        steps : int
            Simulation step, shown in the title.
        fig : matplotlib.figure.Figure
            Figure to draw into; it is cleared first.
        waypoints : (2 x NumWaypoints) array or None
            (x, y) waypoints in cut_gray coordinates.
        """
        plt.figure(fig.number)
        plt.gcf().clear()
        plt.imshow(self.front2bev(state_image_full))

        t = np.linspace(0, 1, 20)
        for spline in (self.lane_boundary1_old, self.lane_boundary2_old):
            if spline is not None:
                u, v = self.cut_to_bev_pixels(np.array(splev(t, spline)))
                plt.plot(u, v, linewidth=3, color='orange')
        if waypoints is not None and len(waypoints):
            u, v = self.cut_to_bev_pixels(np.asarray(waypoints, dtype=np.float64))
            plt.scatter(u, v, color='white')

        plt.title(f'step {steps}')
        plt.axis('off')
        fig.canvas.flush_events()
