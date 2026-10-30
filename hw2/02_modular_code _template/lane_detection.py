import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(
        self,
        cut_size: int = 120,
        spline_smoothness: float = 10,
        gradient_threshold: float = 14,
        distance_maxima_gradient: int = 3,
        image_size: tuple[int, int] = (240, 320),
        camera_fov_deg: float = 90.0,
        camera_xyz: tuple[float, float, float] = (0.0, 0.0, 2.0),
        bev_forward_range_m: tuple[float, float] = (3.0, 21.0),
        bev_lateral_range_m: tuple[float, float] = (-12.0, 12.0),
        obstacle_margin_px: int = 3,
    ) -> None:
        """Configure the lane detector and precompute the front-view to BEV homography.

        Camera defaults mirror the 'rgb_front' sensor in performance_benchmark/leaderboard/team_code/test_agent.py.
        The default BEV ranges give square 7.5 cm pixels.

        Parameters
        ----------
        cut_size : int
            Number of BEV rows, counted from the car, that are kept for lane detection.
        spline_smoothness : float
            Smoothing factor `s` passed to scipy.interpolate.splprep.
        gradient_threshold : float
            Absolute gradients below this value are zeroed during edge detection.
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
            (near, far) distance ahead of the vehicle covered by the BEV image, in metres. A level camera at
            2 m with a 90 deg FOV sees no ground closer than ~2.7 m, so a smaller near limit leaves empty rows.
        bev_lateral_range_m : tuple[float, float]
            (left, right) lateral extent covered by the BEV image, in metres. Left is negative.
        obstacle_margin_px : int
            Front-view pixels the obstacle masks are grown by before they are warped to the BEV.
        """
        self.car_position = np.array([160,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
        self.image_size = image_size
        self.obstacle_margin_px = obstacle_margin_px
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


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 160:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[320,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(
        self,
        state_image_full: npt.NDArray[np.uint8],
        obstacle_mask: npt.NDArray[np.bool_] | None = None,
    ):
        """Detect the two lane boundaries of the ego lane.

        ##### TODO ##### (units 1D/1E: edge assignment and spline fitting)

        Parameters
        ----------
        state_image_full : (ImageHeight x ImageWidth x 3) array
            RGB front camera image.
        obstacle_mask : (ImageHeight x ImageWidth) array or None
            Front-view obstacle mask, e.g. ObjectDetection.detect(...).obstacle_mask(). Edges on obstacles are
            ignored. None disables obstacle suppression.

        Returns
        -------
        lane_boundary1, lane_boundary2
            Splines of the two lane boundaries.
        """
        bev = self.front2bev(state_image_full)
        obstacle_mask_bev = self.obstacle_mask_to_bev(obstacle_mask) if obstacle_mask is not None else None

        # to gray
        gray_state = self.cut_gray(bev)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state, obstacle_mask_bev)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)

            # lane_boundary 1

            # lane_boundary 2

            ################
            

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                pass
                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1

                # lane_boundary 2
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+320-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
