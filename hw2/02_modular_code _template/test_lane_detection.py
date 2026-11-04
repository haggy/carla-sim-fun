import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lane_detection import LaneDetection, Spline
from object_detection import DetectionResult, ObjectDetection
from bev import SceneBEV
from sim_session import CarlaSession, build_arg_parser, sample_spline

from PIL import Image


def plot_step(
    axes: npt.NDArray[np.object_],
    LD_module: LaneDetection,
    annotated_front: npt.NDArray[np.uint8],
    num_detections: int,
    gradient_unmasked: npt.NDArray[np.float64],
    scene_bev: npt.NDArray[np.uint8],
    num_placed: int,
    splines: tuple[Spline | None, Spline | None],
    step: int,
) -> None:
    """Draw object detection and lane boundaries in the front view, the BEV, and the edges before and after
    obstacle suppression with their assigned maxima.

    Parameters
    ----------
    axes : (2 x 2) array of matplotlib Axes
        Grid of axes the four panels are drawn into.
    LD_module : LaneDetection
        Lane detector; its last_frame holds the intermediate results of the frame being drawn.
    annotated_front : (ImageHeight x ImageWidth x 3) array
        RGB front view with detections drawn by ObjectDetection.draw.
    num_detections : int
        Number of obstacles detected, shown in the title.
    gradient_unmasked : (CutSize x ImageWidth) array
        Thresholded gradients without obstacle suppression, row 0 nearest the car.
    scene_bev : (ImageHeight x ImageWidth x 3) array
        Scene BEV from SceneBEV.render, without lane splines (they are drawn here so fallbacks show dashed).
    num_placed : int
        Number of obstacles SceneBEV placed on the ground, shown in the title.
    splines : tuple[Spline or None, Spline or None]
        Left and right lane boundaries returned by lane_detection.
    step : int
        Simulation step, shown in the title.
    """
    frame = LD_module.last_frame
    assert frame is not None
    for ax in axes.flat:
        ax.clear()
        ax.axis('off')

    # A boundary that could not be fitted this frame shows the preceding frame's spline, dashed.
    fitted = (frame.lane_boundary1_fitted, frame.lane_boundary2_fitted)
    samples = [sample_spline(spline) for spline in splines]

    height, width = annotated_front.shape[:2]
    axes[0, 0].imshow(annotated_front)
    for points, is_fitted in zip(samples, fitted):
        u, v = LD_module.cut_to_front_view(points)
        axes[0, 0].plot(u, v, color='orange', linewidth=2, linestyle='-' if is_fitted else '--')
    axes[0, 0].set_xlim(0, width - 1)
    axes[0, 0].set_ylim(height - 1, 0)
    axes[0, 0].set_title(f'object detection: {num_detections} obstacles (step {step})')

    axes[0, 1].imshow(scene_bev)
    for points, is_fitted in zip(samples, fitted):
        u, v = LD_module.cut_to_bev_pixels(points)
        axes[0, 1].plot(u, v, color='orange', linewidth=2, linestyle='-' if is_fitted else '--')
    axes[0, 1].set_title(f'scene BEV: {num_placed} obstacles placed, grey = occluded')

    # origin='lower' puts row 0 (nearest the car) at the bottom, matching the BEV panel.
    axes[1, 0].imshow(gradient_unmasked, cmap='gray', origin='lower')
    axes[1, 0].set_title('edges, no obstacle suppression')

    p1, p2 = frame.lane_boundary1_points, frame.lane_boundary2_points
    axes[1, 1].imshow(frame.gradient_sum, cmap='gray', origin='lower')
    axes[1, 1].scatter(frame.maxima[0], frame.maxima[1], s=1, c='red')
    axes[1, 1].scatter(p1[:, 0], p1[:, 1], s=2, c='cyan')
    axes[1, 1].scatter(p2[:, 0], p2[:, 1], s=2, c='yellow')
    for points in samples:
        axes[1, 1].plot(points[0], points[1], color='orange', linewidth=1)
    axes[1, 1].set_xlim(0, frame.gradient_sum.shape[1] - 1)
    axes[1, 1].set_ylim(0, frame.gradient_sum.shape[0] - 1)
    axes[1, 1].set_title(f'{frame.maxima.shape[1]} maxima, assigned L {p1.shape[0]} / R {p2.shape[0]}')

    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def main() -> None:
    """Drive the ego vehicle (autopilot or arrow keys) in synchronous mode and run obstacle and lane detection
    on every frame."""
    args = build_arg_parser('Live test of LaneDetection BEV, edge detection and obstacle detection in CARLA.',
                            'lane_detection_snapshots').parse_args()

    with CarlaSession(args) as session:
        LD_module = LaneDetection()
        obj_detection = ObjectDetection()
        scene_bev = SceneBEV(LD_module)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        session.attach_figure(fig)
        plt.ion()
        plt.show()

        step = 0
        while session.running(step):
            front = session.tick()
            detections: DetectionResult = obj_detection.detect(front)

            splines = LD_module.lane_detection(front, detections.obstacle_mask())
            obstacles = scene_bev.localize(detections)

            if session.should_draw(step):
                plot_step(
                    axes,
                    LD_module,
                    obj_detection.draw(front, detections),
                    len(detections),
                    LD_module.edge_detection(LD_module.cut_gray(LD_module.last_frame.bev)),
                    scene_bev.render(front, detections, obstacles),
                    len(obstacles),
                    splines,
                    step,
                )
                session.after_draw(step)
            step += 1

        session.save_final()


if __name__ == '__main__':
    main()

    # od = ObjectDetection()
    # img = np.asarray(Image.open("/home/dave/Downloads/bus.jpg").convert("RGB"))
    # boxed = od.draw(img, od.detect(img))

    # fig, axes = plt.subplots(1, 1, figsize=(15, 4.5))

    # axes.imshow(boxed)
    # plt.show()
    
