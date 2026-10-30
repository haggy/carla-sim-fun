import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from bev import SceneBEV
from lane_detection import LaneDetection, Spline
from object_detection import ObjectDetection
from sim_session import CarlaSession, build_arg_parser, sample_spline
from waypoint_prediction import waypoint_prediction

WAYPOINT_COLORS: dict[str, str] = {'center': 'white', 'smooth': 'lime'}


def plot_step(
    axes: npt.NDArray[np.object_],
    LD_module: LaneDetection,
    front: npt.NDArray[np.uint8],
    scene: npt.NDArray[np.uint8],
    splines: tuple[Spline | None, Spline | None],
    waypoints: dict[str, npt.NDArray[np.float64]],
    step: int,
) -> None:
    """Draw lane boundaries and waypoints in the front view and in the scene BEV.

    Parameters
    ----------
    axes : (2,) array of matplotlib Axes
        Axes for the front view and the BEV.
    LD_module : LaneDetection
        Lane detector, used for the coordinate conversions.
    front : (ImageHeight x ImageWidth x 3) array
        RGB front camera image.
    scene : (ImageHeight x ImageWidth x 3) array
        Scene BEV from SceneBEV.render.
    splines : tuple[Spline or None, Spline or None]
        Left and right lane boundaries.
    waypoints : dict[str, (2 x NumWaypoints) array]
        Waypoints per way_type, in cut_gray coordinates; empty when the boundaries are not known yet.
    step : int
        Simulation step, shown in the title.
    """
    for ax in axes:
        ax.clear()
        ax.axis('off')
    height, width = front.shape[:2]

    axes[0].imshow(front)
    axes[1].imshow(scene)
    for to_pixels, ax in ((LD_module.cut_to_front_view, axes[0]), (LD_module.cut_to_bev_pixels, axes[1])):
        for spline in splines:
            u, v = to_pixels(sample_spline(spline))
            ax.plot(u, v, color='orange', linewidth=2)
        for way_type, points in waypoints.items():
            u, v = to_pixels(points)
            ax.plot(u, v, color=WAYPOINT_COLORS[way_type], linewidth=1, marker='o', markersize=4,
                    label=f'{way_type} waypoints')
        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)

    axes[0].set_title(f'front view (step {step})')
    axes[1].set_title('scene BEV: lane splines + waypoints')
    if waypoints:
        axes[1].legend(loc='upper right', fontsize=7)

    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def main() -> None:
    """Run lane detection and waypoint prediction live and draw the predicted path."""
    parser = build_arg_parser('Live test of waypoint prediction in CARLA.', 'waypoint_prediction_snapshots')
    parser.add_argument('--way-types', nargs='+', default=['center'], choices=sorted(WAYPOINT_COLORS),
                        help='Waypoint types to predict and draw.')
    args = parser.parse_args()

    with CarlaSession(args) as session:
        LD_module = LaneDetection()
        obj_detection = ObjectDetection()
        scene_bev = SceneBEV(LD_module)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        session.attach_figure(fig)
        plt.ion()
        plt.show()

        step = 0
        while session.running(step):
            front = session.tick()
            detections = obj_detection.detect(front)

            # lane detection
            lane1, lane2 = LD_module.lane_detection(front, detections.obstacle_mask())

            # waypoint prediction; the target speed follows in unit 2C
            waypoints: dict[str, npt.NDArray[np.float64]] = {}
            if lane1 is not None and lane2 is not None:
                waypoints = {way_type: waypoint_prediction(lane1, lane2, way_type=way_type)
                             for way_type in args.way_types}

            if session.should_draw(step):
                scene = scene_bev.render(front, detections, scene_bev.localize(detections))
                plot_step(axes, LD_module, front, scene, (lane1, lane2), waypoints, step)
                session.after_draw(step)
            step += 1

        session.save_final()


if __name__ == '__main__':
    main()
