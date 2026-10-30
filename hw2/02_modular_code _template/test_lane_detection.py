import argparse
import queue
import random
import sys
from pathlib import Path

import carla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lane_detection import LaneDetection
from object_detection import DetectionResult, ObjectDetection

from PIL import Image

# module/util.py lives three levels up; this folder is not a package, so it is put on the path directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from util import SensorManager, move_spectator_to_vehicle  # noqa: E402

# Mirrors the 'rgb_front' sensor in performance_benchmark/leaderboard/team_code/test_agent.py,
# which is also what LaneDetection's camera defaults assume.
CAMERA_TRANSFORM = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
CAMERA_ATTRIBUTES: dict[str, str] = {'image_size_x': '320', 'image_size_y': '240', 'fov': '90'}
FIXED_DELTA_SECONDS = 0.05
# The first rendered frame can take several seconds while the server warms up its renderer.
FRAME_TIMEOUT_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    """Parse the command line options.

    Returns
    -------
    argparse.Namespace
        host, port, steps (0 runs until the window is closed), seed, npcs (traffic vehicle count) and save
        (snapshot path or None).
    """
    parser = argparse.ArgumentParser(
        description='Live test of LaneDetection BEV, edge detection and obstacle detection in CARLA.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=0, help='Number of simulation steps; 0 runs until closed.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the spawn point choice.')
    parser.add_argument('--npcs', type=int, default=30, help='Autopilot traffic vehicles to spawn as obstacles.')
    parser.add_argument('--save', type=Path, default=None, help='Save the last plotted figure to this path.')
    return parser.parse_args()


def carla_image_to_rgb(image: carla.Image) -> npt.NDArray[np.uint8]:
    """Convert a CARLA camera frame to an RGB array.

    Parameters
    ----------
    image : carla.Image
        Raw frame from a sensor.camera.rgb sensor.

    Returns
    -------
    (ImageHeight x ImageWidth x 3) array
        RGB image.
    """
    # CARLA delivers BGRA bytes.
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, 2::-1].copy()


def plot_step(
    axes: npt.NDArray[np.object_],
    annotated_front: npt.NDArray[np.uint8],
    num_detections: int,
    bev: npt.NDArray[np.uint8],
    obstacle_mask_bev: npt.NDArray[np.bool_],
    gradient_unmasked: npt.NDArray[np.float64],
    gradient_sum: npt.NDArray[np.float64],
    maxima: npt.NDArray[np.int64],
    step: int,
) -> None:
    """Draw object detection, the BEV with obstacles, and the edges before and after obstacle suppression.

    Parameters
    ----------
    axes : (2 x 2) array of matplotlib Axes
        Grid of axes the four panels are drawn into.
    annotated_front : (ImageHeight x ImageWidth x 3) array
        RGB front view with detections drawn by ObjectDetection.draw.
    num_detections : int
        Number of obstacles detected, shown in the title.
    bev : (ImageHeight x ImageWidth x 3) array
        RGB BEV image from front2bev.
    obstacle_mask_bev : (ImageHeight x ImageWidth) array
        BEV obstacle mask from obstacle_mask_to_bev.
    gradient_unmasked : (CutSize x ImageWidth) array
        Thresholded gradients without obstacle suppression, row 0 nearest the car.
    gradient_sum : (CutSize x ImageWidth) array
        Thresholded gradients with obstacle suppression, row 0 nearest the car.
    maxima : (2 x NumMaxima) array
        (x, y) maxima of gradient_sum from find_maxima_gradient_rowwise.
    step : int
        Simulation step, shown in the title.
    """
    for ax in axes.flat:
        ax.clear()
        ax.axis('off')

    axes[0, 0].imshow(annotated_front)
    axes[0, 0].set_title(f'object detection: {num_detections} obstacles (step {step})')

    bev_overlay = bev.copy()
    bev_overlay[obstacle_mask_bev] = (0.5 * bev_overlay[obstacle_mask_bev] + (127, 0, 127)).astype(np.uint8)
    axes[0, 1].imshow(bev_overlay)
    axes[0, 1].set_title('BEV + projected obstacles')

    # origin='lower' puts row 0 (nearest the car) at the bottom, matching the BEV panel.
    axes[1, 0].imshow(gradient_unmasked, cmap='gray', origin='lower')
    axes[1, 0].set_title('edges, no obstacle suppression')
    axes[1, 1].imshow(gradient_sum, cmap='gray', origin='lower')
    axes[1, 1].scatter(maxima[0], maxima[1], s=1, c='red')
    axes[1, 1].set_title(f'edges w/o obstacles + {maxima.shape[1]} maxima')

    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()


def main() -> None:
    """Drive an autopilot vehicle in synchronous mode and run the lane detection stages on every frame."""
    args = parse_args()
    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

    sensors = SensorManager(world)
    vehicle: carla.Actor | None = None
    npcs: list[carla.Actor] = []
    try:
        blueprint = world.get_blueprint_library().filter('vehicle.lincoln.mkz*')[0]
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError('No free spawn point for the ego vehicle.')
        vehicle.set_autopilot(True, traffic_manager.get_port())

        npc_blueprints = [bp for bp in world.get_blueprint_library().filter('vehicle.*')
                          if int(bp.get_attribute('number_of_wheels')) == 4]
        for spawn_point in spawn_points[:args.npcs + 1]:
            npc = world.try_spawn_actor(random.choice(npc_blueprints), spawn_point)
            if npc is not None:
                npc.set_autopilot(True, traffic_manager.get_port())
                npcs.append(npc)

        camera, camera_idx = sensors.add_rgb_camera(vehicle, CAMERA_TRANSFORM, CAMERA_ATTRIBUTES)
        frames: queue.Queue[carla.Image] = queue.Queue()
        sensors.add_listener(camera_idx, frames.put)

        LD_module = LaneDetection()
        obj_detection = ObjectDetection()

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        plt.ion()
        plt.show()

        step = 0
        while args.steps == 0 or step < args.steps:
            frame_id = world.tick()
            image = frames.get(timeout=FRAME_TIMEOUT_SECONDS)
            # Drop stale frames so the image always belongs to this tick.
            while image.frame < frame_id:
                image = frames.get(timeout=FRAME_TIMEOUT_SECONDS)
            move_spectator_to_vehicle(world, vehicle)

            front = carla_image_to_rgb(image)
            detections: DetectionResult = obj_detection.detect(front)

            bev = LD_module.front2bev(front)
            obstacle_mask_bev = LD_module.obstacle_mask_to_bev(detections.obstacle_mask())
            gray = LD_module.cut_gray(bev)
            gradient_sum = LD_module.edge_detection(gray, obstacle_mask_bev)
            maxima = LD_module.find_maxima_gradient_rowwise(gradient_sum)

            if step % 2 == 0:
                plot_step(
                    axes,
                    obj_detection.draw(front, detections),
                    len(detections),
                    bev,
                    obstacle_mask_bev,
                    LD_module.edge_detection(gray),
                    gradient_sum,
                    maxima,
                    step,
                )
            step += 1

            if not plt.fignum_exists(fig.number):
                break

        if args.save is not None:
            fig.savefig(args.save, dpi=100)
            print(f'saved {args.save}')
    finally:
        print("Cleaning up...")
        sensors.shutdown()
        # The traffic manager crashes the client if it is still driving a vehicle when it is destroyed.
        for actor in ([vehicle] if vehicle is not None else []) + npcs:
            actor.set_autopilot(False, traffic_manager.get_port())
            actor.destroy()
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == '__main__':
    main()

    # od = ObjectDetection()
    # img = np.asarray(Image.open("/home/dave/Downloads/bus.jpg").convert("RGB"))
    # boxed = od.draw(img, od.detect(img))

    # fig, axes = plt.subplots(1, 1, figsize=(15, 4.5))

    # axes.imshow(boxed)
    # plt.show()
    
