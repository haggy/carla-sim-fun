import argparse
import queue
import random
import sys
from pathlib import Path

import carla
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scipy.interpolate import splev

from lane_detection import LaneDetection, Spline
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
        host, port, steps (0 runs until the window is closed), seed, npcs (traffic vehicle count), manual
        (arrow-key driving instead of autopilot), snapshot_dir, snapshot_every and save (path or None).
    """
    parser = argparse.ArgumentParser(
        description='Live test of LaneDetection BEV, edge detection and obstacle detection in CARLA.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=0, help='Number of simulation steps; 0 runs until closed.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the spawn point choice.')
    parser.add_argument('--npcs', type=int, default=30, help='Autopilot traffic vehicles to spawn as obstacles.')
    parser.add_argument('--manual', action='store_true',
                        help='Drive with the arrow keys (focus the plot window) instead of the autopilot.')
    parser.add_argument('--snapshot-dir', type=Path, default=Path('lane_detection_snapshots'),
                        help="Folder for snapshots taken with the 's' key or --snapshot-every.")
    parser.add_argument('--snapshot-every', type=int, default=0,
                        help='Also save a snapshot every N steps; 0 disables periodic snapshots.')
    parser.add_argument('--save', type=Path, default=None, help='Save the last plotted figure to this path.')
    return parser.parse_args()


class KeyboardDriver:
    """Arrow-key vehicle control from matplotlib key events.

    Up accelerates, down brakes (and reverses once stopped), left/right steer. Steering ramps towards full lock
    while a key is held and recentres when released, so a tap gives a small correction instead of a jerk.
    """

    THROTTLE = 0.6
    STEER_LIMIT = 0.5
    STEER_RATE = 0.08
    REVERSE_BELOW_MPS = 0.5

    def __init__(self) -> None:
        """Start with no keys held and the wheels straight."""
        self.held: set[str] = set()
        self.steer = 0.0
        self.snapshot_requested = False

    def on_press(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """Record a pressed key.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            Key press from the figure canvas.
        """
        if event.key == 's':
            self.snapshot_requested = True
        elif event.key is not None:
            self.held.add(event.key)

    def on_release(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """Record a released key.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            Key release from the figure canvas.
        """
        self.held.discard(event.key)

    def control(self, speed_mps: float) -> carla.VehicleControl:
        """Vehicle command for the keys currently held.

        Parameters
        ----------
        speed_mps : float
            Current vehicle speed, used to switch the down key from braking to reversing.

        Returns
        -------
        carla.VehicleControl
            Throttle, brake, steer and reverse for this tick.
        """
        target = self.STEER_LIMIT * (('right' in self.held) - ('left' in self.held))
        self.steer += float(np.clip(target - self.steer, -self.STEER_RATE, self.STEER_RATE))

        control = carla.VehicleControl(steer=self.steer)
        if 'up' in self.held:
            control.throttle = self.THROTTLE
        elif 'down' in self.held and speed_mps < self.REVERSE_BELOW_MPS:
            control.throttle, control.reverse = self.THROTTLE, True
        elif 'down' in self.held:
            control.brake = 1.0
        return control


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


def sample_spline(spline: Spline | None, num_samples: int = 40) -> npt.NDArray[np.float64]:
    """Evaluate a lane boundary spline along its full length.

    Parameters
    ----------
    spline : Spline or None
        Lane boundary from LaneDetection.lane_detection.
    num_samples : int
        Number of evenly spaced spline parameters in [0, 1].

    Returns
    -------
    (2 x NumSamples) array
        (x, y) points in cut_gray coordinates; empty (2 x 0) if spline is None.
    """
    if spline is None:
        return np.zeros((2, 0))
    return np.array(splev(np.linspace(0.0, 1.0, num_samples), spline))


def plot_step(
    axes: npt.NDArray[np.object_],
    LD_module: LaneDetection,
    annotated_front: npt.NDArray[np.uint8],
    num_detections: int,
    gradient_unmasked: npt.NDArray[np.float64],
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

    bev_overlay = frame.bev.copy()
    if frame.obstacle_mask_bev is not None:
        covered = frame.obstacle_mask_bev
        bev_overlay[covered] = (0.5 * bev_overlay[covered] + (127, 0, 127)).astype(np.uint8)
    axes[0, 1].imshow(bev_overlay)
    for points, is_fitted in zip(samples, fitted):
        u, v = LD_module.cut_to_bev_pixels(points)
        axes[0, 1].plot(u, v, color='orange', linewidth=2, linestyle='-' if is_fitted else '--')
    axes[0, 1].set_title('BEV + projected obstacles + lane splines')

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
        vehicle.set_autopilot(not args.manual, traffic_manager.get_port())

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

        # Matplotlib binds the arrow keys to toolbar navigation and "s" to its save dialog by default, which would
        # fight the driving and snapshot keys.
        for keymap in ('keymap.back', 'keymap.forward', 'keymap.save'):
            plt.rcParams[keymap] = []
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        driver = KeyboardDriver()
        fig.canvas.mpl_connect('key_press_event', driver.on_press)
        fig.canvas.mpl_connect('key_release_event', driver.on_release)
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
            if args.manual:
                vehicle.apply_control(driver.control(vehicle.get_velocity().length()))

            front = carla_image_to_rgb(image)
            detections: DetectionResult = obj_detection.detect(front)

            splines = LD_module.lane_detection(front, detections.obstacle_mask())

            snapshot_due = args.snapshot_every > 0 and step % args.snapshot_every == 0
            if step % 2 == 0 or snapshot_due or driver.snapshot_requested:
                plot_step(
                    axes,
                    LD_module,
                    obj_detection.draw(front, detections),
                    len(detections),
                    LD_module.edge_detection(LD_module.cut_gray(LD_module.last_frame.bev)),
                    splines,
                    step,
                )
            if snapshot_due or driver.snapshot_requested:
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.snapshot_dir / f'step_{step:05d}.png', dpi=100)
                driver.snapshot_requested = False
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
    
