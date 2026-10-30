"""
Shared CARLA harness for the live pipeline tests: Town01 in synchronous mode, an ego vehicle with the test agent's
front camera, NPC traffic, arrow-key driving and figure snapshots.
"""
import argparse
import queue
import random
import sys
from pathlib import Path
from types import TracebackType

import carla
import matplotlib.backend_bases
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev

from lane_detection import Spline

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


def build_arg_parser(description: str, snapshot_dir: str) -> argparse.ArgumentParser:
    """Command line options shared by the live tests.

    Parameters
    ----------
    description : str
        Description shown by --help.
    snapshot_dir : str
        Default folder for snapshots.

    Returns
    -------
    argparse.ArgumentParser
        Parser with host, port, town, steps (0 runs until the window is closed), seed, npcs, manual,
        snapshot_dir, snapshot_every and save; callers may add their own options before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', default='Town01',
                        help='Map to load if the server is on another one; the assignment allows only Town01.')
    parser.add_argument('--steps', type=int, default=0, help='Number of simulation steps; 0 runs until closed.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the spawn point choice.')
    parser.add_argument('--npcs', type=int, default=30, help='Autopilot traffic vehicles to spawn as obstacles.')
    parser.add_argument('--manual', action='store_true',
                        help='Drive with the arrow keys (focus the plot window) instead of the autopilot.')
    parser.add_argument('--snapshot-dir', type=Path, default=Path(snapshot_dir),
                        help="Folder for snapshots taken with the 's' key or --snapshot-every.")
    parser.add_argument('--snapshot-every', type=int, default=0,
                        help='Also save a snapshot every N steps; 0 disables periodic snapshots.')
    parser.add_argument('--save', type=Path, default=None, help='Save the last plotted figure to this path.')
    return parser


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


class CarlaSession:
    """Context manager owning the simulation for one live test run.

    Entering connects, loads the town, switches to synchronous mode and spawns the ego vehicle (autopilot unless
    --manual), NPC traffic and the front camera. Leaving destroys everything and restores the world settings.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Store the options from build_arg_parser; nothing is spawned until the session is entered.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed options from build_arg_parser.
        """
        self.args = args
        self.driver = KeyboardDriver()
        self.vehicle: carla.Actor | None = None
        self._npcs: list[carla.Actor] = []
        self._frames: queue.Queue[carla.Image] = queue.Queue()

    def __enter__(self) -> 'CarlaSession':
        """Set up the world and actors.

        Returns
        -------
        CarlaSession
            This session, ready for tick().
        """
        args = self.args
        random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        self.world = client.get_world()
        if args.town not in self.world.get_map().name:
            self.world = client.load_world(args.town)
        self.traffic_manager = client.get_trafficmanager()

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)
        self.sensors = SensorManager(self.world)

        try:
            blueprints = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            for spawn_point in spawn_points:
                self.vehicle = self.world.try_spawn_actor(blueprints.filter('vehicle.lincoln.mkz*')[0], spawn_point)
                if self.vehicle is not None:
                    break
            if self.vehicle is None:
                raise RuntimeError('No free spawn point for the ego vehicle.')
            self.vehicle.set_autopilot(not args.manual, self.traffic_manager.get_port())

            npc_blueprints = [bp for bp in blueprints.filter('vehicle.*')
                              if int(bp.get_attribute('number_of_wheels')) == 4]
            for spawn_point in spawn_points[:args.npcs + 1]:
                npc = self.world.try_spawn_actor(random.choice(npc_blueprints), spawn_point)
                if npc is not None:
                    npc.set_autopilot(True, self.traffic_manager.get_port())
                    self._npcs.append(npc)

            _, camera_idx = self.sensors.add_rgb_camera(self.vehicle, CAMERA_TRANSFORM, CAMERA_ATTRIBUTES)
            self.sensors.add_listener(camera_idx, self._frames.put)
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Destroy the actors and restore the world settings, also when the test failed.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            Exception type raised inside the with block, if any.
        exc : BaseException or None
            Exception raised inside the with block, if any.
        traceback : TracebackType or None
            Its traceback.
        """
        self._cleanup()

    def _cleanup(self) -> None:
        """Destroy sensors and vehicles, then hand the world back in its original mode."""
        print('Cleaning up...')
        self.sensors.shutdown()
        # The traffic manager crashes the client if it is still driving a vehicle when it is destroyed.
        for actor in ([self.vehicle] if self.vehicle is not None else []) + self._npcs:
            actor.set_autopilot(False, self.traffic_manager.get_port())
            actor.destroy()
        self.vehicle, self._npcs = None, []
        self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(self._original_settings)

    @property
    def speed_mps(self) -> float:
        """Current ego speed.

        Returns
        -------
        float
            Speed in m/s.
        """
        return self.vehicle.get_velocity().length()

    def tick(self) -> npt.NDArray[np.uint8]:
        """Advance the simulation one step and return the matching front camera frame.

        In --manual mode the arrow-key command is applied before the tick.

        Returns
        -------
        (ImageHeight x ImageWidth x 3) array
            RGB front camera image of this tick.
        """
        if self.args.manual:
            self.vehicle.apply_control(self.driver.control(self.speed_mps))
        frame_id = self.world.tick()
        image = self._frames.get(timeout=FRAME_TIMEOUT_SECONDS)
        # Drop stale frames so the image always belongs to this tick.
        while image.frame < frame_id:
            image = self._frames.get(timeout=FRAME_TIMEOUT_SECONDS)
        move_spectator_to_vehicle(self.world, self.vehicle)
        return carla_image_to_rgb(image)

    def attach_figure(self, fig: matplotlib.figure.Figure) -> None:
        """Route the figure's key events to the arrow-key driver and the snapshot key.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The test's live window.
        """
        # Matplotlib binds the arrow keys to toolbar navigation and "s" to its save dialog by default, which would
        # fight the driving and snapshot keys.
        for keymap in ('keymap.back', 'keymap.forward', 'keymap.save'):
            plt.rcParams[keymap] = []
        fig.canvas.mpl_connect('key_press_event', self.driver.on_press)
        fig.canvas.mpl_connect('key_release_event', self.driver.on_release)
        self._fig = fig

    def running(self, step: int) -> bool:
        """Whether the test loop should continue.

        Parameters
        ----------
        step : int
            Number of steps completed so far.

        Returns
        -------
        bool
            False once --steps is reached or the window has been closed.
        """
        return (self.args.steps == 0 or step < self.args.steps) and plt.fignum_exists(self._fig.number)

    def should_draw(self, step: int) -> bool:
        """Whether this step's figure has to be redrawn: every other step, or when a snapshot is due.

        Parameters
        ----------
        step : int
            Current step.

        Returns
        -------
        bool
            True to redraw the figure this step.
        """
        return step % 2 == 0 or self._snapshot_due(step)

    def _snapshot_due(self, step: int) -> bool:
        """Whether a periodic or keyboard snapshot is pending for this step.

        Parameters
        ----------
        step : int
            Current step.

        Returns
        -------
        bool
            True if the figure should be saved after drawing.
        """
        periodic = self.args.snapshot_every > 0 and step % self.args.snapshot_every == 0
        return periodic or self.driver.snapshot_requested

    def after_draw(self, step: int) -> None:
        """Save a snapshot of the figure if one is due.

        Parameters
        ----------
        step : int
            Current step, used in the file name.
        """
        if self._snapshot_due(step):
            self.args.snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(self.args.snapshot_dir / f'step_{step:05d}.png', dpi=100)
            self.driver.snapshot_requested = False

    def save_final(self) -> None:
        """Write the last figure to --save, if given."""
        if self.args.save is not None:
            self._fig.savefig(self.args.save, dpi=100)
            print(f'saved {self.args.save}')
