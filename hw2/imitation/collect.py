"""
Collect expert driving data on Town01 for the imitation learning experiments (unit 4C).

The traffic manager's autopilot is the expert. Every episode starts at a different spawn point and records the front
camera frame, the expert's control and the ego pose; future waypoints are derived from the poses at training time.
To show the networks how to recover from mistakes, the autopilot is periodically switched off and the car is pushed
off its line with a random steering offset; those steps are flagged so they are never used as labels.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import carla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from common import DATA_DIR  # noqa: F401  (also puts the template folder on the path)
from sim_session import CarlaSession, build_arg_parser


def collect_episode(args: argparse.Namespace, seed: int, out_dir: Path) -> int:
    """Drive one autopilot episode and write it to <out_dir>/episode_<seed>_{rgb.npy,meta.npz}.

    Parameters
    ----------
    args : argparse.Namespace
        Options from build_arg_parser plus episode_steps, save_every, perturb_every, perturb_steps and
        perturb_steer.
    seed : int
        Seed for the spawn point and the perturbations; also names the episode files.
    out_dir : Path
        Folder for the episode files.

    Returns
    -------
    int
        Number of camera frames saved.
    """
    args.seed = seed
    rng = random.Random(seed)
    frames: list[npt.NDArray[np.uint8]] = []
    frame_steps: list[int] = []
    poses: list[tuple[float, float, float]] = []
    speeds: list[float] = []
    controls: list[tuple[float, float, float]] = []
    perturbed: list[bool] = []

    with CarlaSession(args) as session:
        session.attach_figure(plt.figure())
        tm = session.traffic_manager
        vehicle = session.vehicle
        # Lights and other traffic are out of scope for these policies; the expert only follows the road.
        tm.ignore_lights_percentage(vehicle, 100)
        tm.ignore_signs_percentage(vehicle, 100)
        tm.vehicle_percentage_speed_difference(vehicle, 0)
        tm.random_left_lanechange_percentage(vehicle, 0)
        tm.random_right_lanechange_percentage(vehicle, 0)

        perturb_left = 0
        steer_offset = 0.0
        autopilot_resumed = False
        for step in range(args.episode_steps):
            start_perturbation = (step > 0 and perturb_left == 0 and args.perturb_every > 0
                                  and step % args.perturb_every == 0 and session.speed_mps > 2.0)
            if start_perturbation:
                vehicle.set_autopilot(False, tm.get_port())
                perturb_left = args.perturb_steps
                steer_offset = rng.choice((-1, 1)) * rng.uniform(0.5, 1.0) * args.perturb_steer

            control = None
            if perturb_left > 0:
                control = carla.VehicleControl(throttle=0.4, steer=steer_offset)
            front = session.tick(control)
            # On the first tick after the autopilot is re-enabled the car still reports the forced control.
            stale_control = autopilot_resumed
            autopilot_resumed = False
            if perturb_left > 0:
                perturb_left -= 1
                if perturb_left == 0:
                    vehicle.set_autopilot(True, tm.get_port())
                    autopilot_resumed = True

            transform = vehicle.get_transform()
            applied = vehicle.get_control()
            poses.append((transform.location.x, transform.location.y, transform.rotation.yaw))
            speeds.append(session.speed_mps)
            controls.append((applied.steer, applied.throttle, applied.brake))
            perturbed.append(control is not None or stale_control)
            if step % args.save_every == 0:
                frames.append(front)
                frame_steps.append(step)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'episode_{seed:04d}_rgb.npy', np.stack(frames))
    np.savez(out_dir / f'episode_{seed:04d}_meta.npz', frame_steps=np.array(frame_steps), poses=np.array(poses),
             speeds=np.array(speeds), controls=np.array(controls, dtype=np.float32),
             perturbed=np.array(perturbed))
    return len(frames)


def main() -> None:
    """Collect --episodes episodes with consecutive seeds."""
    parser = build_arg_parser('Collect Town01 autopilot data for imitation learning.', 'unused')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--first-seed', type=int, default=100,
                        help='Seed of the first episode; evaluation uses seeds below 100.')
    parser.add_argument('--episode-steps', type=int, default=1200, help='Simulation steps per episode (20 Hz).')
    parser.add_argument('--save-every', type=int, default=2, help='Save every N-th camera frame.')
    parser.add_argument('--perturb-every', type=int, default=100,
                        help='Steps between steering perturbations; 0 disables them.')
    parser.add_argument('--perturb-steps', type=int, default=8, help='Length of a perturbation in steps.')
    parser.add_argument('--perturb-steer', type=float, default=0.3, help='Largest steering offset of a perturbation.')
    parser.add_argument('--out', type=Path, default=DATA_DIR / 'episodes')
    parser.set_defaults(npcs=0)
    args = parser.parse_args()

    for seed in range(args.first_seed, args.first_seed + args.episodes):
        start = time.time()
        num_frames = collect_episode(args, seed, args.out)
        print(f'episode {seed}: {num_frames} frames in {time.time() - start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
