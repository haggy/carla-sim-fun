"""
Dataset of the episodes written by collect.py: one sample per saved camera frame, with the expert's control and the
future waypoints as labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from common import NUM_WAYPOINTS, WAYPOINT_STEP_INTERVAL, LaneDetection, future_waypoints, network_input


class DrivingDataset(Dataset):
    """Frames of several episodes, as RGB or BEV network inputs.

    A frame is used only if its whole label window is free of perturbations: the expert's next control and every
    pose up to the last future waypoint must come from the autopilot, not from a forced steering offset. Frames shortly
    after a perturbation, where the autopilot steers back onto its line, are kept; they teach recovery.
    """

    def __init__(self, episode_files: list[Path], input_type: str) -> None:
        """Index the usable frames of the given episodes.

        Parameters
        ----------
        episode_files : list[Path]
            The episode_<seed>_rgb.npy files to use; the matching _meta.npz must sit next to each.
        input_type : str
            "rgb" or "bev", see common.network_input.
        """
        self.input_type = input_type
        # Built lazily in each DataLoader worker, since the OpenCV homography state is not shared across processes.
        self._lane_detector: LaneDetection | None = None
        self._frames: list[npt.NDArray[np.uint8]] = []
        self._samples: list[tuple[int, int, float, npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []

        horizon = NUM_WAYPOINTS * WAYPOINT_STEP_INTERVAL
        for episode_idx, rgb_path in enumerate(episode_files):
            meta = np.load(str(rgb_path).replace('_rgb.npy', '_meta.npz'))
            self._frames.append(np.load(rgb_path, mmap_mode='r'))
            poses, perturbed = meta['poses'], meta['perturbed']
            for frame_idx, step in enumerate(meta['frame_steps']):
                if step + horizon >= len(poses) or perturbed[step:step + horizon + 1].any():
                    continue
                # The control the expert applied in response to this frame is the one of the next tick.
                control = meta['controls'][step + 1].astype(np.float32)
                waypoints = future_waypoints(poses, step).astype(np.float32)
                self._samples.append((episode_idx, frame_idx, float(meta['speeds'][step]), control, waypoints))

    def __len__(self) -> int:
        """Number of usable frames.

        Returns
        -------
        int
            Dataset size.
        """
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One training sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of 4 tensors
            image (3 x ImageHeight x ImageWidth) uint8, speed () float32 in m/s, control (3,) float32 as
            (steer, throttle, brake), waypoints (NumWaypoints x 2) float32 as (forward, right) metres.
        """
        if self._lane_detector is None:
            self._lane_detector = LaneDetection()
        episode_idx, frame_idx, speed, control, waypoints = self._samples[idx]
        image = network_input(np.asarray(self._frames[episode_idx][frame_idx]), self.input_type, self._lane_detector)
        return (torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1), torch.tensor(speed, dtype=torch.float32),
                torch.from_numpy(control), torch.from_numpy(waypoints))

    def label_statistics(self) -> dict[str, float]:
        """Summary of the labels, for the notes and sanity checks.

        Returns
        -------
        dict[str, float]
            samples, mean_speed_mps, turning_fraction (|steer| > 0.05), braking_fraction (brake > 0.1),
            mean_final_waypoint_forward_m.
        """
        controls = np.stack([s[3] for s in self._samples])
        waypoints = np.stack([s[4] for s in self._samples])
        return {
            'samples': len(self._samples),
            'mean_speed_mps': round(float(np.mean([s[2] for s in self._samples])), 2),
            'turning_fraction': round(float(np.mean(np.abs(controls[:, 0]) > 0.05)), 3),
            'braking_fraction': round(float(np.mean(controls[:, 2] > 0.1)), 3),
            'mean_final_waypoint_forward_m': round(float(waypoints[:, -1, 0].mean()), 2),
        }
