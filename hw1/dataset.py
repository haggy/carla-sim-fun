import glob
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset

from module.hw1.network import IMG_HEIGHT, IMG_WIDTH, image_to_tensor

# Meta rows per sample: 4 movement dynamics + 3 controls (steer, throttle, brake)
NR_DYNAMICS_ROWS = 4
NR_ACTION_ROWS = 3
NR_META_ROWS = NR_DYNAMICS_ROWS + NR_ACTION_ROWS

IMG_ROWS = IMG_HEIGHT * IMG_WIDTH * 3

IMG_SUFFIX = "_img.npy"
META_SUFFIX = "_meta.npy"


class CarlaDataset(Dataset):
    """
    Reads capture pairs written by DataCollection._process_data_frames.

    Each capture is two files, because an ndarray is homogeneous and the pixels
    and the float observations need different dtypes:
        <base>_img.npy   uint8   (IMG_ROWS, frames)
        <base>_meta.npy  float32 (NR_META_ROWS, frames)

    Files are memory-mapped and indexed in place rather than concatenated, so
    startup is instant and memory stays low regardless of dataset size.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob.glob(os.path.join(data_dir, "*" + IMG_SUFFIX)))

        if not self.data_list:
            raise Exception(
                f"No numpy datasets were found at [{data_dir}] "
                f"(expected files ending in {IMG_SUFFIX})"
            )

        self._images: list[npt.NDArray[np.uint8]] = []
        self._meta: list[npt.NDArray[np.float32]] = []

        for img_path in self.data_list:
            meta_path = img_path[: -len(IMG_SUFFIX)] + META_SUFFIX
            images = self._load_npy(img_path)
            meta = self._load_npy(meta_path)

            if images.shape[0] != IMG_ROWS:
                raise Exception(
                    f"[{img_path}] has {images.shape[0]} image rows, expected {IMG_ROWS} "
                    f"({IMG_HEIGHT}x{IMG_WIDTH}x3) - wrong capture resolution?"
                )
            if meta.shape[0] != NR_META_ROWS:
                raise Exception(
                    f"[{meta_path}] has {meta.shape[0]} meta rows, expected {NR_META_ROWS}"
                )
            if images.shape[1] != meta.shape[1]:
                raise Exception(
                    f"[{img_path}] holds {images.shape[1]} frames but its meta file "
                    f"holds {meta.shape[1]} - the pair is out of sync"
                )

            self._images.append(images)
            self._meta.append(meta)

        # Prefix sums let __getitem__ map a global index onto (file, column)
        # without ever concatenating the files together.
        counts = [img.shape[1] for img in self._images]
        self._offsets = np.cumsum([0] + counts)

        print(
            f"Resolved {len(self.data_list)} capture pair(s), {int(self._offsets[-1])} samples, "
            f"memory-mapped ({sum(counts) * IMG_ROWS / 1e9:.2f} GB of images left on disk)"
        )

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
                  image  float32 (3, H, W) scaled to [0, 1]
                  action float32 (3,) - steer, throttle, brake
        """
        file_idx = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        col = int(idx - self._offsets[file_idx])

        # np.asarray materialises just this column out of the memory map.
        img = np.asarray(self._images[file_idx][:, col]).reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        observation = image_to_tensor(img)

        meta = np.asarray(self._meta[file_idx][:, col])
        action = torch.from_numpy(meta[-NR_ACTION_ROWS:].copy())

        return observation, action

    def _load_npy(self, path: str) -> npt.NDArray:
        p = Path(path)
        if not p.exists():
            raise Exception(f"Could not find dataset at [{path}]")

        return np.load(p, mmap_mode="r")


def get_dataloader(data_dir: str, batch_size: int = 64, num_workers: int = 4, shuffle: bool = True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
