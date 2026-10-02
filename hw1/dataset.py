import glob
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset

from module.hw1.network import IMG_HEIGHT, IMG_WIDTH, image_to_raw_tensor

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
        <base>_img.npy   uint8   (frames, IMG_ROWS)
        <base>_meta.npy  float32 (frames, NR_META_ROWS)

    Both are frame-major - one frame per ROW - so a sample is one contiguous
    read. The older column-major (rows, frames) layout spread each image's
    pixels `frames` bytes apart, which made loading ~300x slower; convert old
    captures with transpose_data.py.

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

            if images.shape[1] != IMG_ROWS:
                hint = (" - this looks like the old column-major layout, convert it "
                        "with transpose_data.py" if images.shape[0] == IMG_ROWS
                        else f" ({IMG_HEIGHT}x{IMG_WIDTH}x3) - wrong capture resolution?")
                raise Exception(
                    f"[{img_path}] has shape {images.shape}, expected (frames, {IMG_ROWS}){hint}"
                )
            if meta.shape[1] != NR_META_ROWS:
                raise Exception(
                    f"[{meta_path}] has shape {meta.shape}, expected (frames, {NR_META_ROWS})"
                )
            if images.shape[0] != meta.shape[0]:
                raise Exception(
                    f"[{img_path}] holds {images.shape[0]} frames but its meta file "
                    f"holds {meta.shape[0]} - the pair is out of sync"
                )

            self._images.append(images)
            self._meta.append(meta)

        # Prefix sums let __getitem__ map a global index onto (file, row)
        # without ever concatenating the files together.
        counts = [img.shape[0] for img in self._images]
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
                  image  uint8 (3, H, W), still 0-255 - the training loop
                         scales it with normalize_images once it is on the GPU
                  action float32 (3,) - steer, throttle, brake
        """
        file_idx = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        row = int(idx - self._offsets[file_idx])

        # np.array copies just this frame - one contiguous block - out of the
        # read-only memory map.
        img = np.array(self._images[file_idx][row]).reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        observation = image_to_raw_tensor(img)

        meta = np.array(self._meta[file_idx][row])
        action = torch.from_numpy(meta[-NR_ACTION_ROWS:])

        return observation, action

    def _load_npy(self, path: str) -> npt.NDArray:
        p = Path(path)
        if not p.exists():
            raise Exception(f"Could not find dataset at [{path}]")

        return np.load(p, mmap_mode="r")


def get_dataloader(data_dir: str, batch_size: int = 256, num_workers: int = 12, shuffle: bool = True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=True,
                persistent_workers=True,
            )
