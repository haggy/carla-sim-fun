import glob
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from module.hw1.network import IMG_HEIGHT, IMG_WIDTH


class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.npy') #need to change to your data format

        if not self.data_list:
            raise Exception(f"No numpy datasets were found at [{data_dir}]")

        datasets = [self._load_npy(p) for p in self.data_list]
        print(f"Resolved and loaded {len(datasets)} datasets")

        self._vectorized_data = np.concat(datasets, axis=1)
        print(f"Datasets have been vectorized. Shape: {self._vectorized_data.shape}")

        self.transform_obs = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform_action = transforms.Compose([
            #transforms.ToTensor(),
        ])

    def __len__(self):
        return self._vectorized_data.shape[1]

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        sample: npt.NDArray[np.float32] = self._vectorized_data[:, idx].reshape((-1, 1))

        # Split the observation from the actions
        observation = sample[:-8, :].reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        observation = self.transform_obs(observation)
        action = torch.from_numpy(sample[-4:, :].squeeze())

        return observation, action

    def _load_npy(self, path: str) -> npt.NDArray[np.float32]:
        p = Path(path)
        if not p.exists():
            raise Exception(f"Could not find dataset at [{path}]")

        return np.load(p)


def get_dataloader(data_dir: str, batch_size: int = 64, num_workers: int = 4, shuffle: bool = True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    