"""
Driving policies for unit 4C: an ImageNet-pretrained ResNet-18 on the RGB or BEV image, fused with the current speed,
and a head that outputs either the controls directly (end-to-end) or future waypoints (waypoint-based).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from common import NUM_WAYPOINTS, OUTPUT_TYPES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DrivingPolicy(nn.Module):
    """Image + speed -> (steer, throttle, brake) or (NumWaypoints x 2) future waypoints."""

    def __init__(self, output_type: str, pretrained: bool = True) -> None:
        """Build the network.

        Parameters
        ----------
        output_type : str
            "control" for end-to-end control, "waypoints" for future waypoints.
        pretrained : bool
            Start the backbone from ImageNet weights.
        """
        super().__init__()
        if output_type not in OUTPUT_TYPES:
            raise ValueError(f'unknown output type {output_type!r}; expected one of {OUTPUT_TYPES}')
        self.output_type = output_type

        # resnet18() without arguments is untrained in every torchvision; `weights=` needs torchvision >= 0.13, which
        # the leaderboard kit's Python 3.7 setup may not have.
        if pretrained:
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()
        self.speed_encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        num_outputs = 3 if output_type == 'control' else 2 * NUM_WAYPOINTS
        self.head = nn.Sequential(nn.Linear(512 + 32, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_outputs))
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Predict controls or waypoints.

        Parameters
        ----------
        image : (Batch x 3 x ImageHeight x ImageWidth) tensor
            uint8 RGB or BEV images, 0-255.
        speed : (Batch,) tensor
            Speeds in m/s.

        Returns
        -------
        (Batch x 3) or (Batch x NumWaypoints x 2) tensor
            (steer in [-1, 1], throttle in [0, 1], brake in [0, 1]) for "control"; (forward, right) metres in the
            vehicle frame for "waypoints".
        """
        features = self.backbone((image.float() / 255.0 - self.mean) / self.std)
        # Speeds are ~0-10 m/s; scaling keeps the input near unit range.
        fused = torch.cat([features, self.speed_encoder(speed.view(-1, 1) / 10.0)], dim=1)
        out = self.head(fused)
        if self.output_type == 'control':
            return torch.cat([torch.tanh(out[:, :1]), torch.sigmoid(out[:, 1:])], dim=1)
        return out.view(-1, NUM_WAYPOINTS, 2)

    def save(self, path: Path, input_type: str, legacy_format: bool = False) -> None:
        """Save the weights together with the configuration needed to rebuild the network.

        Parameters
        ----------
        path : Path
            Output .pth file.
        input_type : str
            "rgb" or "bev", stored so evaluation feeds the right image.
        legacy_format : bool
            Write torch's pre-1.6 format, which every torch >= 1.0 can read (the leaderboard kit).
        """
        torch.save({'output_type': self.output_type, 'input_type': input_type, 'state_dict': self.state_dict()},
                   str(path), _use_new_zipfile_serialization=not legacy_format)

    @staticmethod
    def load(path: Path, device: torch.device) -> tuple['DrivingPolicy', str]:
        """Load a network written by save(), in eval mode.

        Parameters
        ----------
        path : Path
            .pth file from save().
        device : torch.device
            Device to put the network on.

        Returns
        -------
        tuple[DrivingPolicy, str]
            The network and its input type ("rgb" or "bev").
        """
        checkpoint = torch.load(str(path), map_location=device)
        model = DrivingPolicy(checkpoint['output_type'], pretrained=False)
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device).eval(), checkpoint['input_type']
