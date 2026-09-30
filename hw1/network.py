from pathlib import Path

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import numpy.typing as npt
import torchvision

def conv_out(size, kernel_size, stride):
    return (size - kernel_size) // stride + 1

def _get_avail_device() -> torch.device:
    return torch.accelerator.current_accelerator() or torch.device("cpu")

# Observation size mandated by the homework (HW1 s1.5): 320 wide by 240 high.
# Arrays are therefore (H, W, C) = (240, 320, 3).
IMG_WIDTH = 320
IMG_HEIGHT = 240

# Action-classes (HW1 s1.2b). Each class decodes to one (steer, throttle, brake)
# triple, matching carla.VehicleControl: steer is a single signed value where
# negative is left, positive is right and 0.0 is straight.
ACTION_CLASSES: tuple[tuple[float, float, float], ...] = (
    ( 0.0, 0.0, 0.0),   # 0 straight + coast
    ( 0.0, 1.0, 0.0),   # 1 straight + gas
    (-0.5, 1.0, 0.0),   # 2 left     + gas
    ( 0.5, 1.0, 0.0),   # 3 right    + gas
    (-0.5, 0.0, 0.0),   # 4 left     + coast
    ( 0.5, 0.0, 0.0),   # 5 right    + coast
    ( 0.0, 0.0, 1.0),   # 6 brake
)
NR_OF_CLASSES = len(ACTION_CLASSES)

# Thresholds that discretise a continuous expert control into action-classes.
# These must suit BOTH experts:
#   keyboard  - steer in 0.1 steps capped at +-0.7; throttle/brake are 0 or 1
#   autopilot - steer is bimodal (p50 0.0006, p95 0.062, max 0.38) and
#               throttle/brake are continuous (throttle mean 0.36, max 0.85)
# 0.05 sits in the autopilot's natural gap between "straight" and a real turn.
STEER_DEADZONE = 0.05
# A low gas threshold matters for autopilot: >0.5 would label 65% of genuinely
# accelerating frames as coasting, since its throttle averages only 0.36.
THROTTLE_THRESHOLD = 0.1
BRAKE_THRESHOLD = 0.5


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, img_height: int = IMG_HEIGHT, img_width: int = IMG_WIDTH):
        """
        Implementation of the network layers. The image size of the input
        observations is 320x240 pixels (width x height).
        """
        super().__init__()

        # Calculate the final linear input size based on multiple 2D convolutions
        h = conv_out(conv_out(img_height, 5, 2), 3, 1)   # 240 -> 118 -> 116
        w = conv_out(conv_out(img_width,  5, 2), 3, 1)   # 320 -> 158 -> 156
        
        self._network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, NR_OF_CLASSES),
        )

        self._softmax = nn.Softmax(dim=1)

    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        return self._network(observation)

    def actions_to_classes(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Map every expert action to its one-hot action-class representation.

        actions:  torch.Tensor of size (N, 3) - (steer, throttle, brake)
        return    torch.Tensor of size (N, C) - one-hot, C = NR_OF_CLASSES
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        steer, throttle, brake = actions[:, 0], actions[:, 1], actions[:, 2]
        braking = brake > BRAKE_THRESHOLD
        gas = throttle > THROTTLE_THRESHOLD
        left = steer < -STEER_DEADZONE
        right = steer > STEER_DEADZONE
        straight = ~left & ~right

        # Defaults to class 0 (straight + coast); braking overrides everything,
        # so a frame that brakes while steering is still labelled "brake".
        idx = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
        idx[straight & gas] = 1
        idx[left & gas] = 2
        idx[right & gas] = 3
        idx[left & ~gas] = 4
        idx[right & ~gas] = 5
        idx[braking] = 6

        return torch.nn.functional.one_hot(idx, NR_OF_CLASSES).float()

    def scores_to_action(self, scores: torch.Tensor) -> tuple[float, float, float]:
        """
        Map the scores predicted by the network to an action-class and return
        the corresponding action.

        scores:   torch.Tensor of size (C,) or (1, C) - logits or probabilities
        return    (steer, throttle, brake), ordered to match carla.VehicleControl
                  and the unpacking in team_code/test_agent.py
        """
        # argmax is invariant under softmax, so this works on logits directly.
        idx = int(scores.reshape(-1, NR_OF_CLASSES).argmax(dim=-1)[0])
        steer, throttle, brake = ACTION_CLASSES[idx]

        return float(steer) * 0.5, float(throttle) * 0.5, float(brake)

    def numpy_img_to_tensor(self, img_in: npt.NDArray[np.float32], copy: bool = False) -> torch.Tensor:
        """
        img_in: float32 (C, H, W)
        return  float32 (C, H, W) image tensor
        """
        return torchvision.transforms.functional.to_tensor(img_in).to(self.get_device())

    def get_device(self) -> torch.device:
        return _get_avail_device()

    @staticmethod
    def load_and_eval(model_path: Path) -> "ClassificationNetwork":
        model = torch.load(model_path, weights_only=False, map_location=_get_avail_device())
        model.eval()
        return model


