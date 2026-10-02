from pathlib import Path

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import numpy.typing as npt

def conv_out(size, kernel_size, stride):
    return (size - kernel_size) // stride + 1

def _get_avail_device() -> torch.device:
    return torch.accelerator.current_accelerator() or torch.device("cpu")

# Observation size mandated by the homework (HW1 s1.5): 320 wide by 240 high.
# Arrays are therefore (H, W, C) = (240, 320, 3).
IMG_WIDTH = 320
IMG_HEIGHT = 240

# Images are stored on disk as uint8 (0-255) and scaled to [0,1] here. Every
# consumer - training, the pygame inference loop and the benchmark agent - MUST
# go through normalize_images (directly or via image_to_tensor) so the model
# never sees a different input range than it was trained on.
PIXEL_SCALE = 255.0


def image_to_raw_tensor(img_hwc) -> torch.Tensor:
    """
    Reorder one RGB observation into the network's (C, H, W) layout WITHOUT
    scaling it, keeping its dtype (normally uint8). Moving uint8 to the GPU
    and scaling there copies 4x fewer bytes than sending float32.

    img_hwc: (H, W, 3) numpy array, uint8 0-255 or float carrying 0-255 values
    return   (3, H, W) tensor, same dtype and value range as the input
    """
    return torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1)


def normalize_images(raw: torch.Tensor) -> torch.Tensor:
    """
    Scale raw 0-255 pixel tensors to float32 [0, 1] on whatever device they
    already live on. Call this AFTER moving the tensor to the GPU.

    raw:     (..., 3, H, W) tensor holding 0-255 values
    return   same shape, float32 scaled to [0, 1]
    """
    return raw.float() / PIXEL_SCALE


def image_to_tensor(img_hwc, device: torch.device | None = None) -> torch.Tensor:
    """
    Convert one RGB observation into the tensor the network expects. The raw
    pixels are moved to `device` first and scaled there.

    img_hwc: (H, W, 3) numpy array, uint8 0-255 or float carrying 0-255 values
    device:  where the result should live; None keeps it on the CPU
    return   (3, H, W) float32 tensor scaled to [0, 1]
    """
    return normalize_images(image_to_raw_tensor(img_hwc).to(device))

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

        self._ct = 0.0
        self._cs = 0.0

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
        # torch.load does not call __init__. Initialise lazily rather than crashing
        self.reset_smoothing(force=False)

        # argmax is invariant under softmax, so this works on logits directly.
        idx = int(scores.reshape(-1, NR_OF_CLASSES).argmax(dim=-1)[0])
        steer, throttle, brake = ACTION_CLASSES[idx]

        if throttle > 0.0:
            self._ct = min(self._ct + 1e-1, 1.0)
        else:
            self._ct = 0.0

        steer_coeff = 0.1
        if steer > 0.0:
            self._cs = min(self._cs + steer_coeff, 0.7)
        elif steer < 0.0:
            self._cs = max(self._cs - steer_coeff, -0.7)
        else:
            # Steer is back to 0 so converge to that from either side,
            # clamping at 0 so a small residual cannot overshoot
            if self._cs > 0.0:
                self._cs = max(self._cs - steer_coeff, 0.0)
            else:
                self._cs = min(self._cs + steer_coeff, 0.0)

        return float(self._cs), float(self._ct), float(brake)

    def reset_smoothing(self, force: bool = True) -> None:
        """
        Zero the ramped throttle/steer state used by scores_to_action.

        force=True  always resets - call at the start of each episode/rollout so
                    one run cannot inherit the previous run's momentum.
        force=False only fills in fields that are missing, which is what
                    scores_to_action needs for checkpoints that predate them.
        """
        if force or not hasattr(self, "_ct"):
            self._ct = 0.0
        if force or not hasattr(self, "_cs"):
            self._cs = 0.0

    def numpy_img_to_tensor(self, img_in: npt.NDArray[np.float32], copy: bool = False) -> torch.Tensor:
        """
        img_in: (H, W, 3) uint8 or float array holding 0-255 pixel values
        return  float32 (3, H, W) tensor scaled to [0, 1], on this model's device
        """
        return image_to_tensor(img_in, self.get_device())

    def get_device(self) -> torch.device:
        return _get_avail_device()

    @staticmethod
    def load_and_eval(model_path: Path) -> "ClassificationNetwork":
        model = torch.load(model_path, weights_only=False, map_location=_get_avail_device())
        model.eval()
        return model


