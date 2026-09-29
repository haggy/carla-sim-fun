import torch
import torch.nn as nn


def conv_out(size, kernel_size, stride):
    return (size - kernel_size) // stride + 1

class ClassificationNetwork(torch.nn.Module):
    # TODO: Change this to match the assignment spec!
    def __init__(self, img_height: int = 320, img_width: int = 320):
        """
        Implementation of the network layers. The image size of the input
        observations is 320x240 pixels.
        """
        super().__init__()

        # Calculate the final linear input size based on multiple 2D convolutions
        h = conv_out(conv_out(img_height, 5, 2), 3, 1)   # 320 -> 158 -> 156
        w = conv_out(conv_out(img_width,  5, 2), 3, 1)
        
        self._network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, 4),
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

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        pass

    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        pass


