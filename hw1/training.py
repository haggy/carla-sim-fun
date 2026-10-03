import os
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from module.hw1.network import ClassificationNetwork, NR_OF_CLASSES, normalize_images
from module.hw1.dataset import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    gpu = torch.device('cuda')

    infer_action = ClassificationNetwork().to(gpu)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)

    nr_epochs = 100
    batch_size = 256
    nr_of_classes = NR_OF_CLASSES
    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size)

    model_path = resolve_model_path(save_path)
    epoch_losses = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(nr_epochs):
        total_loss = 0.0
        nr_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # The loader yields uint8 images; scale them on the GPU so the
            # pinned host->device copy moves 4x fewer bytes than float32.
            batch_in = normalize_images(batch[0].to(gpu, non_blocking=True))
            batch_gt = batch[1].to(gpu, non_blocking=True)

            batch_out = infer_action(batch_in)
            # batch_gt holds the expert's raw (steer, throttle, brake) triple;
            # discretise it into one-hot action-classes for the loss.
            batch_gt = infer_action.actions_to_classes(batch_gt)

            class_weights = torch.tensor([0.90,  0.32, 2.57, 2.80, 6.39, 4.12, 0.60]).to(gpu)
            loss = cross_entropy_loss(batch_out, batch_gt, class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # .item() detaches the value: accumulating the tensor itself would
            # keep every batch's autograd graph alive for the whole epoch.
            total_loss += loss.item()
            nr_batches += 1

        mean_loss = total_loss / max(nr_batches, 1)
        epoch_losses.append(mean_loss)

        # Keep the best model only, so a late-epoch divergence can't overwrite it.
        is_best = mean_loss < best_loss
        if is_best:
            best_loss = mean_loss
            best_epoch = epoch + 1
            torch.save(infer_action, model_path)

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs%s" % (
            epoch + 1, mean_loss, time_left, "\t<- best" if is_best else ""))

    print("\nBest model: epoch %d, loss %.6f -> %s" % (best_epoch, best_loss, model_path))
    plot_path = plot_loss(epoch_losses, model_path)
    print("Loss curve: %s" % plot_path)


def resolve_model_path(save_path):
    """
    Normalise save_path to a writable .pth FILE path.

    torch.save fails on a directory, and the argparse default is "./" - without
    this the run would crash after every epoch had already been trained.
    """
    path = Path(save_path)
    if path.is_dir() or str(save_path).endswith(os.sep):
        path = path / "model.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss(epoch_losses, model_path):
    """Plot training loss against epoch and save it next to the model."""
    import matplotlib
    matplotlib.use("Agg")  # write a file; never open a window that blocks the run
    import matplotlib.pyplot as plt

    plot_path = Path(model_path).with_suffix(".png")
    epochs = range(1, len(epoch_losses) + 1)
    best_idx = min(range(len(epoch_losses)), key=epoch_losses.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(epochs, epoch_losses, color="#2a78d6", linewidth=2, zorder=3)

    # Direct-label the minimum rather than annotating every point.
    ax.scatter([best_idx + 1], [epoch_losses[best_idx]], s=60, color="#2a78d6",
               edgecolor="#fcfcfb", linewidth=2, zorder=4)
    # Flip the label to the inside when the best epoch sits near the right edge,
    # otherwise it collides with the curve and overflows the axes.
    on_right = best_idx > 0.6 * max(len(epoch_losses) - 1, 1)
    ax.annotate("best: %.4f (epoch %d)" % (epoch_losses[best_idx], best_idx + 1),
                xy=(best_idx + 1, epoch_losses[best_idx]),
                xytext=(-10 if on_right else 10, -16), textcoords="offset points",
                ha="right" if on_right else "left", va="top",
                color="#52514e", fontsize=9)

    # Headroom below the minimum so the label has somewhere to sit.
    low, high = min(epoch_losses), max(epoch_losses)
    ax.set_ylim(low - 0.12 * (high - low or 1), high + 0.05 * (high - low or 1))

    ax.set_title("Training loss by epoch", color="#0b0b0b", fontsize=13, pad=12, loc="left")
    ax.set_xlabel("Epoch", color="#52514e", fontsize=10)
    ax.set_ylabel("Mean cross-entropy loss", color="#52514e", fontsize=10)

    # Recessive grid and axes; the data is the only prominent mark.
    ax.grid(True, axis="y", color="#0b0b0b", alpha=0.10, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#52514e")
        ax.spines[side].set_alpha(0.3)
    ax.tick_params(colors="#52514e", labelsize=9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return plot_path


def cross_entropy_loss(batch_out, batch_gt, class_weights=None):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    # TODO: Need to implement my own CE loss function
    return nn.functional.cross_entropy(batch_out, batch_gt, class_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)