import os
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from module.hw1.network import ClassificationNetwork, NR_OF_CLASSES, normalize_images
from module.hw1.dataset import CarlaDataset

# Fraction of samples held out for validation and for the final test. The rest
# (80%) is trained on.
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
# Fixed seed so every run, and every model compared against it, gets the SAME
# test set (it only changes when the dataset itself does).
SPLIT_SEED = 0


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

    # Frames are split at random, so neighbouring (near-identical) frames of one
    # drive can land in different splits; val/test scores will be optimistic.
    dataset = CarlaDataset(data_dir=data_folder)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [1.0 - VAL_FRACTION - TEST_FRACTION, VAL_FRACTION, TEST_FRACTION],
        generator=torch.Generator().manual_seed(SPLIT_SEED))
    print("Split: %d train / %d val / %d test" % (len(train_set), len(val_set), len(test_set)))

    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=12)
    val_loader = make_loader(val_set, batch_size, shuffle=False, num_workers=4)

    class_weights = torch.tensor([0.5775928857749183, 0.3522029372496662, 2.703721223588773, 2.0619288119288117, 6.379092261904762, 2.514960398943972, 0.9802206596924484]).to(gpu)

    model_path = resolve_model_path(save_path)
    epoch_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(nr_epochs):
        infer_action.train()
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

        val_loss, val_acc = evaluate(infer_action, val_loader, class_weights, gpu)
        val_losses.append(val_loss)

        # Keep the model that generalises best rather than the one that fits
        # the training set best, which is what the validation split is for.
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(infer_action, model_path)

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f\t[Val]\tloss: %.6f acc: %.3f\tETA: +%fs%s" % (
            epoch + 1, mean_loss, val_loss, val_acc, time_left, "\t<- best" if is_best else ""))

    # Score the selected model on data that played no part in training or in
    # choosing the epoch - this is the number to report.
    best_model = ClassificationNetwork.load_and_eval(model_path)
    test_loader = make_loader(test_set, batch_size, shuffle=False, num_workers=4, persistent_workers=False)
    test_loss, test_acc = evaluate(best_model, test_loader, class_weights, gpu)

    print("\nBest model: epoch %d, val loss %.6f -> %s" % (best_epoch, best_loss, model_path))
    print("[Test]\tloss: %.6f acc: %.3f" % (test_loss, test_acc))
    plot_path = plot_loss(epoch_losses, model_path, val_losses)
    print("Loss curve: %s" % plot_path)


def make_loader(subset, batch_size, shuffle, num_workers, persistent_workers=True):
    return torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=True,
                persistent_workers=persistent_workers,
            )


@torch.no_grad()
def evaluate(model, loader, class_weights, gpu):
    """
    Mean (class-weighted) loss and plain accuracy of `model` over `loader`.
    Runs in eval mode so dropout is off and BatchNorm uses its running stats,
    then restores the mode the model was in.
    """
    was_training = model.training
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        batch_in = normalize_images(batch[0].to(gpu, non_blocking=True))
        batch_gt = model.actions_to_classes(batch[1].to(gpu, non_blocking=True))
        batch_out = model(batch_in)
        # Weight each batch by its size so a short final batch counts fairly
        total_loss += cross_entropy_loss(batch_out, batch_gt, class_weights).item() * len(batch_in)
        correct += (batch_out.argmax(1) == batch_gt.argmax(1)).sum().item()
        seen += len(batch_in)
    model.train(was_training)
    return total_loss / max(seen, 1), correct / max(seen, 1)


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


def plot_loss(epoch_losses, model_path, val_losses=None):
    """
    Plot training (and validation) loss against epoch and save it next to the
    model. With val_losses, the marked best epoch is the lowest validation loss,
    matching how train() picks the model it saves.
    """
    import matplotlib
    matplotlib.use("Agg")  # write a file; never open a window that blocks the run
    import matplotlib.pyplot as plt

    plot_path = Path(model_path).with_suffix(".png")
    epochs = range(1, len(epoch_losses) + 1)
    best_series = val_losses if val_losses else epoch_losses
    best_color = "#e8710a" if val_losses else "#2a78d6"
    best_idx = min(range(len(best_series)), key=best_series.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(epochs, epoch_losses, color="#2a78d6", linewidth=2, zorder=3)
    if val_losses:
        ax.plot(epochs, val_losses, color="#e8710a", linewidth=2, zorder=3)
        # Direct-label each line at its right end instead of using a legend.
        # The curves often finish close together, so the higher one's label
        # sits above and the lower one's below to keep them from overlapping.
        train_higher = epoch_losses[-1] >= val_losses[-1]
        for series, name, color, above in ((epoch_losses, "train", "#2a78d6", train_higher),
                                           (val_losses, "val", "#e8710a", not train_higher)):
            ax.annotate(name, xy=(len(series), series[-1]), xytext=(6, 7 if above else -7),
                        textcoords="offset points", va="center", color=color, fontsize=9)
        ax.margins(x=0.06)  # room on the right for those labels

    # Direct-label the minimum rather than annotating every point.
    ax.scatter([best_idx + 1], [best_series[best_idx]], s=60, color=best_color,
               edgecolor="#fcfcfb", linewidth=2, zorder=4)
    # Flip the label to the inside when the best epoch sits near the right edge,
    # otherwise it collides with the curve and overflows the axes.
    on_right = best_idx > 0.6 * max(len(epoch_losses) - 1, 1)
    ax.annotate("best%s: %.4f (epoch %d)" % (" val" if val_losses else "", best_series[best_idx], best_idx + 1),
                xy=(best_idx + 1, best_series[best_idx]),
                xytext=(-10 if on_right else 10, -16), textcoords="offset points",
                ha="right" if on_right else "left", va="top",
                color="#52514e", fontsize=9)

    # Headroom below the minimum so the label has somewhere to sit.
    all_losses = epoch_losses + (val_losses or [])
    low, high = min(all_losses), max(all_losses)
    ax.set_ylim(low - 0.12 * (high - low or 1), high + 0.05 * (high - low or 1))

    ax.set_title("Training and validation loss by epoch" if val_losses else "Training loss by epoch", color="#0b0b0b", fontsize=13, pad=12, loc="left")
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