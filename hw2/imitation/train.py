"""
Train one driving policy (unit 4C): RGB or BEV input, end-to-end control or waypoint output.

Episodes are split by seed, not by frame: consecutive frames of one drive are nearly identical, so a random frame
split would put near-copies of the training frames into validation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import DATA_DIR, INPUT_TYPES, MODEL_DIR, OUTPUT_TYPES  # noqa: E402
from dataset import DrivingDataset  # noqa: E402
from network import DrivingPolicy  # noqa: E402

# Steering errors matter most for staying on the road, and most steer labels are near 0, so steering is weighted up.
CONTROL_LOSS_WEIGHTS = (2.0, 1.0, 1.0)


def loss_and_metrics(
    model: DrivingPolicy,
    batch: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 loss of one batch and its per-quantity errors.

    Parameters
    ----------
    model : DrivingPolicy
        Network being trained or validated.
    batch : list of 4 tensors
        image, speed, control and waypoints as returned by DrivingDataset.
    device : torch.device
        Device the model is on.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        Scalar loss, and the batch's mean absolute errors: steer/throttle/brake for "control", ade_m (all waypoints)
        and fde_m (final waypoint, Euclidean) for "waypoints".
    """
    image, speed, control, waypoints = (t.to(device, non_blocking=True) for t in batch)
    prediction = model(image, speed)
    if model.output_type == 'control':
        errors = (prediction - control).abs().mean(dim=0)
        loss = sum(w * e for w, e in zip(CONTROL_LOSS_WEIGHTS, errors)) / sum(CONTROL_LOSS_WEIGHTS)
        metrics = {name: float(e.detach()) for name, e in zip(('steer', 'throttle', 'brake'), errors)}
    else:
        loss = (prediction - waypoints).abs().mean()
        distance = (prediction - waypoints).detach().norm(dim=-1)
        metrics = {'ade_m': float(distance.mean()), 'fde_m': float(distance[:, -1].mean())}
    return loss, metrics


def run_epoch(
    model: DrivingPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """One pass over a loader; trains if an optimizer is given, otherwise only evaluates.

    Parameters
    ----------
    model : DrivingPolicy
        Network.
    loader : DataLoader
        Batches from DrivingDataset.
    device : torch.device
        Device the model is on.
    optimizer : torch.optim.Optimizer or None
        Optimizer for training; None for validation.

    Returns
    -------
    dict[str, float]
        Sample-weighted mean of the loss and of the metrics from loss_and_metrics.
    """
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, metrics = loss_and_metrics(model, batch, device)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            size = len(batch[0])
            for name, value in {'loss': float(loss), **metrics}.items():
                totals[name] = totals.get(name, 0.0) + value * size
            count += size
    return {name: total / count for name, total in totals.items()}


def main() -> None:
    """Train one input/output combination and save the best model by validation loss."""
    parser = argparse.ArgumentParser(description='Train an imitation learning driving policy.')
    parser.add_argument('--input', choices=INPUT_TYPES, required=True)
    parser.add_argument('--output', choices=OUTPUT_TYPES, required=True)
    parser.add_argument('--episodes', type=Path, default=DATA_DIR / 'episodes')
    parser.add_argument('--val-episodes', type=int, default=3, help='The last N episodes (by seed) are validation.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=0, help='Stop each epoch after N batches (smoke tests).')
    args = parser.parse_args()

    episode_files = sorted(args.episodes.glob('episode_*_rgb.npy'))
    if len(episode_files) <= args.val_episodes:
        raise SystemExit(f'need more than {args.val_episodes} episodes in {args.episodes}, found {len(episode_files)}')
    train_set = DrivingDataset(episode_files[:-args.val_episodes], args.input)
    val_set = DrivingDataset(episode_files[-args.val_episodes:], args.input)
    print('train', train_set.label_statistics())
    print('val  ', val_set.label_statistics())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrivingPolicy(args.output).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers, 'pin_memory': True,
                     'persistent_workers': args.workers > 0}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    if args.max_batches:
        train_loader = [b for _, b in zip(range(args.max_batches), train_loader)]
        val_loader = [b for _, b in zip(range(args.max_batches), val_loader)]

    name = f'{args.input}_{args.output}'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f'{name}.pth'
    history: list[dict[str, float]] = []
    best = float('inf')
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, None)
        scheduler.step()
        history.append({'epoch': epoch + 1, **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()}})
        is_best = val_metrics['loss'] < best
        if is_best:
            best = val_metrics['loss']
            model.save(model_path, args.input)
        print(f'epoch {epoch + 1:2d} ({time.time() - start:4.0f} s) train '
              + ' '.join(f'{k} {v:.4f}' for k, v in train_metrics.items()) + ' | val '
              + ' '.join(f'{k} {v:.4f}' for k, v in val_metrics.items()) + ('  <- best' if is_best else ''),
              flush=True)

    best_epoch = min(history, key=lambda h: h['val_loss'])
    with open(MODEL_DIR / f'{name}_history.json', 'w') as f:
        json.dump({'args': {k: str(v) for k, v in vars(args).items()}, 'best': best_epoch, 'history': history,
                   'train_labels': train_set.label_statistics(), 'val_labels': val_set.label_statistics()}, f,
                  indent=1)
    plt.figure(figsize=(5, 3.5))
    plt.plot([h['train_loss'] for h in history], label='train')
    plt.plot([h['val_loss'] for h in history], label='val')
    plt.xlabel('epoch')
    plt.ylabel('L1 loss')
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f'{name}_loss.png', dpi=100)
    print(f'best epoch {best_epoch["epoch"]}: val loss {best_epoch["val_loss"]:.4f} -> {model_path}')


if __name__ == '__main__':
    main()
