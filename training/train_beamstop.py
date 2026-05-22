import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.Dataset.dataset_beamstop import BeamstopDataset
from models.beamstop import BeamstopModel, build_sinc_operator, smoothness_loss

LAMBDA_SMOOTH = 1e-3   # weight of second-derivative penalty in total loss


def warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps):
    """Linear warmup for n_warmup_steps, then cosine decay to 0."""
    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / max(1, n_warmup_steps)
        progress = (step - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches  = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            delta_S0_masked = batch["delta_S0_masked"].to(device)  # (batch, nq)
            delta_S0_norm   = batch["delta_S0_norm"].to(device)    # (batch, nq)
            mask            = batch["mask"].to(device)             # (batch, nq)

            pred = model(delta_S0_masked, mask)                    # (batch, nq)

            mse  = nn.functional.mse_loss(pred, delta_S0_norm)
            smooth = smoothness_loss(pred, mask)
            loss = mse + LAMBDA_SMOOTH * smooth

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


def train(
    train_loader,
    val_loader,
    B_sinc,
    n_epochs=100,
    lr=1e-3,
    n_warmup_epochs=5,
    n_channels=32,
    checkpoint_dir="checkpoints/beamstop",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = BeamstopModel(B_sinc, n_channels=n_channels, kernel_size=5).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")
    print(f"B_sinc shape     : {tuple(B_sinc.shape)}  ({B_sinc.nbytes / 1e6:.1f} MB)")
    print(f"Smoothness weight: {LAMBDA_SMOOTH}")

    optimizer = AdamW(model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader)
    n_total_steps   = n_epochs * steps_per_epoch
    n_warmup_steps  = n_warmup_epochs * steps_per_epoch

    scheduler = warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, device, train=False)

        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:03d}/{n_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"lr: {curr_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  New best model saved (val loss = {val_loss:.6f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-channels",      type=int,   default=32,                      help="Hidden channels in encoder CNNs")
    parser.add_argument("--checkpoint-dir",  type=str,   default="checkpoints/beamstop",  help="Directory to save checkpoints")
    args = parser.parse_args()

    data_dir = "data/trajectory_data_cap200"

    dataset = BeamstopDataset(data_dir, q_min_lo=0.5, q_min_hi=1.0)
    print(f"Dataset: {len(dataset):,} samples from {len(dataset.files)} molecules")

    # Build sinc operator from dataset grids (same for all molecules).
    q_np   = dataset.q.numpy()
    R_np   = dataset.R_grid.numpy()
    print("Building B_sinc ...")
    B_sinc = build_sinc_operator(q_np, R_np)

    # File-level train/val split.
    n_files = len(dataset.files)
    n_val   = max(1, int(0.1 * n_files))

    rng = np.random.default_rng(42)
    file_indices = np.arange(n_files)
    rng.shuffle(file_indices)
    np.save("split_beamstop.npy", file_indices)

    val_files   = set(file_indices[:n_val])
    train_files = set(file_indices[n_val:])

    train_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in train_files]
    val_idx   = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]

    print(f"Train samples: {len(train_idx):,}  |  Val samples: {len(val_idx):,}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train(train_loader=train_loader, val_loader=val_loader, B_sinc=B_sinc,
          n_channels=args.n_channels, checkpoint_dir=args.checkpoint_dir)
