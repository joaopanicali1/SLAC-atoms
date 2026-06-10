import argparse
import os
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models.unet.dataset.dataset import MolScatteringDataset
from models.unet.code.unet import PairDensityUNet

PROJECT_DIR = Path(__file__).resolve().parents[3]


def warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps):
    """
    Linear warmup for n_warmup_steps, then cosine decay to 0.
    Returns a LambdaLR schedule that scales the base lr
    """
    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / max(1, n_warmup_steps)
        progress = (step - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def run_epoch(model, loader, optimizer, scheduler, device, train=True, max_steps=None):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for i, batch in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            pd_limited = batch["pd_limited"].to(device)
            pd_true = batch["pd_true"].to(device)
            V_vib = batch["V_vib"].to(device)
            omegas = batch["omegas_ASE"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            mode_mask = batch["mode_mask"].to(device)

            pred = model(pd_limited, V_vib, omegas, atom_mask, mode_mask)
            loss = nn.functional.mse_loss(pred, pd_true)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            if i % 20 == 0:
                print(f"  step {i}/{len(loader)} | loss {loss.item():.4f}", flush=True)
    return total_loss / n_batches

def train(train_loader, val_loader, n_epochs=100, lr=1e-3, n_warmup_epochs=5, checkpoint_dir="checkpoints", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = PairDensityUNet(
        n_R=1257,
        features=[32, 64, 128, 256],
        d_model=256,
        n_heads=8,
        max_atoms=50,
        max_modes=144,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader)
    n_total_steps = n_epochs * steps_per_epoch
    n_warmup_steps = n_warmup_epochs * steps_per_epoch

    scheduler = warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, scheduler, device, train=False)

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
                "val_loss": val_loss
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  New best model saved (val loss = {val_loss:.6f})")

        # saving checkpoint every epoch so we don't lose progress if job times out
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))
    
    print(f"Training complete. Best val loss: {best_val_loss:6f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the UNet pair-density model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "trajectory_data_cap200_new",
        help="Directory containing mol_*_pd.npz trajectory files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "unet",
        help="Directory for best_model.pt and latest_checkpoint.pt.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = MolScatteringDataset(
        data_dir=args.data_dir,
        max_atoms=50,
        max_modes=144,
    )

    n_files = len(dataset.files)
    n_val = int(0.1 * n_files)

    rng = np.random.default_rng(42)
    file_indices = np.arange(n_files)
    rng.shuffle(file_indices)
    split_dir = PROJECT_DIR / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "split_unet.npy", file_indices)

    val_files = set(file_indices[:n_val])
    train_files = set(file_indices[n_val:])

    train_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in train_files]
    val_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        n_warmup_epochs=args.n_warmup_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
