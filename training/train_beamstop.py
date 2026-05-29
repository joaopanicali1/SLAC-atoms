import os
import math
import zipfile
import zlib
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.Dataset.dataset_beamstop import BeamstopDataset
from models.beamstop import BeamstopModel, build_sinc_operator, masked_loss, boundary_continuity_loss


def compute_dataset_R_max(data_dir_or_files, margin=1.2):
    """
    Find the largest interatomic distance present in any frame of any molecule
    by scanning the saved pd_true arrays (which are histograms over R bins).

    Adds a multiplicative margin so the truncated R-grid covers slightly more
    than any observed pair distance. Returning R_max < ~31 Å (the default
    npz grid limit) removes excess R-space degrees of freedom — those extra
    bins can only fit noise, which is especially damaging in the low-q region
    the beamstop model is trying to recover.
    """
    max_R   = 0.0
    R_grid  = None
    if isinstance(data_dir_or_files, (str, Path)):
        files = sorted(Path(data_dir_or_files).glob("mol_*_pd.npz"))
    else:
        files = list(data_dir_or_files)

    n_skipped = 0
    for fpath in files:
        try:
            d = np.load(fpath)
            if R_grid is None:
                R_grid = d["R_grid"]
            nonzero = (d["pd_true"] > 0).any(axis=(0, 1))   # (nR,)
        except (zipfile.BadZipFile, OSError, zlib.error) as e:
            print(f"  WARNING: skipping {Path(fpath).name} for R_max — {e}")
            n_skipped += 1
            continue

        idx = np.where(nonzero)[0]
        if len(idx) > 0:
            max_R = max(max_R, float(R_grid[idx.max()]))
    if n_skipped:
        print(f"  {n_skipped} file(s) skipped while computing R_max.")
    if R_grid is None:
        raise RuntimeError("Could not compute R_max: no readable pd_true arrays found.")
    return max_R * margin


def build_truncated_R_grid(R_max, dR=0.025, R_min=0.025):
    """Uniform R-grid from R_min up to R_max."""
    return np.arange(R_min, R_max + dR, dR, dtype=np.float32)


class FixedMaskSubset(Dataset):
    """Subset wrapper with a fixed, deterministic q_min per base sample."""
    def __init__(self, dataset, indices, seed=12345):
        self.dataset = dataset
        self.indices = list(indices)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.q_min = rng.uniform(
            dataset.q_min_lo,
            dataset.q_min_hi,
            size=len(self.indices),
        ).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        fi, ti, fr = self.dataset._index[base_idx]

        delta_S0_norm = torch.from_numpy(self.dataset._delta_S0[fi][ti, fr])
        q_min_idx = int(torch.searchsorted(
            self.dataset.q,
            torch.tensor(self.q_min[idx], dtype=self.dataset.q.dtype),
            right=False,
        ))
        q_min_idx = max(0, min(q_min_idx, len(self.dataset.q) - 1))

        mask = torch.ones(len(self.dataset.q), dtype=torch.float32)
        mask[:q_min_idx] = 0.0

        sample = {
            "delta_S0_masked": delta_S0_norm * mask,
            "delta_S0_norm":   delta_S0_norm,
            "mask":            mask,
            "mol_idx":         fi,
            "traj_idx":        ti,
            "frame_idx":       fr,
        }
        if self.dataset.use_mol_features:
            sample["mol_features"] = torch.from_numpy(self.dataset._mol_features[fi])
        return sample


def warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps):
    """Linear warmup for n_warmup_steps, then cosine decay to 0."""
    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / max(1, n_warmup_steps)
        progress = (step - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, device,
              lambda_smooth, lambda_boundary, lambda_pd,
              train=True, use_mol_features=False):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches  = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            delta_S0_masked = batch["delta_S0_masked"].to(device)
            delta_S0_norm   = batch["delta_S0_norm"].to(device)
            mask            = batch["mask"].to(device)

            if use_mol_features:
                pred, delta_pd = model(
                    delta_S0_masked, mask,
                    mol_features=batch["mol_features"].to(device),
                )
            else:
                pred, delta_pd = model(delta_S0_masked, mask)

            loss = masked_loss(pred, delta_S0_norm, mask,
                               lambda_smooth=lambda_smooth)
            if lambda_boundary > 0:
                loss = loss + lambda_boundary * boundary_continuity_loss(
                    pred, delta_S0_norm, mask)
            if lambda_pd > 0:
                # Tikhonov regularisation on the R-space intermediate.
                # Penalises large pair-density coefficients, which suppresses
                # high-R (fast q-space oscillating) modes — the same mechanism
                # that makes NSK outputs smooth.
                loss = loss + lambda_pd * delta_pd.pow(2).mean()

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
    use_mol_features=False,
    mol_feature_dim=13,
    lambda_smooth=1e-3,
    lambda_boundary=0.0,
    lambda_pd=0.0,
    checkpoint_dir="checkpoints/beamstop",
    init_checkpoint=None,
    max_runtime_seconds=None,
    stop_buffer_seconds=600.0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on {device}")
    print(f"use_mol_features : {use_mol_features}")
    print(f"lambda_smooth    : {lambda_smooth}")
    print(f"lambda_boundary  : {lambda_boundary}")
    print(f"lambda_pd        : {lambda_pd}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = BeamstopModel(
        B_sinc, n_channels=n_channels, kernel_size=5,
        use_feature_conditioning=use_mol_features,
        mol_feature_dim=mol_feature_dim,
    ).to(device)

    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Initialized model weights from {init_checkpoint}")
        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(f"  source epoch    : {checkpoint['epoch']}")
            if "val_loss" in checkpoint:
                print(f"  source val loss : {checkpoint['val_loss']:.6f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")
    print(f"B_sinc shape     : {tuple(B_sinc.shape)}  ({B_sinc.nbytes / 1e6:.1f} MB)")

    optimizer = AdamW(model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader)
    n_total_steps   = n_epochs * steps_per_epoch
    n_warmup_steps  = n_warmup_epochs * steps_per_epoch

    scheduler = warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps)
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    train_t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device,
                               lambda_smooth, lambda_boundary, lambda_pd, train=True,
                               use_mol_features=use_mol_features)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, device,
                               lambda_smooth, lambda_boundary, lambda_pd, train=False,
                               use_mol_features=use_mol_features)
        epoch_seconds = time.perf_counter() - epoch_t0

        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:03d}/{n_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"lr: {curr_lr:.2e} | "
            f"time: {epoch_seconds:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  New best model saved (val loss = {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/5 epochs")
            if epochs_no_improve >= 5:
                print(f"Early stopping triggered — val loss has not improved "
                      f"for 5 consecutive epochs. Best: {best_val_loss:.6f}")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))

        if max_runtime_seconds is not None:
            elapsed = time.perf_counter() - train_t0
            projected_next = elapsed + epoch_seconds + stop_buffer_seconds
            if projected_next >= max_runtime_seconds:
                print(
                    "Time-budget stop triggered — returning best checkpoint "
                    f"before walltime. elapsed={elapsed:.1f}s, "
                    f"last_epoch={epoch_seconds:.1f}s, "
                    f"budget={max_runtime_seconds:.1f}s, "
                    f"buffer={stop_buffer_seconds:.1f}s, "
                    f"best={best_val_loss:.6f}",
                    flush=True,
                )
                break

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model, best_val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-channels",       type=int,   default=32,                      help="Hidden channels in encoder CNNs")
    parser.add_argument("--data-dir",         type=str,   default="data/trajectory_data_cap200", help="Directory containing mol_*_pd.npz trajectory files")
    parser.add_argument("--checkpoint-dir",   type=str,   default="checkpoints/beamstop",  help="Directory to save checkpoints")
    parser.add_argument("--use-mol-features", action="store_true", default=False,           help="Enable FiLM conditioning on scalar molecular features")
    parser.add_argument("--lambda-smooth",    type=float, default=0.0,                     help="Smoothness penalty weight (default 0; B_sinc is band-limited by construction)")
    parser.add_argument("--R-margin",         type=float, default=1.2,                     help="Multiplicative margin on max interatomic distance for R-grid truncation")
    parser.add_argument("--R-cap",            type=float, default=10.0,                    help="Hard cap on R_max (Å); architectural band-limit — smaller cap = smoother low-q output")
    parser.add_argument("--lambda-boundary",  type=float, default=0.0,                     help="Weight on slope+curvature continuity at q_min (0 = off)")
    parser.add_argument("--lambda-pd",        type=float, default=0.0,                     help="Tikhonov weight on R-space intermediate delta_pd (suppresses high-R fast-oscillating modes; try 1e-3)")
    parser.add_argument("--no-pin-memory",    action="store_true", default=False,           help="Disable pin_memory in DataLoader (use on nodes where IPC is unreliable)")
    parser.add_argument("--num-workers",      type=int,   default=4,                        help="DataLoader worker processes (0 disables IPC entirely — needed for large-tensor batches on some nodes)")
    parser.add_argument("--batch-size",       type=int,   default=32,                       help="Batch size for both loaders")
    parser.add_argument("--lr",               type=float, default=1e-3,                     help="Peak learning rate after warmup")
    parser.add_argument("--n-epochs",         type=int,   default=100,                      help="Maximum training epochs")
    parser.add_argument("--n-warmup-epochs",  type=int,   default=5,                        help="Linear LR warmup epochs")
    parser.add_argument("--init-checkpoint",  type=str,   default=None,                     help="Initialize model weights from this checkpoint, but start a fresh optimizer/scheduler")
    parser.add_argument("--max-runtime-min",  type=float, default=None,                     help="Stop cleanly before this many training minutes elapse")
    parser.add_argument("--stop-buffer-min",  type=float, default=10.0,                     help="Runtime buffer minutes reserved before max runtime")
    args = parser.parse_args()

    dataset = BeamstopDataset(args.data_dir, q_min_lo=0.5, q_min_hi=1.0,
                              max_atoms=75, max_modes=219,
                              use_conditioning=False,
                              use_mol_features=args.use_mol_features)
    print(
        f"Dataset: {len(dataset):,} samples from {len(dataset.loaded_files)} loaded "
        f"molecules ({len(dataset.files)} found, {len(dataset.skipped_files)} skipped)"
    )
    mol_feature_dim = 0
    if args.use_mol_features:
        mol_feature_dim = len(dataset._mol_features[0])
        print(f"mol_feature_dim  : {mol_feature_dim}")

    # Build data-driven R-grid: truncate to ~1.2× max interatomic distance.
    # Removes excess R-space DoF that can only fit low-q noise.
    print("Scanning dataset for max interatomic distance ...")
    R_max_data = compute_dataset_R_max(dataset.loaded_files, margin=args.R_margin)
    R_max      = min(R_max_data, args.R_cap)
    dR         = float(dataset.R_grid[1] - dataset.R_grid[0])
    R_grid_truncated = build_truncated_R_grid(R_max, dR=dR)
    print(f"R_max_data (margin={args.R_margin}): {R_max_data:.2f} Å  |  R_cap: {args.R_cap:.2f} Å  |  "
          f"R_max used: {R_max:.2f} Å  |  R-grid: {len(R_grid_truncated)} pts (was {len(dataset.R_grid)})")

    q_np = dataset.q.numpy()
    print("Building B_sinc ...")
    B_sinc = build_sinc_operator(q_np, R_grid_truncated)

    # File-level train/val split.
    n_files = len(dataset.loaded_files)
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

    pin_memory  = not args.no_pin_memory
    num_workers = args.num_workers
    print(f"pin_memory       : {pin_memory}")
    print(f"num_workers      : {num_workers}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        FixedMaskSubset(dataset, val_idx, seed=12345),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    train(train_loader=train_loader, val_loader=val_loader,
          B_sinc=B_sinc,
          n_channels=args.n_channels, checkpoint_dir=args.checkpoint_dir,
          use_mol_features=args.use_mol_features,
          mol_feature_dim=mol_feature_dim,
          lambda_smooth=args.lambda_smooth,
          lambda_boundary=args.lambda_boundary,
          lambda_pd=args.lambda_pd,
          lr=args.lr,
          n_epochs=args.n_epochs,
          n_warmup_epochs=args.n_warmup_epochs,
          init_checkpoint=args.init_checkpoint,
          max_runtime_seconds=None if args.max_runtime_min is None else args.max_runtime_min * 60.0,
          stop_buffer_seconds=args.stop_buffer_min * 60.0)
