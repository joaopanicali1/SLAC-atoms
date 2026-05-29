#!/usr/bin/env python3
"""Compare expanded fine-tuned, feature-conditioned, and NSK on one val subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.baseline_beamstop import NSKBaseline
from data.Dataset.dataset_beamstop import BeamstopDataset
from models.beamstop import BeamstopModel, build_sinc_operator
from training.train_beamstop import (
    FixedMaskSubset,
    build_truncated_R_grid,
    compute_dataset_R_max,
)


def masked_mse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inv_mask = 1.0 - mask
    denom = np.maximum(inv_mask.sum(axis=-1), 1.0)
    return ((pred - target) ** 2 * inv_mask).sum(axis=-1) / denom


def summarize(label: str, values: list[np.ndarray], detail: str = "") -> dict:
    vals = np.concatenate(values)
    return {
        "label": label,
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "n_samples": int(vals.size),
        "detail": detail,
    }


def load_model(path: Path, B_sinc: torch.Tensor, n_channels: int,
               use_features: bool, mol_feature_dim: int, device: torch.device):
    model = BeamstopModel(
        B_sinc,
        n_channels=n_channels,
        use_conditioning=False,
        use_feature_conditioning=use_features,
        mol_feature_dim=mol_feature_dim,
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/trajectory_data_cap200_new")
    p.add_argument("--direct-checkpoint", required=True)
    p.add_argument("--feature-checkpoint", required=True)
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--R-cap", type=float, required=True)
    p.add_argument("--R-margin", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--csv", default=None, help="Optional output CSV for summary metrics")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--nsk-r-max", type=float, default=10.0)
    p.add_argument("--nsk-dr-basis", type=float, default=0.1)
    p.add_argument("--nsk-eps", type=float, default=1e-3)
    p.add_argument("--nsk-lambda-R", type=float, default=1.0)
    p.add_argument("--nsk-lambda-cont", type=float, default=10.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BeamstopDataset(
        args.data_dir,
        q_min_lo=0.5,
        q_min_hi=1.0,
        max_atoms=75,
        max_modes=219,
        use_conditioning=False,
        use_mol_features=True,
    )
    mol_feature_dim = len(dataset._mol_features[0])
    print(
        f"Dataset: {len(dataset):,} samples from {len(dataset.loaded_files)} loaded "
        f"molecules ({len(dataset.files)} found, {len(dataset.skipped_files)} skipped)"
    )
    print(f"mol_feature_dim={mol_feature_dim}")

    n_files = len(dataset.loaded_files)
    n_val = max(1, int(0.1 * n_files))
    rng = np.random.default_rng(args.split_seed)
    file_indices = np.arange(n_files)
    rng.shuffle(file_indices)
    val_files = set(file_indices[:n_val])
    val_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]
    if args.max_samples:
        val_idx = val_idx[:args.max_samples]
    print(f"Evaluating {len(val_idx):,} validation samples on {device}")

    q_np = dataset.q.numpy()
    R_max_data = compute_dataset_R_max(dataset.loaded_files, margin=args.R_margin)
    R_max = min(R_max_data, args.R_cap)
    dR = float(dataset.R_grid[1] - dataset.R_grid[0])
    R_grid = build_truncated_R_grid(R_max, dR=dR)
    B_sinc = build_sinc_operator(q_np, R_grid, verbose=False).to(device)
    print(f"R_max_data={R_max_data:.2f} Å R_cap={args.R_cap:.2f} Å R_max={R_max:.2f} Å")

    direct = load_model(Path(args.direct_checkpoint), B_sinc, args.n_channels,
                        use_features=False, mol_feature_dim=mol_feature_dim,
                        device=device)
    feature = load_model(Path(args.feature_checkpoint), B_sinc, args.n_channels,
                         use_features=True, mol_feature_dim=mol_feature_dim,
                         device=device)
    nsk = NSKBaseline(
        R_max=args.nsk_r_max,
        dR=args.nsk_dr_basis,
        eps=args.nsk_eps,
        lambda_R=args.nsk_lambda_R,
        lambda_cont=args.nsk_lambda_cont,
    )

    loader = DataLoader(
        FixedMaskSubset(dataset, val_idx, seed=args.mask_seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    direct_losses: list[np.ndarray] = []
    feature_losses: list[np.ndarray] = []
    nsk_losses: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["delta_S0_masked"].to(device)
            target = batch["delta_S0_norm"].to(device)
            mask = batch["mask"].to(device)
            features = batch["mol_features"].to(device)

            direct_pred, _ = direct(x, mask)
            feature_pred, _ = feature(x, mask, mol_features=features)

            x_np = batch["delta_S0_masked"].numpy()
            target_np = batch["delta_S0_norm"].numpy()
            mask_np = batch["mask"].numpy()
            nsk_pred = nsk.predict(q_np, x_np, mask_np)

            direct_losses.append(masked_mse_np(direct_pred.cpu().numpy(), target_np, mask_np))
            feature_losses.append(masked_mse_np(feature_pred.cpu().numpy(), target_np, mask_np))
            nsk_losses.append(masked_mse_np(nsk_pred, target_np, mask_np))

    print(f"\n{'label':<28} {'mean':>12} {'median':>12} {'p95':>12} detail")
    print("-" * 104)
    rows = [
        summarize("expanded_finetune", direct_losses, args.direct_checkpoint),
        summarize("feature_conditioned", feature_losses, args.feature_checkpoint),
        summarize("NSK", nsk_losses, f"R_max={args.nsk_r_max}, dR={args.nsk_dr_basis}"),
    ]
    for row in rows:
        print(
            f"{row['label']:<28} {row['mean']:>12.6e} "
            f"{row['median']:>12.6e} {row['p95']:>12.6e} {row['detail']}"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "mean", "median", "p95", "n_samples", "detail"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
