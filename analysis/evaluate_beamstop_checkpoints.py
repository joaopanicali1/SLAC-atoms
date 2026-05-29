#!/usr/bin/env python3
"""Evaluate beamstop checkpoints on a shared validation subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.Dataset.dataset_beamstop import BeamstopDataset
from models.beamstop import BeamstopModel, build_sinc_operator
from training.train_beamstop import (
    FixedMaskSubset,
    build_truncated_R_grid,
    compute_dataset_R_max,
)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv_mask = 1.0 - mask
    denom = inv_mask.sum(dim=-1).clamp_min(1.0)
    return ((pred - target).pow(2) * inv_mask).sum(dim=-1) / denom


def parse_checkpoint(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).parent.name
    return label, Path(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/trajectory_data_cap200_new")
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Checkpoint path, optionally label=path. Can be repeated.")
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--use-mol-features", action="store_true", default=False)
    p.add_argument("--R-cap", type=float, required=True)
    p.add_argument("--R-margin", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--mask-seed", type=int, default=12345)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BeamstopDataset(
        args.data_dir,
        q_min_lo=0.5,
        q_min_hi=1.0,
        max_atoms=75,
        max_modes=219,
        use_conditioning=False,
        use_mol_features=args.use_mol_features,
    )
    print(
        f"Dataset: {len(dataset):,} samples from {len(dataset.loaded_files)} loaded "
        f"molecules ({len(dataset.files)} found, {len(dataset.skipped_files)} skipped)"
    )
    mol_feature_dim = 0
    if args.use_mol_features:
        mol_feature_dim = len(dataset._mol_features[0])
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

    loader = DataLoader(
        FixedMaskSubset(dataset, val_idx, seed=args.mask_seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print(f"\n{'label':<24} {'mean':>12} {'median':>12} {'p95':>12} {'checkpoint'}")
    print("-" * 96)
    for label, ckpt_path in [parse_checkpoint(s) for s in args.checkpoint]:
        model = BeamstopModel(
            B_sinc,
            n_channels=args.n_channels,
            use_conditioning=False,
            use_feature_conditioning=args.use_mol_features,
            mol_feature_dim=mol_feature_dim or 13,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        losses = []
        with torch.no_grad():
            for batch in loader:
                x = batch["delta_S0_masked"].to(device)
                target = batch["delta_S0_norm"].to(device)
                mask = batch["mask"].to(device)
                if args.use_mol_features:
                    pred, _ = model(x, mask, mol_features=batch["mol_features"].to(device))
                else:
                    pred, _ = model(x, mask)
                losses.append(masked_mse(pred, target, mask).cpu().numpy())
        vals = np.concatenate(losses)
        print(
            f"{label:<24} {vals.mean():>12.6e} {np.median(vals):>12.6e} "
            f"{np.quantile(vals, 0.95):>12.6e} {ckpt_path}"
        )


if __name__ == "__main__":
    main()
