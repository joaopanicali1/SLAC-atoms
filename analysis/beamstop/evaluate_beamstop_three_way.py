#!/usr/bin/env python3
"""Compare expanded fine-tuned, feature-conditioned, and NSK on one val subset."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.beamstop.baseline_beamstop import NSKBaseline
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


def masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inv_mask = 1.0 - mask
    denom = np.maximum(inv_mask.sum(axis=-1), 1.0)
    return (np.abs(pred - target) * inv_mask).sum(axis=-1) / denom


def summarize(label: str, values: list[np.ndarray], detail: str = "") -> dict:
    vals = np.concatenate(values)
    return {
        "label": label,
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "p99": float(np.quantile(vals, 0.99)),
        "rmse": float(np.sqrt(vals.mean())),
        "n_samples": int(vals.size),
        "detail": detail,
    }


def load_pyplot():
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    mpl_cache = tmpdir / f"matplotlib-{os.environ.get('USER', 'user')}"
    xdg_cache = tmpdir / f"xdg-cache-{os.environ.get('USER', 'user')}"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_model(path: Path, B_sinc: torch.Tensor, n_channels: int,
               use_features: bool, mol_feature_dim: int, device: torch.device):
    model = BeamstopModel(
        B_sinc,
        n_channels=n_channels,
        use_feature_conditioning=use_features,
        mol_feature_dim=mol_feature_dim,
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def write_example_outputs(outdir: Path, stem: str, picks: list[tuple[str, dict]], plt,
                          include_nsk_delta: bool = False) -> None:
    fields = [
        "kind", "sample_index", "mol_idx", "traj_idx", "frame_idx",
        "feature_conditioned_mse", "expanded_finetune_mse", "NSK_mse",
    ]
    if include_nsk_delta:
        fields.extend(["NSK_minus_conditioned_mse", "NSK_over_conditioned_mse"])

    with (outdir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for kind, ex in picks:
            row = {k: ex[k] for k in fields if k in ex}
            row["kind"] = kind
            if include_nsk_delta:
                row["NSK_minus_conditioned_mse"] = ex["NSK_mse"] - ex["feature_conditioned_mse"]
                row["NSK_over_conditioned_mse"] = ex["NSK_mse"] / max(ex["feature_conditioned_mse"], 1e-12)
            writer.writerow(row)

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.0 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, (kind, ex) in zip(axes, picks):
        q = ex["q"]
        mask = ex["mask"]
        missing = mask < 0.5
        q_min = q[mask > 0.5].min()
        ax.axvspan(q.min(), q_min, color="khaki", alpha=0.35, label="masked")
        ax.plot(q, ex["target"], "k-", lw=1.2, label="ground truth")
        ax.plot(q, np.where(missing, ex["feature_conditioned"], np.nan),
                "C3-", lw=1.0, label="conditioned")
        ax.plot(q, np.where(missing, ex["expanded_finetune"], np.nan),
                "C2-", lw=1.0, alpha=0.9, label="expanded unconditioned")
        ax.plot(q, np.where(missing, ex["NSK"], np.nan),
                "C0-", lw=1.0, alpha=0.9, label="NSK")
        ax.set_ylabel(r"$\Delta S_0^{norm}$")
        ax.grid(alpha=0.25)
        title = (
            f"{kind}: sample={ex['sample_index']} mol={ex['mol_idx']} "
            f"traj={ex['traj_idx']} frame={ex['frame_idx']} "
            f"conditioned MSE={ex['feature_conditioned_mse']:.3e}"
        )
        if include_nsk_delta:
            title += f" NSK MSE={ex['NSK_mse']:.3e}"
        ax.set_title(title, fontsize=9)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("q (Å$^{-1}$)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=150)
    plt.close(fig)


def select_distinct_trajectories(candidates: list[dict], n: int = 3) -> list[dict]:
    selected = []
    seen = set()
    for ex in candidates:
        key = (ex["mol_idx"], ex["traj_idx"])
        if key in seen:
            continue
        selected.append(ex)
        seen.add(key)
        if len(selected) == n:
            break
    return selected


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
    p.add_argument("--output-dir", default=None, help="Optional directory for final CSVs and figures")
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
    direct_mae: list[np.ndarray] = []
    feature_mae: list[np.ndarray] = []
    nsk_mae: list[np.ndarray] = []
    sample_rows = []
    example_cache = []
    q_sse = {
        "expanded_finetune": np.zeros_like(q_np, dtype=np.float64),
        "feature_conditioned": np.zeros_like(q_np, dtype=np.float64),
        "NSK": np.zeros_like(q_np, dtype=np.float64),
    }
    q_count = np.zeros_like(q_np, dtype=np.float64)
    sample_offset = 0
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
            direct_np = direct_pred.cpu().numpy()
            feature_np = feature_pred.cpu().numpy()
            nsk_pred = nsk.predict(q_np, x_np, mask_np)

            batch_direct = masked_mse_np(direct_np, target_np, mask_np)
            batch_feature = masked_mse_np(feature_np, target_np, mask_np)
            batch_nsk = masked_mse_np(nsk_pred, target_np, mask_np)
            direct_losses.append(batch_direct)
            feature_losses.append(batch_feature)
            nsk_losses.append(batch_nsk)
            direct_mae.append(masked_mae_np(direct_np, target_np, mask_np))
            feature_mae.append(masked_mae_np(feature_np, target_np, mask_np))
            nsk_mae.append(masked_mae_np(nsk_pred, target_np, mask_np))

            inv_mask = 1.0 - mask_np
            q_count += inv_mask.sum(axis=0)
            q_sse["expanded_finetune"] += (((direct_np - target_np) ** 2) * inv_mask).sum(axis=0)
            q_sse["feature_conditioned"] += (((feature_np - target_np) ** 2) * inv_mask).sum(axis=0)
            q_sse["NSK"] += (((nsk_pred - target_np) ** 2) * inv_mask).sum(axis=0)

            for i in range(target_np.shape[0]):
                sample_index = sample_offset + i
                sample_rows.append({
                    "sample_index": sample_index,
                    "mol_idx": int(batch["mol_idx"][i]),
                    "traj_idx": int(batch["traj_idx"][i]),
                    "frame_idx": int(batch["frame_idx"][i]),
                    "expanded_finetune_mse": float(batch_direct[i]),
                    "feature_conditioned_mse": float(batch_feature[i]),
                    "NSK_mse": float(batch_nsk[i]),
                })
                example_cache.append({
                    "sample_index": sample_index,
                    "mol_idx": int(batch["mol_idx"][i]),
                    "traj_idx": int(batch["traj_idx"][i]),
                    "frame_idx": int(batch["frame_idx"][i]),
                    "feature_conditioned_mse": float(batch_feature[i]),
                    "expanded_finetune_mse": float(batch_direct[i]),
                    "NSK_mse": float(batch_nsk[i]),
                    "q": q_np.copy(),
                    "mask": mask_np[i].copy(),
                    "target": target_np[i].copy(),
                    "expanded_finetune": direct_np[i].copy(),
                    "feature_conditioned": feature_np[i].copy(),
                    "NSK": nsk_pred[i].copy(),
                })
            sample_offset += target_np.shape[0]

    print(f"\n{'label':<28} {'mean':>12} {'median':>12} {'p95':>12} detail")
    print("-" * 104)
    rows = [
        summarize("expanded_finetune", direct_losses, args.direct_checkpoint),
        summarize("feature_conditioned", feature_losses, args.feature_checkpoint),
        summarize("NSK", nsk_losses, f"R_max={args.nsk_r_max}, dR={args.nsk_dr_basis}"),
    ]
    mae_rows = {
        "expanded_finetune": summarize("expanded_finetune", direct_mae, "MAE"),
        "feature_conditioned": summarize("feature_conditioned", feature_mae, "MAE"),
        "NSK": summarize("NSK", nsk_mae, "MAE"),
    }
    for row in rows:
        print(
            f"{row['label']:<28} {row['mean']:>12.6e} "
            f"{row['median']:>12.6e} {row['p95']:>12.6e} {row['detail']}"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "metric", "mean", "median", "p95", "p99", "rmse", "n_samples", "detail"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"metric": "masked_mse", **row})
            for row in mae_rows.values():
                writer.writerow({"metric": "masked_mae", **row})
        print(f"\nWrote {out}")

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "per_sample_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)

        mol_groups = {}
        for row in sample_rows:
            mol_groups.setdefault(row["mol_idx"], []).append(row)
        with (outdir / "per_molecule_metrics.csv").open("w", newline="") as f:
            fields = ["mol_idx", "n_samples", "expanded_finetune_mse", "feature_conditioned_mse", "NSK_mse"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for mol_idx, group in sorted(mol_groups.items()):
                writer.writerow({
                    "mol_idx": mol_idx,
                    "n_samples": len(group),
                    "expanded_finetune_mse": float(np.mean([r["expanded_finetune_mse"] for r in group])),
                    "feature_conditioned_mse": float(np.mean([r["feature_conditioned_mse"] for r in group])),
                    "NSK_mse": float(np.mean([r["NSK_mse"] for r in group])),
                })

        q_rows = []
        valid = q_count > 0
        for i, q in enumerate(q_np):
            row = {"q": float(q), "n_masked": int(q_count[i])}
            for label in q_sse:
                row[f"{label}_rmse"] = float(np.sqrt(q_sse[label][i] / q_count[i])) if valid[i] else np.nan
            q_rows.append(row)
        with (outdir / "q_resolved_rmse.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
            writer.writeheader()
            writer.writerows(q_rows)

        plt = load_pyplot()
        sample_vals = {k: np.asarray([r[f"{k}_mse"] for r in sample_rows]) for k in q_sse}
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([sample_vals[k] for k in sample_vals], tick_labels=list(sample_vals.keys()), showfliers=False)
        ax.set_yscale("log")
        positive_vals = np.concatenate([vals[vals > 0] for vals in sample_vals.values()])
        if positive_vals.size:
            low, high = float(positive_vals.min()), float(positive_vals.max())
            ax.set_ylim(low / 1.8, high * 1.8)
        ax.set_ylabel("masked MSE")
        fig.tight_layout()
        fig.savefig(outdir / "masked_mse_boxplot.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sample_vals["NSK"], sample_vals["feature_conditioned"], s=4, alpha=0.25)
        lim = [min(sample_vals["NSK"].min(), sample_vals["feature_conditioned"].min()),
               max(sample_vals["NSK"].max(), sample_vals["feature_conditioned"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("NSK masked MSE")
        ax.set_ylabel("feature-conditioned masked MSE")
        fig.tight_layout()
        fig.savefig(outdir / "feature_vs_nsk_scatter.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        q_csv = {label: np.asarray([r[f"{label}_rmse"] for r in q_rows]) for label in q_sse}
        for label, vals in q_csv.items():
            ax.plot(q_np, vals, label=label)
        ax.set_xlim(q_np[valid].min(), q_np[valid].max())
        ax.set_xlabel("q (Å$^{-1}$)")
        ax.set_ylabel("RMSE over masked samples")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "q_resolved_rmse.png", dpi=150)
        plt.close(fig)
        if example_cache:
            worst = sorted(
                example_cache,
                key=lambda r: r["feature_conditioned_mse"],
                reverse=True,
            )
            worst = select_distinct_trajectories(worst, n=3)
            write_example_outputs(
                outdir,
                "representative_examples",
                [(f"worst_{i}", ex) for i, ex in enumerate(worst, start=1)],
                plt,
            )

            nsk_gap = [
                ex for ex in example_cache
                if ex["NSK_mse"] > ex["feature_conditioned_mse"]
            ]
            nsk_gap = sorted(
                nsk_gap,
                key=lambda r: r["NSK_mse"] - r["feature_conditioned_mse"],
                reverse=True,
            )
            nsk_gap = select_distinct_trajectories(nsk_gap, n=3)
            if nsk_gap:
                write_example_outputs(
                    outdir,
                    "conditioned_beats_nsk_examples",
                    [(f"conditioned_beats_nsk_{i}", ex) for i, ex in enumerate(nsk_gap, start=1)],
                    plt,
                    include_nsk_delta=True,
                )
        print(f"Wrote final artifacts under {outdir}")


if __name__ == "__main__":
    main()
