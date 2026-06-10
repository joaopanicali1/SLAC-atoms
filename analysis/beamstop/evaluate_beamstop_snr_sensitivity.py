#!/usr/bin/env python3
"""Evaluate beamstop reconstruction sensitivity to Gaussian noise in Delta S0."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analysis.beamstop.baseline_beamstop import NSKBaseline
from models.beamstop.dataset.dataset_beamstop import BeamstopDataset
from models.beamstop.code.beamstop import BeamstopModel, build_sinc_operator
from models.beamstop.training.train_beamstop import build_truncated_R_grid, compute_dataset_R_max


CONDITIONED = "beamstop_conditioned"
UNCONDITIONED = "beamstop_unconditioned"
NSK = "NSK"
DISPLAY_LABELS = {
    CONDITIONED: "beamstop conditioned",
    UNCONDITIONED: "beamstop unconditioned",
    NSK: "NSK",
}


class FixedMaskNoisySubset(Dataset):
    """Subset with deterministic q_min masks and deterministic Gaussian input noise."""

    def __init__(
        self,
        dataset: BeamstopDataset,
        indices: list[int],
        q_min: np.ndarray,
        snr: float,
        noise_seed: int = 20260531,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.q_min = np.asarray(q_min, dtype=np.float32)
        if len(self.q_min) != len(self.indices):
            raise ValueError("q_min must have one value per subset index")
        self.snr = float(snr)
        if self.snr <= 0:
            raise ValueError("SNR values must be positive")
        self.noise_seed = int(noise_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
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

        x = delta_S0_norm * mask
        observed = mask.bool()
        if observed.any():
            observed_signal = delta_S0_norm[observed].numpy()
            signal_rms = float(np.sqrt(np.mean(observed_signal ** 2)))
            if signal_rms > 0:
                rng = np.random.default_rng(self.noise_seed + idx)
                noise = rng.normal(
                    loc=0.0,
                    scale=signal_rms / self.snr,
                    size=int(observed.sum().item()),
                ).astype(np.float32)
                x = x.clone()
                x[observed] += torch.from_numpy(noise)

        sample = {
            "delta_S0_masked": x,
            "delta_S0_norm": delta_S0_norm,
            "mask": mask,
            "mol_idx": fi,
            "traj_idx": ti,
            "frame_idx": fr,
        }
        if self.dataset.use_mol_features:
            sample["mol_features"] = torch.from_numpy(self.dataset._mol_features[fi])
        return sample


def masked_mse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inv_mask = 1.0 - mask
    denom = np.maximum(inv_mask.sum(axis=-1), 1.0)
    return ((pred - target) ** 2 * inv_mask).sum(axis=-1) / denom


def masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inv_mask = 1.0 - mask
    denom = np.maximum(inv_mask.sum(axis=-1), 1.0)
    return (np.abs(pred - target) * inv_mask).sum(axis=-1) / denom


def summarize(snr: float, model: str, mse_chunks: list[np.ndarray],
              mae_chunks: list[np.ndarray]) -> dict:
    mse = np.concatenate(mse_chunks)
    mae = np.concatenate(mae_chunks)
    return {
        "snr": float(snr),
        "model": DISPLAY_LABELS[model],
        "masked_mse_mean": float(mse.mean()),
        "masked_mse_median": float(np.median(mse)),
        "masked_mse_p95": float(np.quantile(mse, 0.95)),
        "masked_mse_p99": float(np.quantile(mse, 0.99)),
        "masked_rmse": float(np.sqrt(mse.mean())),
        "masked_mae_mean": float(mae.mean()),
        "masked_mae_median": float(np.median(mae)),
        "masked_mae_p95": float(np.quantile(mae, 0.95)),
        "n_samples": int(mse.size),
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
               use_features: bool, mol_feature_dim: int, device: torch.device) -> BeamstopModel:
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


def parse_snrs(args: argparse.Namespace) -> list[float]:
    values = [float(x) for x in args.snrs.split(",")]
    if any(v <= 0 for v in values):
        raise ValueError("All SNR values must be positive")
    return sorted(set(values))


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
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--noise-seed", type=int, default=20260531)
    p.add_argument("--qmin-lo", type=float, default=0.5)
    p.add_argument("--qmin-hi", type=float, default=1.0)
    p.add_argument("--snrs", default="1,2,5,10,20,50,100",
                   help="Comma-separated SNR values, defined as observed-signal RMS / noise RMS.")
    p.add_argument("--nsk-r-max", type=float, default=10.0)
    p.add_argument("--nsk-dr-basis", type=float, default=0.1)
    p.add_argument("--nsk-eps", type=float, default=1e-3)
    p.add_argument("--nsk-lambda-R", type=float, default=1.0)
    p.add_argument("--nsk-lambda-cont", type=float, default=10.0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BeamstopDataset(
        args.data_dir,
        q_min_lo=args.qmin_lo,
        q_min_hi=args.qmin_hi,
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

    q_rng = np.random.default_rng(args.mask_seed)
    q_min = q_rng.uniform(args.qmin_lo, args.qmin_hi, size=len(val_idx)).astype(np.float32)

    q_np = dataset.q.numpy()
    R_max_data = compute_dataset_R_max(dataset.loaded_files, margin=args.R_margin)
    R_max = min(R_max_data, args.R_cap)
    dR = float(dataset.R_grid[1] - dataset.R_grid[0])
    R_grid = build_truncated_R_grid(R_max, dR=dR)
    B_sinc = build_sinc_operator(q_np, R_grid, verbose=False).to(device)
    print(f"R_max_data={R_max_data:.2f} Å R_cap={args.R_cap:.2f} Å R_max={R_max:.2f} Å")

    unconditioned = load_model(
        Path(args.direct_checkpoint),
        B_sinc,
        args.n_channels,
        use_features=False,
        mol_feature_dim=mol_feature_dim,
        device=device,
    )
    conditioned = load_model(
        Path(args.feature_checkpoint),
        B_sinc,
        args.n_channels,
        use_features=True,
        mol_feature_dim=mol_feature_dim,
        device=device,
    )
    nsk = NSKBaseline(
        R_max=args.nsk_r_max,
        dR=args.nsk_dr_basis,
        eps=args.nsk_eps,
        lambda_R=args.nsk_lambda_R,
        lambda_cont=args.nsk_lambda_cont,
    )

    snrs = parse_snrs(args)
    print(f"SNR values: {', '.join(f'{snr:g}' for snr in snrs)}")

    rows = []
    with torch.no_grad():
        for snr in snrs:
            loader = DataLoader(
                FixedMaskNoisySubset(
                    dataset,
                    val_idx,
                    q_min=q_min,
                    snr=snr,
                    noise_seed=args.noise_seed,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=4 if args.num_workers > 0 else None,
            )
            mse = {UNCONDITIONED: [], CONDITIONED: [], NSK: []}
            mae = {UNCONDITIONED: [], CONDITIONED: [], NSK: []}

            for batch in loader:
                x = batch["delta_S0_masked"].to(device)
                target = batch["delta_S0_norm"].to(device)
                mask = batch["mask"].to(device)
                features = batch["mol_features"].to(device)

                pred_uncond, _ = unconditioned(x, mask)
                pred_cond, _ = conditioned(x, mask, mol_features=features)

                x_np = batch["delta_S0_masked"].numpy()
                target_np = batch["delta_S0_norm"].numpy()
                mask_np = batch["mask"].numpy()
                uncond_np = pred_uncond.cpu().numpy()
                cond_np = pred_cond.cpu().numpy()
                nsk_np = nsk.predict(q_np, x_np, mask_np)

                predictions = {
                    UNCONDITIONED: uncond_np,
                    CONDITIONED: cond_np,
                    NSK: nsk_np,
                }
                for model, pred_np in predictions.items():
                    mse[model].append(masked_mse_np(pred_np, target_np, mask_np))
                    mae[model].append(masked_mae_np(pred_np, target_np, mask_np))

            snr_rows = {}
            for model in (UNCONDITIONED, CONDITIONED, NSK):
                row = summarize(snr, model, mse[model], mae[model])
                rows.append(row)
                snr_rows[model] = row

            print(
                f"SNR={snr:g} "
                f"{DISPLAY_LABELS[CONDITIONED]} mean MSE={snr_rows[CONDITIONED]['masked_mse_mean']:.6e} "
                f"{DISPLAY_LABELS[NSK]} mean MSE={snr_rows[NSK]['masked_mse_mean']:.6e}",
                flush=True,
            )

    csv_path = outdir / "snr_sensitivity_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in (CONDITIONED, UNCONDITIONED, NSK):
        xs = [row["snr"] for row in rows if row["model"] == DISPLAY_LABELS[model]]
        ys = [row["masked_mse_mean"] for row in rows if row["model"] == DISPLAY_LABELS[model]]
        ax.plot(xs, ys, marker="o", label=DISPLAY_LABELS[model])
    ax.set_xlabel("SNR (observed signal RMS / noise RMS)")
    ax.set_ylabel("mean masked MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    plot_path = outdir / "snr_sensitivity_mse.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
