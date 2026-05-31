"""
eval_methods.py

Compares four pair-density recovery methods on the validation set:
  0. Raw DBT         (pd_limited as-is, no recovery)
  1. L2 ridge        (l2_recovery.py)
  2. L1/TV ADMM      (decon_TVL1 from your pipeline)
  3. UNet model      (PairDensityUNet)

Usage
-----
python eval_methods.py \
    --val-split   checkpoints/split.npy \
    --traj-dir    data/trajectory_data \
    --db-path     train_4M/data0000.aselmdb \
    --ckpt        checkpoints/best_model.pt \
    --out-dir     eval_results \
    [--n-workers  4] \
    [--device     cuda]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.special import spherical_jn

# ── project imports ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.Dataset.dataset import MolScatteringDataset
from models.unet import PairDensityUNet
from processing.analyse_mol_v2 import decon_TVL1, DBT, lorch   # adjust import path
from models.baselines.l2_recovery import recover_l2, build_D_dict, auto_lam_l2

# ── metrics ────────────────────────────────────────────────────────────────

def mse(pred, true):
    return float(np.mean((pred - true) ** 2))

def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))

def r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def peak_position_error(pred, true, R_grid, prominence=0.01):
    """
    Mean absolute error between matched peak positions (Angstrom).
    Matches nearest peaks between pred and true.
    Returns NaN if either has no peaks.
    """
    pred_norm = pred / (np.max(np.abs(pred)) + 1e-12)
    true_norm = true / (np.max(np.abs(true)) + 1e-12)

    pred_peaks, _ = find_peaks(pred_norm, prominence=prominence)
    true_peaks, _ = find_peaks(true_norm, prominence=prominence)

    if len(pred_peaks) == 0 or len(true_peaks) == 0:
        return float("nan")

    # greedy nearest-neighbour matching
    pred_R = R_grid[pred_peaks]
    true_R = R_grid[true_peaks]
    errors = []
    for rp in pred_R:
        errors.append(np.min(np.abs(rp - true_R)))
    return float(np.mean(errors))

def compute_all_metrics(pred, true, R_grid):
    return {
        "mse":   mse(pred, true),
        "mae":   mae(pred, true),
        "r2":    r2(pred, true),
        "peak_err_A": peak_position_error(pred, true, R_grid),
    }

# ── classical recovery helpers ─────────────────────────────────────────────

def make_D_for_sample(npz_data: dict):
    """Build D_dict from the q / R_grid saved in one npz file."""
    q      = npz_data["q"].astype(np.float64)
    R_grid = npz_data["R_grid"].astype(np.float64)
    dq     = q[1] - q[0]
    dR     = R_grid[1] - R_grid[0]
    M      = int(np.ceil(np.pi / (dR * dq))) + 1
    det    = lorch(q)
    return build_D_dict(q, R_grid, det, M, DBT), R_grid

def recover_classical(pd_limited_1d, D, method="l1", lam=None):
    """Run one frame through L1 or L2 recovery. Returns 1-D recovered pd."""
    if method == "l2":
        if lam is None:
            lam = auto_lam_l2(pd_limited_1d, D)
        return recover_l2(pd_limited_1d, D, lam=lam)

    elif method == "l1":
        opts = dict(normalizeColumns=True, lambdaFrac=1e-2,
                    rho=1.0, maxIter=500, tolRel=1e-3, tolAbs=1e-6,
                    adaptiveRho=True, debias=True)
        recovered, info = decon_TVL1(pd_limited_1d, D, lam=lam, opts=opts)
        # use debiased result if available
        return info.get("dPD_TV_debiased", recovered).squeeze()

    else:
        raise ValueError(f"Unknown method: {method}")

# ── main eval loop ─────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load dataset ────────────────────────────────────────────────────────
    dataset = MolScatteringDataset(
        data_dir=args.traj_dir,
        db_path=args.db_path,
        max_atoms=50,
        max_modes=144,
    )

    split     = np.load(args.val_split)
    n_files   = len(dataset.files)
    n_val     = max(1, int(0.1 * n_files))
    val_files = set(split[:n_val].tolist())
    val_idx   = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]
    print(f"Val samples: {len(val_idx):,}")

    # ── load model ──────────────────────────────────────────────────────────
    model = PairDensityUNet(
        n_R=args.n_R, features=[32, 64, 128, 256],
        d_model=256, n_heads=8, max_atoms=50, max_modes=144,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

    # ── per-molecule results ─────────────────────────────────────────────────
    methods = ["raw_dbt", "l2", "l1", "unet"]
    results = {m: [] for m in methods}   # list of per-sample metric dicts
    timing  = {m: [] for m in methods}

    os.makedirs(args.out_dir, exist_ok=True)

    # cache D per file (expensive to rebuild per frame)
    D_cache = {}

    for sample_i, global_idx in enumerate(val_idx):
        fi, ti, fr = dataset._index[global_idx]
        sample     = dataset[global_idx]

        pd_limited = sample["pd_limited"].numpy()   # (nR,)
        pd_true    = sample["pd_true"].numpy()       # (nR,)
        R_grid     = sample["R_grid"].numpy()        # (nR,)
        mol_idx    = sample["mol_idx"]

        # ── build D once per file ───────────────────────────────────────────
        if fi not in D_cache:
            raw = np.load(dataset.files[fi])
            D_cache[fi], _ = make_D_for_sample(raw)
        D = D_cache[fi]

        # ── raw DBT ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        raw_pred = pd_limited
        timing["raw_dbt"].append(time.perf_counter() - t0)
        results["raw_dbt"].append({
            **compute_all_metrics(raw_pred, pd_true, R_grid),
            "mol_idx": mol_idx, "fi": fi,
        })

        # ── L2 ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        l2_pred = recover_classical(pd_limited, D, method="l2")
        timing["l2"].append(time.perf_counter() - t0)
        results["l2"].append({
            **compute_all_metrics(l2_pred, pd_true, R_grid),
            "mol_idx": mol_idx, "fi": fi,
        })

        # ── L1 ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        l1_pred = recover_classical(pd_limited, D, method="l1")
        timing["l1"].append(time.perf_counter() - t0)
        results["l1"].append({
            **compute_all_metrics(l1_pred, pd_true, R_grid),
            "mol_idx": mol_idx, "fi": fi,
        })

        # ── UNet ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        with torch.no_grad():
            unet_pred = model(
                sample["pd_limited"].unsqueeze(0).to(device),
                sample["V_vib"].unsqueeze(0).to(device),
                sample["omegas_ASE"].unsqueeze(0).to(device),
                sample["atom_mask"].unsqueeze(0).to(device),
                sample["mode_mask"].unsqueeze(0).to(device),
            ).squeeze(0).cpu().numpy()
        timing["unet"].append(time.perf_counter() - t0)
        results["unet"].append({
            **compute_all_metrics(unet_pred, pd_true, R_grid),
            "mol_idx": mol_idx, "fi": fi,
        })

        if sample_i % 100 == 0:
            print(f"  [{sample_i}/{len(val_idx)}] mol={mol_idx}")

    # ── aggregate ──────────────────────────────────────────────────────────
    print("\n── Aggregate metrics (mean ± std over val set) ──")
    agg = {}
    for m in methods:
        mets = results[m]
        agg[m] = {}
        for key in ["mse", "mae", "r2", "peak_err_A"]:
            vals = [r[key] for r in mets if not np.isnan(r[key])]
            agg[m][key] = {"mean": np.mean(vals), "std": np.std(vals)}
        agg[m]["time_ms"] = {"mean": np.mean(timing[m]) * 1e3,
                             "std":  np.std(timing[m])  * 1e3}
        print(f"\n{m}:")
        for k, v in agg[m].items():
            print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")

    # ── save ───────────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "per_sample_results.npy"), results)
    np.save(os.path.join(args.out_dir, "aggregate_metrics.npy"), agg)
    print(f"\nSaved results to {args.out_dir}/")
    return results, agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split",  required=True)
    parser.add_argument("--traj-dir",   required=True)
    parser.add_argument("--db-path",    required=True)
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--out-dir",    default="eval_results")
    parser.add_argument("--n-R",        type=int, default=1257)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    evaluate(args)