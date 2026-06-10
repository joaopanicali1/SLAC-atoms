"""
plot_recovery.py

Two plot types:
  1. per_molecule_recovery  — for a handful of molecules, plot pd_limited,
                              L2, L1, UNet, and ground truth on one figure.
  2. method_comparison      — scatter/box plots of MSE per molecule across
                              methods, and MSE vs n_atoms.

Usage
-----
python analysis/pd_recovery/plot_baseline_comps.py \
    --results-dir  analysis/pd_recovery/results \
    --traj-dir     data/trajectory_data_cap200_new \
    --out-dir      analysis/pd_recovery/results/figures \
    [--n-examples  6]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

METHODS      = ["raw_dbt", "l2", "l1", "unet"]
METHOD_LABELS = {
    "raw_dbt": "Raw DBT",
    "l2":      "L2 Ridge",
    "l1":      "L1/TV ADMM",
    "unet":    "UNet (ours)",
}
METHOD_COLORS = {
    "raw_dbt": "#aaaaaa",
    "l2":      "#4477AA",
    "l1":      "#EE6677",
    "unet":    "#228833",
}

# ── helpers ────────────────────────────────────────────────────────────────

def load_results(results_dir):
    per_sample = np.load(
        os.path.join(results_dir, "per_sample_results.npy"), allow_pickle=True
    ).item()
    agg = np.load(
        os.path.join(results_dir, "aggregate_metrics.npy"), allow_pickle=True
    ).item()
    return per_sample, agg


def load_raw_predictions(traj_dir, fi, ti, fr, dataset=None):
    """
    Reload raw arrays for one (fi, ti, fr) sample so we can plot curves.
    Returns pd_limited, pd_true, R_grid as numpy arrays.
    """
    files = sorted(Path(traj_dir).glob("mol_*_pd.npz"))
    data = np.load(files[fi])
    return (
        data["pd_limited"][ti, fr],
        data["pd_true"][ti, fr],
        data["R_grid"],
    )


# ── plot 1: per-molecule recovery curves ──────────────────────────────────

def plot_per_molecule(
    results_dir,
    traj_dir,
    out_dir,
    n_examples=6,
    recovered_dir=None,   # optional: dir with saved per-sample predictions
):
    """
    For n_examples molecules, plot one row per molecule showing:
      col 0: pd_limited (noisy input)
      col 1: L2 recovered
      col 2: L1 recovered
      col 3: UNet recovered
      col 4: ground truth
    """
    per_sample, _ = load_results(results_dir)

    # pick diverse examples: sort val samples by UNet MSE, take spread
    unet_mses = [(i, r["mse"]) for i, r in enumerate(per_sample["unet"])]
    unet_mses.sort(key=lambda x: x[1])
    n = len(unet_mses)
    chosen_indices = [unet_mses[int(n * f)][0]
                      for f in np.linspace(0.05, 0.95, n_examples)]

    fig, axes = plt.subplots(
        n_examples, 5,
        figsize=(18, 3.0 * n_examples),
        constrained_layout=True,
    )
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input (Raw DBT)", "L2 Ridge", "L1/TV ADMM", "UNet (ours)", "Ground Truth"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    dataset_files = sorted(Path(traj_dir).glob("mol_*_pd.npz"))

    for row, sample_i in enumerate(chosen_indices):
        ref = per_sample["unet"][sample_i]
        fi  = ref["fi"]
        mol_idx = ref["mol_idx"]

        # we need (ti, fr) — stored indirectly via dataset._index
        # reload file to get shapes and pick frame 0 traj 0
        data = np.load(dataset_files[fi])
        # find a representative (ti, fr): pick the one matching sample_i
        # by scanning for first match of fi in per_sample indices
        # (safe fallback: just use [0,0])
        ti, fr = 0, 0
        pd_lim  = data["pd_limited"][ti, fr]
        pd_true = data["pd_true"][ti, fr]
        R_grid  = data["R_grid"]

        # load predictions if saved separately, otherwise re-run
        # here we just re-use the metric dicts to annotate — 
        # full prediction arrays should be saved during eval if needed.
        # For now we plot what we have (pd_limited and pd_true),
        # and mark the MSE from each method.
        curves = {
            "raw_dbt": pd_lim,
            "l2":      None,   # fill in if you save predictions during eval
            "l1":      None,
            "unet":    None,
        }

        plot_cols = ["raw_dbt", "l2", "l1", "unet"]
        for col, method in enumerate(plot_cols):
            ax = axes[row, col]
            curve = curves[method]
            m_mse = per_sample[method][sample_i]["mse"]

            if curve is not None:
                ax.plot(R_grid, curve, color=METHOD_COLORS[method], lw=1.2)
            else:
                ax.text(0.5, 0.5, f"MSE={m_mse:.4f}\n(load pred to plot)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray")

            ax.set_xlim(R_grid[0], min(R_grid[-1], 6.0))
            ax.set_xlabel("R (Å)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"mol {mol_idx}", fontsize=8)
            ax.tick_params(labelsize=7)

        # ground truth
        ax = axes[row, 4]
        ax.plot(R_grid, pd_true, color="black", lw=1.2)
        ax.set_xlim(R_grid[0], min(R_grid[-1], 6.0))
        ax.set_xlabel("R (Å)", fontsize=8)
        ax.tick_params(labelsize=7)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_molecule_recovery.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_molecule_with_preds(
    R_grids,        # list of (nR,) arrays
    pd_limiteds,    # list of (nR,)
    pd_trues,       # list of (nR,)
    l2_preds,       # list of (nR,)
    l1_preds,       # list of (nR,)
    unet_preds,     # list of (nR,)
    mol_labels,     # list of str
    out_path,
    xlim=(0.5, 6.0),
):
    """
    Call this when you have all prediction arrays in memory.
    Produces one row per molecule.
    """
    n = len(R_grids)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.0 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input (Raw DBT)", "L2 Ridge", "L1/TV ADMM", "UNet (ours)", "Ground Truth"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    pred_map = [pd_limiteds, l2_preds, l1_preds, unet_preds]
    colors   = [METHOD_COLORS[m] for m in ["raw_dbt", "l2", "l1", "unet"]]

    for row in range(n):
        R = R_grids[row]
        true = pd_trues[row]

        for col, (preds, color) in enumerate(zip(pred_map, colors)):
            ax = axes[row, col]
            ax.plot(R, preds[row], color=color, lw=1.2)
            ax.plot(R, true, color="black", lw=0.8, ls="--", alpha=0.5, label="truth")
            mse_val = float(np.mean((preds[row] - true) ** 2))
            ax.set_title(f"MSE={mse_val:.4f}", fontsize=7)
            ax.set_xlim(*xlim)
            ax.set_xlabel("R (Å)", fontsize=8)
            if col == 0:
                ax.set_ylabel(mol_labels[row], fontsize=8)
            ax.tick_params(labelsize=7)

        # ground truth panel
        ax = axes[row, 4]
        ax.plot(R, true, color="black", lw=1.2)
        ax.set_xlim(*xlim)
        ax.set_xlabel("R (Å)", fontsize=8)
        ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot 2: aggregate method comparison ───────────────────────────────────

def plot_method_comparison(results_dir, out_dir):
    per_sample, agg = load_results(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── 2a: box plots of MSE per method ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, metric in zip(axes, ["mse", "mae", "r2"]):
        data  = [np.array([r[metric] for r in per_sample[m]]) for m in METHODS]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} distribution (val set)")
        if metric == "mse":
            ax.set_yscale("log")

    path = os.path.join(out_dir, "boxplots_metrics.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 2b: scatter — UNet MSE vs L1 MSE (one dot per sample) ─────────────
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    l1_mses   = np.array([r["mse"] for r in per_sample["l1"]])
    unet_mses = np.array([r["mse"] for r in per_sample["unet"]])

    ax.scatter(l1_mses, unet_mses, alpha=0.3, s=8, color=METHOD_COLORS["unet"])
    lims = [
        min(l1_mses.min(), unet_mses.min()) * 0.8,
        max(l1_mses.max(), unet_mses.max()) * 1.2,
    ]
    ax.plot(lims, lims, "k--", lw=1, label="equal")
    ax.set_xlabel("L1/TV ADMM MSE")
    ax.set_ylabel("UNet MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("UNet vs L1 — per sample MSE\n(below diagonal = UNet wins)")

    path = os.path.join(out_dir, "scatter_unet_vs_l1.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 2c: bar chart of mean MSE with error bars ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    means = [agg[m]["mse"]["mean"] for m in METHODS]
    stds  = [agg[m]["mse"]["std"]  for m in METHODS]
    x     = np.arange(len(METHODS))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[METHOD_COLORS[m] for m in METHODS], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Mean MSE")
    ax.set_title("Mean MSE on validation set")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

    path = os.path.join(out_dir, "bar_mean_mse.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 2d: summary table printed to stdout ───────────────────────────────
    print("\n── Summary table ──")
    header = f"{'Method':<15} {'MSE':>10} {'MAE':>10} {'R2':>8} {'PeakErr(Å)':>12} {'Time(ms)':>10}"
    print(header)
    print("-" * len(header))
    for m in METHODS:
        a = agg[m]
        print(
            f"{METHOD_LABELS[m]:<15} "
            f"{a['mse']['mean']:>10.5f} "
            f"{a['mae']['mean']:>10.5f} "
            f"{a['r2']['mean']:>8.4f} "
            f"{a['peak_err_A']['mean']:>12.4f} "
            f"{a['time_ms']['mean']:>10.2f}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="analysis/pd_recovery/results")
    parser.add_argument("--traj-dir",    required=True)
    parser.add_argument("--out-dir",     default="analysis/pd_recovery/results/figures")
    parser.add_argument("--n-examples",  type=int, default=6)
    args = parser.parse_args()

    plot_per_molecule(
        results_dir=args.results_dir,
        traj_dir=args.traj_dir,
        out_dir=args.out_dir,
        n_examples=args.n_examples,
    )
    plot_method_comparison(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
    )
