"""
plot_lpd_predictions.py — Plot LPD predicted pair density vs ground truth
for four molecules selected by atom count.

For each target atom count in --target-atoms (default 5, 10, 25, 45), finds
the loaded molecule whose atom count is closest, picks a random frame from
it, runs the trained LearnedPrimalDual model, and saves a 2×2 grid of
panels showing pd_true(R) overlaid with pd_pred(R) for R ∈ [0, 10] Å.

Atom count is read from the V_vib.npy header inside each mol_*_pd.npz
without decompressing the data — fast even for 700+ files.

Usage
-----
    python analysis/plot_lpd_predictions.py \
        --checkpoint checkpoints/lpd/best_model.pt \
        --plot lpd_predictions.png
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

# Silence c10 NNPACK noise on CPU nodes — set before `import torch`.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch
from numpy.lib import format as npy_fmt

torch.backends.nnpack.enabled = False


def load_model(checkpoint_path: Path, A_dbt: torch.Tensor, device: torch.device,
               n_primal=5, n_dual=5, n_iter=10, n_channels=32, kernel_size=5):
    from models.learned_primal_dual import LearnedPrimalDual
    model = LearnedPrimalDual(
        A_dbt=A_dbt, n_primal=n_primal, n_dual=n_dual,
        n_iter=n_iter, n_channels=n_channels, kernel_size=kernel_size,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6e}")
    return model


def _read_v_vib_shape(npz_path: Path) -> tuple[int, ...]:
    """Read V_vib.npy header from the npz without decompressing the array."""
    with zipfile.ZipFile(npz_path) as z:
        with z.open("V_vib.npy") as f:
            ver = npy_fmt.read_magic(f)
            reader = (npy_fmt.read_array_header_1_0 if ver == (1, 0)
                      else npy_fmt.read_array_header_2_0)
            shape, _, _ = reader(f)
    return shape


def scan_atom_counts(dataset, data_dir: Path) -> dict[int, int]:
    """Map loaded_idx → atom count for every successfully-loaded molecule."""
    counts: dict[int, int] = {}
    for loaded_idx, mol_id in enumerate(dataset._mol_idx):
        fpath = data_dir / f"mol_{mol_id:06d}_pd.npz"
        try:
            counts[loaded_idx] = _read_v_vib_shape(fpath)[0] // 3
        except Exception as e:
            print(f"  WARN: could not read V_vib from {fpath.name}: {e}")
    return counts


def pick_one_per_atom_target(counts: dict[int, int], targets: list[int]) -> dict[int, int]:
    """For each target atom count, return {target → loaded_idx of closest match}.

    No duplicates: once a loaded_idx is picked, it can't be picked again, so
    each panel shows a distinct molecule even if two targets land on the same
    closest match.
    """
    picks: dict[int, int] = {}
    used: set[int] = set()
    items = list(counts.items())
    for t in targets:
        candidates = [(li, n) for li, n in items if li not in used]
        if not candidates:
            raise RuntimeError("Ran out of distinct molecules to assign")
        best_li, best_n = min(candidates, key=lambda c: (abs(c[1] - t), c[0]))
        picks[t] = best_li
        used.add(best_li)
        print(f"  target={t:>3d} atoms → loaded_idx={best_li}  (actual atoms={best_n})")
    return picks


def pick_frame(dataset, loaded_idx: int, rng) -> tuple[dict, tuple[int, int, int]]:
    """Pick one random (sample, idx_tuple) for the given molecule."""
    mol_samples = [(i, t) for i, t in enumerate(dataset._index) if t[0] == loaded_idx]
    if not mol_samples:
        raise RuntimeError(f"No samples for loaded_idx={loaded_idx}")
    i, idx_tuple = mol_samples[rng.integers(len(mol_samples))]
    return dataset[i], idx_tuple


def run(args):
    from data.Dataset.dataset_lpd import LPDDataset
    from models.learned_primal_dual import build_dbt_operator

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    dataset = LPDDataset(data_dir)
    q_np    = dataset.q.numpy()
    R_np    = dataset.R_grid.numpy()
    M       = len(R_np) + 1
    print(f"Building A_dbt (M={M}) ...")
    A_dbt, _ = build_dbt_operator(q_np, M)

    model = load_model(Path(args.checkpoint), A_dbt, device)

    print("Scanning V_vib headers to find atom counts ...")
    atom_counts = scan_atom_counts(dataset, data_dir)
    print(f"  got atom counts for {len(atom_counts)}/{len(dataset._mol_idx)} molecules")

    targets = [int(t) for t in args.target_atoms.split(",")]
    print(f"Selecting molecules for atom-count targets {targets}:")
    picks = pick_one_per_atom_target(atom_counts, targets)

    rng     = np.random.default_rng(args.seed)
    panels  = []
    dS0_list = []
    for t in targets:
        li = picks[t]
        sample, idx_tuple = pick_frame(dataset, li, rng)
        dS0_list.append(sample["dS0"])
        panels.append({
            "target_atoms": t,
            "actual_atoms": atom_counts[li],
            "mol_id":       dataset._mol_idx[li],
            "traj":         idx_tuple[1],
            "frame":        idx_tuple[2],
            "pd_true":      sample["pd_true"].numpy(),
        })

    dS0_batch = torch.stack(dS0_list).to(device)
    with torch.no_grad():
        pred_batch = model(dS0_batch).cpu().numpy()
    for p, pred in zip(panels, pred_batch):
        p["pd_pred"] = pred

    _plot(panels, R_np, args.plot, args.r_max)
    print(f"Wrote {args.plot}")


def _plot(panels, R, outpath, r_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    ncols    = 2
    nrows    = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows),
                              squeeze=False)

    for ax, p in zip(axes.flat, panels):
        ax.plot(R, p["pd_true"], "k-",  lw=1.0, label="ground truth")
        ax.plot(R, p["pd_pred"], "C3-", lw=1.0, alpha=0.85, label="LPD prediction")
        ax.set_title(
            f"target={p['target_atoms']} atoms (actual={p['actual_atoms']}) "
            f"| mol={p['mol_id']}  traj={p['traj']}  frame={p['frame']}",
            fontsize=9,
        )
        ax.set_xlabel("R (Å)")
        ax.set_ylabel("pair density")
        ax.set_xlim(0, r_max)
        ax.grid(alpha=0.3)

    # Hide any unused subplot cells.
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",      default="data/trajectory_data_cap200")
    p.add_argument("--checkpoint",    default="checkpoints/lpd/best_model.pt")
    p.add_argument("--target-atoms",  default="5,10,25,45",
                   help="Comma-separated atom counts to find molecules for")
    p.add_argument("--r-max",         type=float, default=10.0,
                   help="Upper x-axis limit (Å)")
    p.add_argument("--plot",          default="lpd_predictions.png")
    p.add_argument("--seed",          type=int, default=0)
    args = p.parse_args()
    run(args)
