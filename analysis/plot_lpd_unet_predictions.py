"""
plot_lpd_unet_predictions.py — Plot LPD and UNet predicted pair density vs
ground truth for four molecules selected by atom count.

Usage
-----
    python analysis/plot_lpd_unet_predictions.py \
        --data-dir data/trajectory_data \
        --lpd-checkpoint checkpoints/lpd/best_model_lpd.pt \
        --unet-checkpoint checkpoints/unet/best_model_unet.pt \
        --plot lpd_unet_predictions.png
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch
from numpy.lib import format as npy_fmt

torch.backends.nnpack.enabled = False


# ── model loaders ─────────────────────────────────────────────────────────

def load_lpd(checkpoint_path, A_dbt, device,
             n_primal=5, n_dual=5, n_iter=10, n_channels=32, kernel_size=5):
    from models.learned_primal_dual import LearnedPrimalDual
    model = LearnedPrimalDual(
        A_dbt=A_dbt, n_primal=n_primal, n_dual=n_dual,
        n_iter=n_iter, n_channels=n_channels, kernel_size=kernel_size,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded LPD  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6e}")
    return model


def load_unet(checkpoint_path, device):
    from models.unet import PairDensityUNet
    model = PairDensityUNet(
        n_R=1257, features=[32, 64, 128, 256],
        d_model=256, n_heads=8, max_atoms=50, max_modes=144,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded UNet epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6e}")
    return model


# ── atom count scanning ────────────────────────────────────────────────────

def _read_v_vib_shape(npz_path):
    with zipfile.ZipFile(npz_path) as z:
        with z.open("V_vib.npy") as f:
            ver = npy_fmt.read_magic(f)
            reader = (npy_fmt.read_array_header_1_0 if ver == (1, 0)
                      else npy_fmt.read_array_header_2_0)
            shape, _, _ = reader(f)
    return shape


def scan_atom_counts(dataset, data_dir):
    counts = {}
    for loaded_idx, mol_id in enumerate(dataset._mol_idx):
        fpath = data_dir / f"mol_{mol_id:06d}_pd.npz"
        try:
            counts[loaded_idx] = _read_v_vib_shape(fpath)[0] // 3
        except Exception as e:
            print(f"  WARN: could not read V_vib from {fpath.name}: {e}")
    return counts


def pick_one_per_atom_target(counts, targets):
    picks = {}
    used  = set()
    items = list(counts.items())
    for t in targets:
        candidates = [(li, n) for li, n in items if li not in used]
        if not candidates:
            raise RuntimeError("Ran out of distinct molecules")
        best_li, best_n = min(candidates, key=lambda c: (abs(c[1] - t), c[0]))
        picks[t] = best_li
        used.add(best_li)
        print(f"  target={t:>3d} atoms -> loaded_idx={best_li}  "
              f"(actual atoms={best_n})")
    return picks


def pick_frame(dataset, loaded_idx, rng):
    mol_samples = [(ti, fr) for (li, ti, fr) in dataset._index
                   if li == loaded_idx]
    if not mol_samples:
        raise RuntimeError(f"No samples for loaded_idx={loaded_idx}")
    return mol_samples[rng.integers(len(mol_samples))]


# ── UNet input preparation ─────────────────────────────────────────────────

def prepare_unet_inputs(pd_limited, V_vib_raw, omegas_raw, device,
                        max_atoms=50, max_modes=144):
    three_N, n_modes = V_vib_raw.shape
    n_atoms = three_N // 3

    V_vib_padded  = np.zeros((3 * max_atoms, max_modes), dtype=np.float32)
    omegas_padded = np.zeros(max_modes,                  dtype=np.float32)
    V_vib_padded[:three_N, :n_modes] = V_vib_raw
    omegas_padded[:n_modes]          = omegas_raw

    atom_mask = torch.zeros(max_atoms, dtype=torch.bool)
    mode_mask = torch.zeros(max_modes, dtype=torch.bool)
    atom_mask[:n_atoms] = True
    mode_mask[:n_modes] = True

    return (
        torch.from_numpy(pd_limited).unsqueeze(0).to(device),
        torch.from_numpy(V_vib_padded).unsqueeze(0).to(device),
        torch.from_numpy(omegas_padded).unsqueeze(0).to(device),
        atom_mask.unsqueeze(0).to(device),
        mode_mask.unsqueeze(0).to(device),
    )


# ── main ──────────────────────────────────────────────────────────────────

def run(args):
    from models.learned_primal_dual import build_dbt_operator

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # scan all npz files for atom counts directly — no dataset needed
    files = sorted(data_dir.glob("mol_*_pd.npz"))
    print(f"Scanning {len(files)} files for atom counts...")

    atom_counts = {}  # fpath -> n_atoms
    for f in files:
        try:
            atom_counts[f] = _read_v_vib_shape(f)[0] // 3
        except Exception as e:
            print(f"  WARN {f.name}: {e}")

    # get q and R grids from first valid file
    ref_file = next(
        f for f in files
        if np.load(f, mmap_mode='r').keys().__contains__("q")
    )
    ref = np.load(ref_file)
    q_np = ref["q"].astype(np.float64)
    R_np = ref["R_grid"].astype(np.float64)

    M = len(R_np) + 1
    print(f"Building A_dbt (M={M}) ...")
    A_dbt, _ = build_dbt_operator(q_np, M)

    lpd_model  = load_lpd(Path(args.lpd_checkpoint),  A_dbt, device)
    unet_model = load_unet(Path(args.unet_checkpoint), device)

    targets = [int(t) for t in args.target_atoms.split(",")]
    print(f"Selecting molecules for targets {targets}:")

    # pick one file per target atom count
    picks = {}
    used  = set()
    items = list(atom_counts.items())
    for t in targets:
        candidates = [(f, n) for f, n in items if f not in used]
        if not candidates:
            raise RuntimeError("Ran out of distinct molecules")
        best_f, best_n = min(candidates, key=lambda c: (abs(c[1] - t), c[0]))
        picks[t] = (best_f, best_n)
        used.add(best_f)
        print(f"  target={t} -> {best_f.name} atoms={best_n}")

    rng    = np.random.default_rng(args.seed)
    panels = []

    for t, (fpath, n_atoms) in picks.items():
        d = np.load(fpath)
        n_traj, n_frames, _ = d["pd_limited"].shape
        ti = int(rng.integers(n_traj))
        fr = int(rng.integers(n_frames))

        pd_true    = d["pd_true"][ti, fr].astype(np.float32)
        pd_limited = d["pd_limited"][ti, fr].astype(np.float32)
        dS0        = torch.from_numpy(d["dS0"][ti, fr].astype(np.float32))
        V_vib_raw  = d["V_vib"].astype(np.float32)
        omegas_raw = d["omegas_ASE"].astype(np.float32)
        mol_id     = int(d["mol_idx"])

        alpha = np.mean(pd_true) / (np.mean(np.abs(pd_limited)) + 1e-12)

        with torch.no_grad():
            lpd_pred = lpd_model(
                dS0.unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

        pd_lim_t, V_t, om_t, am_t, mm_t = prepare_unet_inputs(
            pd_limited, V_vib_raw, omegas_raw, device
        )
        with torch.no_grad():
            unet_log  = unet_model(pd_lim_t, V_t, om_t, am_t, mm_t)
            unet_pred = np.expm1(unet_log.squeeze(0).cpu().numpy())

        print(f"  mol {mol_id} | atoms={n_atoms} | alpha={alpha:.1f} | "
              f"pd_true max={pd_true.max():.1f} | "
              f"lpd max={lpd_pred.max():.1f} | "
              f"unet max={unet_pred.max():.1f}")

        panels.append({
            "target_atoms": t,
            "actual_atoms": n_atoms,
            "mol_id":       mol_id,
            "traj":         ti,
            "frame":        fr,
            "pd_true":      pd_true,
            "pd_limited":   pd_limited * alpha,
            "lpd_pred":     lpd_pred,
            "unet_pred":    unet_pred,
        })

    _plot(panels, R_np, args.plot, args.r_max)


# ── plotting ──────────────────────────────────────────────────────────────

def _plot(panels, R, outpath, r_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    ncols    = 2
    nrows    = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12, 3.5 * nrows),
                             squeeze=False,
                             constrained_layout=True)

    for ax, p in zip(axes.flat, panels):
        R_mask = R <= r_max
        R_plot = R[R_mask]

        ax.plot(R_plot, p["pd_true"][R_mask],
                color="black",   lw=1.6, ls="--", label="Ground truth",     zorder=5)
        ax.plot(R_plot, p["pd_limited"][R_mask],
                color="#aaaaaa", lw=1.0,           label="Raw DBT (scaled)", zorder=2)
        ax.plot(R_plot, p["lpd_pred"][R_mask],
                color="#CC3311", lw=1.2, alpha=0.9, label="LPD",            zorder=3)
        ax.plot(R_plot, p["unet_pred"][R_mask],
                color="#009988", lw=1.2, alpha=0.9, label="UNet (ours)",    zorder=4)

        ax.set_title(
            f"{p['actual_atoms']} atoms  |  mol {p['mol_id']}  "
            f"traj {p['traj']}  frame {p['frame']}",
            fontsize=9,
        )
        ax.set_xlabel("R (A)")
        ax.set_ylabel("Pair density")
        ax.set_xlim(0, r_max)
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",        default="data/trajectory_data")
    p.add_argument("--lpd-checkpoint",  required=True)
    p.add_argument("--unet-checkpoint", required=True)
    p.add_argument("--target-atoms",    default="5,10,25,45")
    p.add_argument("--r-max",           type=float, default=6.0)
    p.add_argument("--plot",            default="lpd_unet_predictions.png")
    p.add_argument("--seed",            type=int,   default=0)
    args = p.parse_args()
    run(args)