#!/usr/bin/env python3
"""Plot PD and scattering diagnostics for one random molecule.

The second bottom-row panel shows the normalized delta S0 signal used by the
beamstop model before low-q masking augmentation. The two right-most panels show
frame-relative differences of ``pd_limited``. The raw ``pd_limited`` array is
the same resolution-limited PD-like input returned by ``models/unet/dataset/dataset.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("pd_limited", "pd_true", "S0", "fe", "q", "R_grid", "time_fs", "mol_idx")


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"{path} is missing required keys: {', '.join(missing)}")
        return {key: np.asarray(data[key]) for key in REQUIRED_KEYS}


def choose_file(data_dir: Path, rng: np.random.Generator, mol_idx: int | None) -> tuple[Path, dict[str, np.ndarray]]:
    if mol_idx is not None:
        path = data_dir / f"mol_{mol_idx:06d}_pd.npz"
        return path, load_npz(path)

    files = sorted(data_dir.glob("mol_*_pd.npz"))
    if not files:
        raise FileNotFoundError(f"No mol_*_pd.npz files found in {data_dir}")

    order = rng.permutation(len(files))
    errors: list[str] = []
    for i in order:
        path = files[int(i)]
        try:
            return path, load_npz(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    preview = "\n".join(errors[:5])
    raise RuntimeError(f"Could not load any molecule file from {data_dir}.\n{preview}")


def choose_peak_frame(pd_true: np.ndarray, r_mask: np.ndarray, frame_idx: int | None) -> int:
    if frame_idx is not None:
        if frame_idx < 0 or frame_idx >= pd_true.shape[0]:
            raise IndexError(f"--frame-idx {frame_idx} is outside [0, {pd_true.shape[0] - 1}]")
        return frame_idx

    delta_pd = pd_true - pd_true[0:1]
    score = np.nanmax(np.abs(delta_pd[:, r_mask]), axis=1)
    if score.size > 1:
        score[0] = -np.inf
    return int(np.nanargmax(score))


def symmetric_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    lim = float(np.percentile(np.abs(finite), percentile))
    return max(lim, 1e-12)


def plot(args) -> Path:
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path, arrays = choose_file(data_dir, rng, args.mol_idx)
    pd_limited_all = arrays["pd_limited"].astype(np.float64)
    pd_true_all = arrays["pd_true"].astype(np.float64)
    S0_all = arrays["S0"].astype(np.float64)
    fe = arrays["fe"].astype(np.float64)
    q = arrays["q"].astype(np.float64)
    R = arrays["R_grid"].astype(np.float64)
    time_fs = arrays["time_fs"].astype(np.float64)
    mol_idx = int(arrays["mol_idx"])

    n_traj, n_frames, _ = pd_true_all.shape
    if args.traj_idx is None:
        traj_idx = int(rng.integers(n_traj))
    else:
        traj_idx = args.traj_idx
        if traj_idx < 0 or traj_idx >= n_traj:
            raise IndexError(f"--traj-idx {traj_idx} is outside [0, {n_traj - 1}]")

    r_mask = (R >= args.r_min) & (R <= args.r_max)
    if not np.any(r_mask):
        raise ValueError(f"No R_grid points inside [{args.r_min}, {args.r_max}]")

    pd_true = pd_true_all[traj_idx]
    pd_limited = pd_limited_all[traj_idx]
    S0 = S0_all[traj_idx]
    peak_frame = choose_peak_frame(pd_true, r_mask, args.frame_idx)

    static_pd = pd_true[0]
    delta_pd = pd_true - static_pd[None, :]
    delta_pd_peak = delta_pd[peak_frame]
    delta_pd_limited = pd_limited - pd_limited[0:1]
    delta_pd_limited_peak = delta_pd_limited[peak_frame]
    delta_S0_norm = (S0[peak_frame] - S0[0]) / (fe + 1e-12)

    R_plot = R[r_mask]
    time_plot = time_fs[:n_frames]
    peak_time = float(time_plot[peak_frame])

    plt = load_pyplot()
    fig = plt.figure(figsize=(20, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax_static = fig.add_subplot(grid[0, 0])
    ax_delta_peak = fig.add_subplot(grid[0, 1])
    ax_delta_heat = fig.add_subplot(grid[0, 2])
    ax_empty = fig.add_subplot(grid[0, 3])
    ax_s0 = fig.add_subplot(grid[1, 0])
    ax_beamstop_input = fig.add_subplot(grid[1, 1])
    ax_pd_limited_peak = fig.add_subplot(grid[1, 2])
    ax_pd_limited_heat = fig.add_subplot(grid[1, 3])
    fig.suptitle(
        f"mol_{mol_idx:06d}  |  trajectory {traj_idx}  |  peak frame {peak_frame} "
        f"(t = {peak_time:.1f} fs)",
        fontsize=13,
    )
    ax_empty.axis("off")

    ax = ax_static
    ax.plot(R_plot, static_pd[r_mask], color="black", lw=1.0)
    ax.set_title(r"Static $\mathrm{PD}(R)$")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("weighted pair density")
    ax.grid(alpha=0.2)

    ax = ax_delta_peak
    ax.plot(R_plot, delta_pd_peak[r_mask], color="C0", lw=1.2)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_title(r"$\Delta \mathrm{PD}$ (peak frame)")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel(r"$\Delta \mathrm{PD}$")
    ax.grid(alpha=0.2)

    ax = ax_delta_heat
    top_vlim = symmetric_limit(delta_pd[:, r_mask])
    im = ax.imshow(
        delta_pd[:, r_mask].T,
        origin="lower",
        aspect="auto",
        extent=[time_plot[0], time_plot[-1], R_plot[0], R_plot[-1]],
        cmap="RdBu_r",
        vmin=-top_vlim,
        vmax=top_vlim,
    )
    ax.axvline(peak_time, color="0.25", lw=0.8, ls="--")
    ax.set_title(r"$\Delta \mathrm{PD}(R,t)$")
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("R (Angstrom)")
    fig.colorbar(im, ax=ax, label=r"$\Delta \mathrm{PD}$")

    ax = ax_s0
    ax.plot(q, S0[0], color="C3", lw=1.1, label="ground (t=0)")
    ax.plot(q, S0[peak_frame], color="C0", lw=1.1, ls="--", label=f"peak (t={peak_time:.1f} fs)")
    ax.set_title(r"Isotropic scattering $S_0(q)$")
    ax.set_xlabel(r"q ($\AA^{-1}$)")
    ax.set_ylabel(r"$S_0(q)$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = ax_beamstop_input
    ax.plot(q, delta_S0_norm, color="C4", lw=1.3)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_title(r"$\Delta S_{0,\mathrm{norm}}(q)$")
    ax.set_xlabel(r"q ($\AA^{-1}$)")
    ax.set_ylabel(r"$\Delta S_{0,\mathrm{norm}}$")
    ax.grid(alpha=0.2)

    ax = ax_pd_limited_peak
    ax.plot(R_plot, delta_pd_limited_peak[r_mask], color="purple", lw=1.3)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_title(r"Resolution Limited $\Delta \mathrm{PD}(R)$ (peak frame)")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel(r"Resolution Limited $\Delta \mathrm{PD}$")
    ax.grid(alpha=0.2)

    ax = ax_pd_limited_heat
    bottom_vlim = symmetric_limit(delta_pd_limited[:, r_mask])
    im = ax.imshow(
        delta_pd_limited[:, r_mask].T,
        origin="lower",
        aspect="auto",
        extent=[time_plot[0], time_plot[-1], R_plot[0], R_plot[-1]],
        cmap="RdBu_r",
        vmin=-bottom_vlim,
        vmax=bottom_vlim,
    )
    ax.axvline(peak_time, color="0.25", lw=0.8, ls="--")
    ax.set_title(r"Resolution Limited $\Delta \mathrm{PD}(R,t)$")
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("R (Angstrom)")
    fig.colorbar(im, ax=ax, label=r"Resolution Limited $\Delta \mathrm{PD}$")

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Source file: {npz_path}")
    print(
        f"mol_idx={mol_idx} traj_idx={traj_idx} peak_frame={peak_frame} "
        f"peak_time_fs={peak_time:.3f}"
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/trajectory_data_cap200_new",
                        help="Directory containing mol_*_pd.npz files.")
    parser.add_argument("--output", default="analysis/beamstop/final_results/random_molecule_pd_limited.png",
                        help="Output image path.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for molecule/trajectory selection.")
    parser.add_argument("--mol-idx", type=int, default=None,
                        help="Specific molecule index to plot, e.g. 80 for mol_000080_pd.npz.")
    parser.add_argument("--traj-idx", type=int, default=None,
                        help="Specific trajectory index within the molecule.")
    parser.add_argument("--frame-idx", type=int, default=None,
                        help="Specific frame index. By default, choose largest Delta PD frame.")
    parser.add_argument("--r-min", type=float, default=0.0,
                        help="Minimum R value to plot.")
    parser.add_argument("--r-max", type=float, default=5.5,
                        help="Maximum R value to plot.")
    parser.add_argument("--dpi", type=int, default=180,
                        help="Saved figure DPI.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot(args)


if __name__ == "__main__":
    main()
