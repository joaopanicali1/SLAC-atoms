"""
plot_lpd_predictions.py — Plot predicted pair densities vs ground truth for four molecules
selected by atom count.

For each target atom count in --target-atoms (default 5, 10, 25, 45), finds the molecule of the
chosen pool (default: the validation molecules of data/splits/split_lpd.npy) whose atom count is closest,
picks a random frame from it, runs the selected models, and saves a 2×2 grid of panels showing
the ground truth overlaid with each prediction for R ∈ [0, --r-max] Å.

Units: all curves are NORMALISED pair densities p(R) = pd(R) / Σ_{a<b} Z_a Z_b, i.e. the fraction
of the total pair weight per 0.025 Å bin — the quantity the scattering determines (dS0 is
divided by fe ∝ ΣZZ) and the one analysis/pd_recovery/eval_pd_recovery.py scores. LPD v1 predicts absolute
pd with an unknown per-molecule scale, so its output is divided by its own sum, as in the harness.
Each legend entry gives that frame's W1 [Å] (earth-mover distance up to 1 Å past the longest true
pair distance — same definition as the harness).

Methods (--models, comma-separated; default v2,nsk_paper,dbt)
    v2      LearnedPrimalDualV2 (models/lpd/code/learned_primal_dual_v2.py), checkpoint --checkpoint-v2.
            It receives the processed signal and the analysis window of the condition.
    nsk_paper  NSK-ℓ1 as published (Natan, PRA 107, 023105, 2023; models/nsk/code/nsk.py): DBT on a grid of
            spacing --nsk-dr (Fig. 5 of the paper: ≈ 0.1 Å), dictionary of natural scattering
            kernels on the same grid, exact ℓ1 solution with ε at the L-curve corner, chosen per
            frame. Non-negative weights for p, signed for Δp. Drawn as stems at the dictionary
            positions (the ℓ1 weights, p = w/2), as in the paper's Figs. 2 and 5; with --blur the
            stems are blurred like every other curve (the paper's solid red line in Fig. 5 is the
            stems blurred by its 0.15 Å minimum-separation distance, i.e. --blur 0.064). Metrics use
            the stems moved onto the 0.025 Å grid (weight split between the two nearest bins).
    dbt     Discrete Bessel Transform of dS0 times the global scale k.
    v1      LearnedPrimalDual (models/lpd/code/learned_primal_dual.py), checkpoint --checkpoint. Trained on
            the full q-range only, so with --qmin > 0 it is out of distribution. Off by default.
nsk_paper has no tuned parameter (ε is chosen per frame by the L-curve, as in the paper).
The DBT scale k is read from the harness results of the same condition (--tuned; by default analysis/pd_recovery/results/lpd_v2/pd_eval_clean.json for
q_min = 0 and pd_eval_bs_q08.json for q_min = 0.8), so the curves are exactly the estimators the
harness tables score. (The pipeline's ADMM NSK, nsk_l1, was removed together with its harness
entry; the published NSK above replaces it.)

Measurement condition
    --qmin  beamstop edge [Å⁻¹]: S0 is processed as (S0 − Σf²)/fe · W with the Lorch window W
            over q ≥ q_min only (q_min = 0 reproduces the stored dS0).
    --delta time-difference task, as in eval_pd_recovery.py --delta: target Δp = p(t) − p(0) of the
            same trajectory (frame 0 is the reference; the frame t ≥ 1 is drawn as without
            --delta, so the panels show the same molecules and frames as the absolute plots).
            LPD v2 / v1: difference of the two predictions, v(t) − v(0) (neither was trained on Δ).
            DBT: k · DBT(dS0(t) − dS0(0)). NSK (paper): signed ℓ1 weights (no positivity) for the
            difference signal. DBT scale from pd_eval_delta.json by default.
            The legend also gives R² at σ = 0.1 Å (prediction and truth both blurred), the
            harness's tuning criterion for this task.
    --blur  display only: Gaussian σ [Å] applied to every curve (truth and predictions) before
            plotting; metrics are always computed on the unblurred 0.025 Å histograms. Useful for
            Δp, whose raw histogram is dominated by ± spike pairs from distances that moved by one
            bin.

Atom counts are read from the V_vib.npy header inside each mol_*_pd.npz without decompressing
the data, and only the four selected files are loaded — fast enough for a login node.

Usage
-----
    python analysis/pd_recovery/plot_lpd_predictions.py                         # clean, validation molecules
    python analysis/pd_recovery/plot_lpd_predictions.py --qmin 0.8              # beamstop at 0.8 Å⁻¹
    python analysis/pd_recovery/plot_lpd_predictions.py --pool all --models v1  # old behaviour (v1, any molecule)
    python analysis/pd_recovery/plot_lpd_predictions.py --delta                 # Δp = p(t) − p(0)
    python analysis/pd_recovery/plot_lpd_predictions.py --delta --blur 0.1      # same, displayed at σ = 0.1 Å
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Silence c10 NNPACK noise on CPU nodes — set before `import torch`.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from numpy.lib import format as npy_fmt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

torch.backends.nnpack.enabled = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STYLE = {"v2": dict(color="C3", lw=1.2, label="LPD v2", zorder=4),
         "nsk_paper": dict(color="C4", lw=1.2, label="NSK-ℓ1 (paper)", zorder=5),
         "dbt": dict(color="C0", lw=1.0, ls="--", alpha=0.8, label="DBT", zorder=2),
         "v1": dict(color="C4", lw=0.9, alpha=0.7, label="LPD v1", zorder=2)}


# ── molecule selection ─────────────────────────────────────────────────────

def pool_files(data_dir: Path, split_path: Path, pool: str) -> list[Path]:
    """Files of the requested pool. split_path holds a permutation of the sorted file list whose
    first 10 % are the validation molecules (models/lpd/training/train_lpd.py convention)."""
    files = sorted(data_dir.glob("mol_*_pd.npz"))
    if pool == "all":
        return files
    perm = np.load(split_path)
    n_val = max(1, int(0.1 * len(files)))
    idx = perm[:n_val] if pool == "val" else perm[n_val:]
    return [files[i] for i in idx]


def _read_v_vib_shape(npz_path: Path) -> tuple[int, ...]:
    """Read V_vib.npy header from the npz without decompressing the array."""
    with zipfile.ZipFile(npz_path) as z:
        with z.open("V_vib.npy") as f:
            ver = npy_fmt.read_magic(f)
            reader = (npy_fmt.read_array_header_1_0 if ver == (1, 0)
                      else npy_fmt.read_array_header_2_0)
            shape, _, _ = reader(f)
    return shape


def scan_atom_counts(files: list[Path]) -> dict[int, int]:
    """Map file index → atom count."""
    counts: dict[int, int] = {}
    for i, f in enumerate(files):
        try:
            counts[i] = _read_v_vib_shape(f)[0] // 3
        except Exception as e:
            print(f"  WARN: could not read V_vib from {f.name}: {e}")
    return counts


def pick_one_per_atom_target(counts: dict[int, int], targets: list[int]) -> dict[int, int]:
    """For each target atom count, return {target → file index of closest match}.

    No duplicates: once a file is picked, it can't be picked again, so each panel shows a
    distinct molecule even if two targets land on the same closest match.
    """
    picks: dict[int, int] = {}
    used: set[int] = set()
    items = list(counts.items())
    for t in targets:
        candidates = [(i, n) for i, n in items if i not in used]
        if not candidates:
            raise RuntimeError("Ran out of distinct molecules to assign")
        best_i, best_n = min(candidates, key=lambda c: (abs(c[1] - t), c[0]))
        picks[t] = best_i
        used.add(best_i)
        print(f"  target={t:>3d} atoms → {best_n} atoms")
    return picks


def load_frame(path: Path, rng, delta: bool = False) -> dict:
    """One random (trajectory, frame) of one molecule: raw S0 and the normalised target.

    With delta, also the reference frame 0 of the same trajectory, and the target is
    p(t) − p(0); frame t = 0 (an identically zero target) is redrawn.
    """
    d = np.load(path)
    S0, pd = d["S0"], d["pd_true"]
    ti, fr = int(rng.integers(S0.shape[0])), int(rng.integers(S0.shape[1]))
    if delta and fr == 0:
        fr = 1 + int(rng.integers(S0.shape[1] - 1))
    p = pd[ti, fr].astype(np.float64)
    out = dict(mol_id=int(d["mol_idx"]), traj=ti, frame=fr,
               S0=S0[ti, fr].astype(np.float32), self_scatter=d["self_scatter"].astype(np.float32),
               fe=d["fe"].astype(np.float32), p=p / p.sum(),
               q=d["q"].astype(np.float32), R=d["R_grid"].astype(np.float32))
    if delta:
        p0 = pd[ti, 0].astype(np.float64)
        out.update(S0_ref=S0[ti, 0].astype(np.float32), p=out["p"] - p0 / p0.sum())
    return out


# ── models ─────────────────────────────────────────────────────────────────

def load_v1(checkpoint_path: Path, q: np.ndarray, nR: int):
    from models.lpd.code.learned_primal_dual import LearnedPrimalDual, build_dbt_operator
    A_dbt, _ = build_dbt_operator(q, nR + 1, verbose=False)
    model = LearnedPrimalDual(A_dbt)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded v1 checkpoint {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def load_v2(checkpoint_path: Path):
    from models.lpd.code.learned_primal_dual_v2 import LearnedPrimalDualV2
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = LearnedPrimalDualV2(ckpt["q"], ckpt["R_grid"], **ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded v2 checkpoint {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def load_tuned(path: Path, qmin: float, delta: bool) -> dict:
    """Baseline parameter k_dbt tuned by eval_pd_recovery.py for this condition."""
    with open(path) as fh:
        res = json.load(fh)
    a = res["args"]
    if abs(a["qmin"] - qmin) > 1e-9 or a.get("snr_db") is not None or bool(a.get("delta")) != delta:
        raise SystemExit(f"{path} was tuned for q_min={a['qmin']}, snr={a.get('snr_db')}, "
                         f"delta={a.get('delta')} — not this condition (q_min={qmin}, delta={delta}); "
                         f"pass --tuned")
    print(f"Baseline parameters from {path}: {res['tuned']}")
    return res["tuned"]


def w1_rcut(pred: np.ndarray, tgt: np.ndarray, dR: float, margin: float = 1.0) -> float:
    """W1 up to the last nonzero target bin + margin (eval_pd_recovery definition)."""
    nz = np.nonzero(tgt)[0]
    cut = len(tgt) if len(nz) == 0 else min(len(tgt), int(nz.max()) + int(np.ceil(margin / dR)) + 1)
    return float(np.abs(np.cumsum(pred - tgt))[:cut].sum() * dR)


def r2_blurred(pred: np.ndarray, tgt: np.ndarray, sigma: float, dR: float) -> float:
    """R² after blurring prediction and truth by a Gaussian of σ [Å] (eval_pd_recovery definition)."""
    pb = gaussian_filter1d(pred, sigma / dR, mode="constant")
    t = gaussian_filter1d(tgt, sigma / dR, mode="constant")
    sst = ((t - t.mean()) ** 2).sum()
    return float(1 - ((pb - t) ** 2).sum() / sst) if sst > 0 else float("nan")


# ── main ───────────────────────────────────────────────────────────────────

def run(args):
    from models.lpd.code.learned_primal_dual_v2 import process_scattering

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    files = pool_files(ROOT / args.data_dir, ROOT / args.split, args.pool)
    print(f"Pool '{args.pool}': {len(files)} molecules. Scanning V_vib headers for atom counts ...")
    counts = scan_atom_counts(files)
    targets = [int(t) for t in args.target_atoms.split(",")]
    picks = pick_one_per_atom_target(counts, targets)

    rng = np.random.default_rng(args.seed)
    panels = [dict(target_atoms=t, actual_atoms=counts[picks[t]],
                   **load_frame(files[picks[t]], rng, args.delta)) for t in targets]
    q, R = panels[0]["q"], panels[0]["R"]
    dR = float(R[1] - R[0])
    n = len(panels)

    # Process frame t (and, with --delta, the reference frame 0) exactly as the harness does.
    keys = ["S0", "S0_ref"] if args.delta else ["S0"]
    S0 = torch.tensor(np.stack([p[k] for k in keys for p in panels]))           # (n·len(keys), nq)
    ss = torch.tensor(np.stack([p["self_scatter"] for _ in keys for p in panels]))
    fe = torch.tensor(np.stack([p["fe"] for _ in keys for p in panels]))
    dS0, W = process_scattering(S0, ss, fe, torch.tensor(q), torch.full((len(S0),), args.qmin))

    def diff(a):
        """Rows [frame t | frame 0] → t − 0 with --delta; identity otherwise."""
        return a[:n] - a[n:] if args.delta else a

    with torch.no_grad():
        if "v2" in models:
            pred, _ = load_v2(ROOT / args.checkpoint_v2)(dS0, W)
            for p, x in zip(panels, diff(pred.numpy().astype(np.float64))):
                p.setdefault("preds", {})["v2"] = x
        if "v1" in models:
            out = load_v1(ROOT / args.checkpoint, q, len(R))(dS0).numpy().astype(np.float64)
            for p, x in zip(panels, diff(out / out.sum(-1, keepdims=True))):
                p.setdefault("preds", {})["v1"] = x

    if "dbt" in models:
        from models.lpd.code.learned_primal_dual import _dbt
        if args.tuned:
            tuned_path = ROOT / args.tuned
        elif args.delta:
            tuned_path = ROOT / "analysis/pd_recovery/results/lpd_v2/pd_eval_delta.json"
        else:
            tuned_path = ROOT / "analysis/pd_recovery/results/lpd_v2" / (
                "pd_eval_clean.json" if args.qmin <= 0 else f"pd_eval_bs_q{args.qmin * 10:02.0f}.json")
        tuned = load_tuned(tuned_path, args.qmin, args.delta)
        q64 = q.astype(np.float64)
        M = len(R) + 1
        x = diff(dS0.numpy().astype(np.float64))                                # (n, nq) signal or Δsignal
        y = _dbt(0, x.T, q64, M)[0]                                             # (nR, n)
        for p, v in zip(panels, (tuned["k_dbt"] * y).T):
            p.setdefault("preds", {})["dbt"] = v

    if "nsk_paper" in models:
        from models.nsk.code.nsk import NSK
        nsk_p = NSK(q.astype(np.float64), W[0].numpy().astype(np.float64), args.nsk_dr)
        stems, info = nsk_p.solve(diff(dS0.numpy().astype(np.float64)), positive=not args.delta)
        fine = nsk_p.to_grid(stems, R.astype(np.float64) + dR / 2)
        print(f"nsk_paper: dictionary ΔR = {nsk_p.dR:.4f} Å ({len(nsk_p.R)} kernels); per panel "
              f"ε/ε_max = {', '.join(f'{v:.1e}' for v in info['lam'])}; stems ≠ 0: {info['nnz']}")
        for p, v, s in zip(panels, fine, stems):
            p.setdefault("preds", {})["nsk_paper"] = v
            p["stems"] = (nsk_p.R, s)

    for p in panels:     # legend order = order given in --models
        p["preds"] = {m: p["preds"][m] for m in models if m in p["preds"]}
        p["w1"] = {m: w1_rcut(v, p["p"], dR) for m, v in p["preds"].items()}
        if args.delta:
            p["r2"] = {m: r2_blurred(v, p["p"], 0.1, dR) for m, v in p["preds"].items()}

    if args.plot:
        out_path = ROOT / args.plot
    else:
        name = f"lpd_predictions{'_delta' if args.delta else ''}_qmin{args.qmin:g}"
        out_path = ROOT / "analysis/pd_recovery/results/lpd_v2/figures" / (name + (f"_blur{args.blur:g}" if args.blur > 0 else "") + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(panels, R, out_path, args.r_max, args.qmin, args.pool, args.delta, args.blur)
    print(f"Wrote {out_path}")


def _plot(panels, R, outpath, r_max, qmin, pool, delta=False, blur=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dR = float(R[1] - R[0])
    show = (lambda v: gaussian_filter1d(v, blur / dR, mode="constant")) if blur > 0 else (lambda v: v)

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)

    for ax, p in zip(axes.flat, panels):
        if delta:
            ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        ax.plot(R, show(p["p"]), "k-", lw=1.0, label="ground truth")
        for m, v in p["preds"].items():
            st = dict(STYLE[m])
            st["label"] = f"{st['label']} (W1 {p['w1'][m]:.3f} Å" + (
                f", R²@0.1 {p['r2'][m]:.2f})" if delta else ")")
            if m == "nsk_paper" and blur <= 0:            # the ℓ1 weights as stems, as in the paper
                Rs, s = p["stems"]
                nz = np.abs(s) > 1e-12 * max(np.abs(s).max(), 1e-300)
                ax.vlines(Rs[nz], 0, s[nz], color=st["color"], lw=1.0, zorder=st["zorder"])
                ax.plot(Rs[nz], s[nz], "o", color=st["color"], ms=3, zorder=st["zorder"], label=st["label"])
            else:
                ax.plot(R, show(v), **st)
        frames = f"frame={p['frame']} − frame 0" if delta else f"frame={p['frame']}"
        ax.set_title(f"{p['actual_atoms']} atoms | mol={p['mol_id']}  traj={p['traj']}  {frames}", fontsize=9)
        ax.set_xlabel("R (Å)")
        ax.set_ylabel(("Δ " if delta else "") + "fraction of pair weight per bin")
        ax.set_xlim(0, r_max)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    cond = "full q-range" if qmin <= 0 else f"beamstop q ≥ {qmin:g} Å⁻¹"
    what = "Pair-density difference p(t) − p(0)" if delta else "Normalised pair density"
    shown = f", curves blurred by σ = {blur:g} Å for display" if blur > 0 else ""
    if blur <= 0 and any("stems" in p for p in panels):
        shown += "\nstems = NSK-ℓ1 weight per dictionary kernel (one per ≈0.1 Å); lines = weight per 0.025 Å bin"
    fig.suptitle(f"{what}, {pool} molecules, {cond}, noise-free{shown}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/trajectory_data_cap200")
    p.add_argument("--split", default="data/splits/split_lpd.npy",
                   help="molecule permutation; first 10%% are validation molecules")
    p.add_argument("--pool", default="val", choices=("val", "train", "all"),
                   help="which molecules to pick from")
    p.add_argument("--models", default="v2,nsk_paper,dbt",
                   help="comma-separated subset of v2,nsk_paper,dbt,v1 (legend order)")
    p.add_argument("--nsk-dr", type=float, default=0.1,
                   help="nsk_paper dictionary resolution [Å] (paper Fig. 5: ≈ 0.1 Å, i.e. M/N ≈ 3)")
    p.add_argument("--tuned", default=None,
                   help="harness results JSON with the baselines' tuned parameters for this condition "
                        "(default: analysis/pd_recovery/results/lpd_v2/pd_eval_clean.json or pd_eval_bs_q08.json)")
    p.add_argument("--checkpoint-v2", default="models/lpd/code/lpd_v2_final.pt")
    p.add_argument("--checkpoint", default="models/lpd/code/lpd_final.pt", help="LPD v1 checkpoint")
    p.add_argument("--qmin", type=float, default=0.0, help="beamstop edge [Å⁻¹]; 0 = full q-range")
    p.add_argument("--delta", action="store_true",
                   help="plot Δp = p(t) − p(0) of the same trajectory instead of p(t)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="display-only Gaussian σ [Å] applied to all curves (metrics use raw histograms)")
    p.add_argument("--target-atoms", default="5,10,25,45",
                   help="Comma-separated atom counts to find molecules for")
    p.add_argument("--r-max", type=float, default=10.0, help="Upper x-axis limit (Å)")
    p.add_argument("--plot", default=None,
                   help="output path (default analysis/pd_recovery/results/lpd_v2/figures/lpd_predictions[_delta]_qmin<q>[_blur<σ>].png)")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
