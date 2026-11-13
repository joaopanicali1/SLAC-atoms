"""
eval_pd_recovery.py — Benchmark pair-density (PD) recovery methods at matched resolution.

Compares, on the held-out molecules of the LPD train/val split, how well each method recovers
the ground-truth pair density of a single frame from its scattering signal:

    dbt       Discrete Bessel Transform of dS0 (the naive inversion, = pd_limited in the npz
              files), times one global scale k fitted on noise-free training frames (k is a fixed
              normalisation constant of the transform for a given q-range, not a noise parameter).
    tikhonov  Ridge-regularised least squares through the Debye forward operator F,
              p = (FᵀF + λI)⁻¹ Fᵀ dS0, with λ tuned on training molecules. This is the best
              *linear* estimator; it uses no prior beyond smoothness.
    nsk_paper NSK-ℓ1 as published (Natan, PRA 107, 023105 (2023); models/nsk/code/nsk.py): DBT on its own
              coarse grid of spacing --nsk-dr (≈ 0.1 Å as in Fig. 5 of the paper; the dictionary
              resolution, below the ≈ 0.31 Å diffraction limit π/q_max), dictionary of NSKs on the
              same grid, the EXACT lasso minimiser (homotopy path), and ε chosen per frame at the
              corner of the L-curve (as in the paper's Fig. 2d). Uses no training data and no
              ground truth. Non-negative for absolute PDs, signed for --delta (as in the paper).
              The output is a set of stems (ℓ1 weights) on the coarse grid; for the metrics each
              stem is split between the two nearest 0.025 Å bins (conserves weight and centroid,
              so W1 is unchanged). R2@0 therefore compares a 0.1 Å-spaced comb with the 0.025 Å
              histogram; R2@σ ≥ 0.05 is the fair comparison (the paper itself blurs the stems by
              the minimum separation δR ≈ 0.15 Å FWHM, i.e. σ ≈ 0.064 Å, for display).
    nsk_paper_tuned  The same exact NSK-ℓ1 path, but with ONE global ε = λ_rel · ε_max (ε_max =
              max|Dᵀy| of each frame) tuned on training molecules like the other baselines. Tells
              apart the method from its ε rule. Shares the path computation with nsk_paper.
              (Results files written before 2026-11-11 contain "nsk_l1" instead: the pipeline's
              ADMM port of decon_TVL1 on the 0.025 Å grid. With its 500-iteration cap it does not
              converge on that highly correlated dictionary and returned a smooth, non-sparse
              iterate rather than the lasso minimiser, so it was removed from the harness.
              models/nsk/code/nsk.admm_lasso keeps that solver for comparison.)
    lpd       Trained LearnedPrimalDual checkpoint (models/lpd/code/learned_primal_dual.py).
    lpd_v2    Trained LearnedPrimalDualV2 checkpoint (models/lpd/code/learned_primal_dual_v2.py), given the
              processed signal and the analysis window of the condition. Default checkpoint:
              models/lpd/code/lpd_v2_final.pt (--checkpoint-v2).

Units
-----
Everything is compared as the NORMALISED pair density

    p(R) = pd(R) / Σ_{a<b} Z_a Z_b        (fraction of total pair weight per R bin, Σ_R p = 1).

The input dS0 = (S0 − Σf²)/fe · Lorch is divided by fe ∝ ΣZZ, so the absolute scale of pd is
not in the input; p is the quantity the scattering actually determines. The linear methods
produce p directly (up to the global k for dbt); the LPD output is in absolute units with an
unknown per-molecule scale, so it is divided by its own sum (uses no ground-truth information).

Metrics (per frame; the table reports medians over all evaluated frames)
-------
    mass     Σ_R p̂. Should be 1 (0 in --delta mode).
    W1 [Å]   1-D Wasserstein (earth-mover) distance Σ_{R ≤ R_cut} |CDF(p̂) − CDF(p)| · ΔR, i.e. the
             average distance the predicted pair weight must be moved to match the truth.
             Resolution-free: a peak displaced by δ costs δ × its weight, rather than 2 × weight²
             as in MSE. R_cut = longest true pair distance of the frame + 1 Å, so that a small
             total-mass error is not multiplied by the empty 31 Å tail of the R grid.
    junk     Σ_{R > R_cut} |p̂| — predicted weight placed beyond any real pair distance
             (hallucinated or ringing structure).
    R2@σ     1 − ‖G_σ∗p̂ − G_σ∗p‖² / ‖G_σ∗p − mean‖²: R² between prediction and truth after blurring
             BOTH with a Gaussian of standard deviation σ [Å], i.e. agreement on all structure
             coarser than σ. A perfect prediction scores 1 at every σ; the σ at which a method
             reaches R² ≈ 0.9 is its effective resolution. σ = 0 is the raw 0.025 Å histogram. For
             qmax = 10 Å⁻¹ with a Lorch window the DBT's own resolution is roughly σ ≈ 0.1–0.15 Å,
             so gains at σ ≤ 0.05 Å are what "super-resolution" means here. Frames whose target has
             zero variance (in --delta mode: no pair changed histogram bin) are excluded from R²;
             the number of frames used is reported as n_R2.

Measurement conditions (applied to the stored S0, as an experiment would be processed)
----------------------
    --qmin, --qmax  observed q-range [Å⁻¹]. Points outside are discarded (beamstop / detector
                    edge) and the Lorch window is re-applied over the observed points only.
                    qmin=0, qmax=10 reproduces the stored dS0 exactly (checked at start-up).
    --snr-db        white Gaussian noise added to the fe-normalised signal (S0 − Σf²)/fe of each
                    frame, before the window, with SNR = 10·log10(var_q(signal) / σ_noise²) over the
                    observed q points. Same convention as the noise analyses in the report
                    (Fig. 2b, App. D). Real photon-counting noise grows relative to the signal at
                    high q, so this convention is optimistic there. Omit for noise-free data.
    --delta         time-difference task: input dS0(t) − dS0(0) of the same trajectory, target
                    p(t) − p(0). This is the metric of Table 2 of the report. (The trajectories have
                    random vibrational phases, so this ΔPD is single-molecule thermal jitter, not a
                    coherent pump-probe signal.)

The LPD checkpoint was trained noise-free on 0 ≤ q ≤ 10 Å⁻¹ with absolute targets; any other
condition is out of distribution for it.

Train/val split: identical to models/lpd/training/train_lpd.py — data/splits/split_lpd.npy holds a permutation of the
sorted file list; the first 10% are validation molecules. Tuning (k, λ, the global ε of
nsk_paper_tuned) uses --n-tune molecules from the training part only.

Run time: the exact NSK path costs ≈ 0.1–0.4 s per frame (one path per frame serves both
nsk_paper variants); everything else is ≈ 0.15 s per frame.

Usage
-----
    python analysis/pd_recovery/eval_pd_recovery.py                          # 20 val molecules, clean
    python analysis/pd_recovery/eval_pd_recovery.py --qmin 0.8 --snr-db 20   # beamstop + noise
    python analysis/pd_recovery/eval_pd_recovery.py --delta                  # ΔPD (report Table 2 metric)
    python analysis/pd_recovery/eval_pd_recovery.py --n-mol 0 --out analysis/pd_recovery/results/lpd_v2/pd_eval_clean.json   # all val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import spherical_jn

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.lpd.code.learned_primal_dual import LearnedPrimalDual, build_dbt_operator, _dbt  # noqa: E402
from models.lpd.code.learned_primal_dual_v2 import LearnedPrimalDualV2  # noqa: E402
from models.nsk.code.nsk import NSK, path_at  # noqa: E402

ALL_METHODS = ("dbt", "tikhonov", "nsk_paper", "nsk_paper_tuned", "lpd", "lpd_v2")
NSK_METHODS = ("nsk_paper", "nsk_paper_tuned")
TIKHONOV_GRID = 10.0 ** -np.arange(1, 9)            # λ relative to σ_max(F)²
# global ε of nsk_paper_tuned, as a fraction of ε_max = max|Dᵀy| of each frame (the L-curve picks
# ≈ 1e-4–1e-2 on these data; noise-free data favour small ε, ε → 0 being basis pursuit
# min ‖w‖₁ s.t. Dw = PD0); must stay ≥ NSK_PATH_MIN, where the computed path ends
NSK_PAPER_GRID = (1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
NSK_PATH_MIN = 1e-8
# The L-curve is evaluated over the path from ε_max down to NSK_LCURVE_MIN · ε_max — the range of
# models.nsk.NSK.solve's default, so the corner is the one of analysis/pd_recovery/replicate_nsk_paper.py and
# the nsk_paper curves of analysis/pd_recovery/plot_lpd_predictions.py (the knee depends on the range shown).
NSK_LCURVE_MIN = 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# Measurement model
# ═════════════════════════════════════════════════════════════════════════════

def lorch_window(mask: np.ndarray) -> np.ndarray:
    """Lorch window over the observed q points, normalised to unit sum.

    Identical to analyse_mol_v2.lorch() when every point is observed.
    """
    w = np.zeros(mask.shape)
    w[mask] = np.sin(np.linspace(0, np.pi, int(mask.sum())))
    return w / w.sum()


class Measurement:
    """Observed q-range and analysis window for one evaluation condition."""

    def __init__(self, q: np.ndarray, qmin: float, qmax: float):
        self.mask = (q >= qmin) & (q <= qmax + 1e-6)
        if self.mask.sum() < 10:
            raise ValueError(f"only {self.mask.sum()} q points in [{qmin}, {qmax}]")
        self.W = lorch_window(self.mask)

    def process(self, S0, self_scatter, fe, snr_db, rng):
        """S0 (n, nq) → dS0 = [(S0 − Σf²)/fe (+ noise)] · window, zero outside the q-range."""
        x = (S0 - self_scatter) / fe
        if snr_db is not None:
            sd = x[:, self.mask].std(-1, keepdims=True) * 10 ** (-snr_db / 20)
            x = x + rng.normal(size=x.shape) * sd
        return x * self.W


# ═════════════════════════════════════════════════════════════════════════════
# Operators for the baselines
# ═════════════════════════════════════════════════════════════════════════════

class Operators:
    """
    F   : (nq, nR) Debye forward operator for the normalised PD,  dS0 ≈ F p,
          F[i, m] = 2 · W(q_i) · j0(q_i · Rc_m),  Rc = bin centres of the pd_true histogram.
          Exact up to the point-atom effective form factor fe = ΣZZ · f̂² used in the pipeline.
    nsk : models.nsk.NSK for this q-range and window at dictionary resolution nsk_dr [Å]
          (None if nsk_dr is None): D[:, m] = DBT(W · j0(q · R_m)) on the DBT's own coarse grid R_m,
          so DBT(dS0) ≈ D (2p).
    """

    def __init__(self, q: np.ndarray, R: np.ndarray, meas: Measurement, nsk_dr: float | None = None):
        self.q = q.astype(np.float64)
        self.M = len(R) + 1                      # DBT order: nR = M − 1
        self.dR = float(R[1] - R[0])
        self.Rc = R.astype(np.float64) + self.dR / 2
        J = spherical_jn(0, np.outer(self.q, self.Rc))

        self.F = 2 * meas.W[:, None] * J
        self._svd = np.linalg.svd(self.F, full_matrices=False)
        self.nsk = None if nsk_dr is None else NSK(self.q, meas.W, nsk_dr)

    def dbt(self, x: np.ndarray) -> np.ndarray:
        """DBT of each row of x (n, nq) → (n, nR). Same routine that produced pd_limited."""
        return _dbt(0, x.T.astype(np.float64), self.q, self.M)[0].T

    def tikhonov_matrix(self, lam_rel: float) -> np.ndarray:
        """(FᵀF + λI)⁻¹Fᵀ with λ = lam_rel · σ_max(F)²; shape (nR, nq)."""
        U, s, Vt = self._svd
        lam = lam_rel * s[0] ** 2
        return (Vt.T * (s / (s ** 2 + lam))) @ U.T


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def split_files(data_dir: Path, split_path: Path, val_frac: float = 0.1):
    files = sorted(data_dir.glob("mol_*_pd.npz"))
    perm = np.load(split_path)
    if len(perm) != len(files):
        print(f"WARNING: split has {len(perm)} entries but {len(files)} files are present; "
              f"the split may not match the one used for training.")
    n_val = max(1, int(val_frac * len(files)))
    return [files[i] for i in perm[:n_val]], [files[i] for i in perm[n_val:]]


def load_samples(path: Path, n: int, rng, delta: bool, with_stored: bool = False) -> dict:
    """Draw n random (trajectory, frame ≥ 1) samples from one molecule file."""
    d = np.load(path)
    S0, pd = d["S0"], d["pd_true"]
    n_traj, n_frames, _ = S0.shape
    flat = rng.choice(n_traj * (n_frames - 1), size=min(n, n_traj * (n_frames - 1)), replace=False)
    ti, fi = flat // (n_frames - 1), 1 + flat % (n_frames - 1)
    s = dict(
        name=path.stem,
        S0=S0[ti, fi].astype(np.float64),
        pd=pd[ti, fi].astype(np.float64),
        self_scatter=d["self_scatter"].astype(np.float64),
        fe=d["fe"].astype(np.float64),
    )
    if delta:
        s["S0_ref"] = S0[ti, 0].astype(np.float64)
        s["pd_ref"] = pd[ti, 0].astype(np.float64)
    if with_stored:
        s["dS0_stored"] = d["dS0"][ti, fi].astype(np.float64)
        s["pdlim_stored"] = d["pd_limited"][ti, fi].astype(np.float64)
    return s


def normalise(pd: np.ndarray) -> np.ndarray:
    return pd / pd.sum(-1, keepdims=True)


def prepare(s: dict, meas: Measurement, ops: Operators, snr_db, rng, delta: bool) -> dict:
    """Process one molecule's samples into method inputs (x, its DBT y, raw g for LPD) and targets."""
    g = meas.process(s["S0"], s["self_scatter"], s["fe"], snr_db, rng)
    p = normalise(s["pd"])
    b = dict(name=s["name"], g=g)
    if delta:
        g0 = meas.process(s["S0_ref"], s["self_scatter"], s["fe"], snr_db, rng)
        b.update(g0=g0, x=g - g0, tgt=p - normalise(s["pd_ref"]))
    else:
        b.update(x=g, tgt=p)
    b["y"] = ops.dbt(b["x"])
    return b


# ═════════════════════════════════════════════════════════════════════════════
# Methods and metrics
# ═════════════════════════════════════════════════════════════════════════════

def lpd_normalised(model, g: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(g, dtype=torch.float32)).numpy().astype(np.float64)
    tot = out.sum(-1, keepdims=True)
    tot[np.abs(tot) < 1e-12] = 1e-12
    return out / tot


def lpd_v2_predict(model_v2, g: np.ndarray, W: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        gt = torch.tensor(g, dtype=torch.float32)
        Wt = torch.tensor(W, dtype=torch.float32).expand_as(gt)
        p_hat, _ = model_v2(gt, Wt)
    return p_hat.numpy().astype(np.float64)


def nsk_paths(b: dict, ops: Operators, delta: bool) -> list:
    """
    Exact lasso path of every frame of batch b on the paper's NSK dictionary, computed once and
    cached in b (b["nsk_y"]: PD0 on the coarse grid; b["nsk_paths"]: (λ breakpoints, solutions)).
    Non-negative for absolute PDs; signed for ΔPD, which has negative lobes (depleted distances).
    """
    if "nsk_paths" not in b:
        b["nsk_y"] = ops.nsk.transform(b["x"])
        b["nsk_paths"] = [ops.nsk.path(y, positive=not delta, lam_min_rel=NSK_PATH_MIN) for y in b["nsk_y"]]
    return b["nsk_paths"]


def nsk_predict(b: dict, ops: Operators, delta: bool, lam_rel: float | None):
    """
    Paper NSK-ℓ1 for every frame of b, ε by the L-curve (lam_rel None) or ε = lam_rel · ε_max.
    Returns p on the 0.025 Å evaluation grid (stems split between neighbouring bins, see
    NSK.to_grid) and per-frame (ε/ε_max, number of non-zero stems).
    """
    paths = nsk_paths(b, ops, delta)                   # fills b["nsk_y"] on first use
    stems, eps, nnz = [], [], []
    for y, (lams, Z) in zip(b["nsk_y"], paths):
        if lam_rel is None and lams[-1] < NSK_LCURVE_MIN * lams[0]:
            # cut the path at NSK_LCURVE_MIN · ε_max (exact: the path is linear between breakpoints)
            lo = NSK_LCURVE_MIN * lams[0]
            keep = lams > lo
            lams, Z = np.append(lams[keep], lo), np.vstack([Z[keep], path_at(lams, Z, np.array([lo]))])
        p, e, k, _ = ops.nsk.weights(y, lams, Z, lam_rel)
        stems.append(p)
        eps.append(e)
        nnz.append(k)
    return ops.nsk.to_grid(np.array(stems), ops.Rc), np.array(eps), np.array(nnz)


def predict(b: dict, method: str, cfg: dict, ops: Operators, model, delta: bool) -> np.ndarray:
    if method == "dbt":
        return cfg["k_dbt"] * b["y"]
    if method == "tikhonov":
        return b["x"] @ cfg["P_tik"].T
    if method in NSK_METHODS:
        lam_rel = None if method == "nsk_paper" else cfg["nsk_paper_lam_rel"]
        p, eps, nnz = nsk_predict(b, ops, delta, lam_rel)
        b.setdefault("nsk_info", {})[method] = dict(eps_rel=eps, nnz=nnz)
        return p
    if method == "lpd":
        f = lpd_normalised(model, b["g"])
        return f - lpd_normalised(model, b["g0"]) if delta else f
    if method == "lpd_v2":
        m2, W = cfg["model_v2"], cfg["window"]
        f = lpd_v2_predict(m2, b["g"], W)
        return f - lpd_v2_predict(m2, b["g0"], W) if delta else f
    raise ValueError(method)


def w1_and_junk(pred: np.ndarray, tgt: np.ndarray, dR: float, margin: float = 1.0):
    """W1 over R ≤ R_cut and |weight| beyond R_cut, with R_cut = last nonzero target bin + margin."""
    nR = tgt.shape[-1]
    last = nR - 1 - np.argmax(tgt[:, ::-1] != 0, axis=-1)
    cut = np.minimum(nR, last + int(np.ceil(margin / dR)) + 1)
    inside = np.arange(nR)[None, :] < cut[:, None]
    w1 = (np.abs(np.cumsum(pred - tgt, axis=-1)) * inside).sum(-1) * dR
    junk = (np.abs(pred) * ~inside).sum(-1)
    return w1, junk


def frame_metrics(pred: np.ndarray, tgt: np.ndarray, sigmas, dR: float) -> dict:
    w1, junk = w1_and_junk(pred, tgt, dR)
    m = {"mass": pred.sum(-1), "W1": w1, "junk": junk}
    for s in sigmas:
        if s <= 0:
            pb, t = pred, tgt
        else:
            pb = gaussian_filter1d(pred, s / dR, axis=-1, mode="constant")
            t = gaussian_filter1d(tgt, s / dR, axis=-1, mode="constant")
        sst = ((t - t.mean(-1, keepdims=True)) ** 2).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1 - ((pb - t) ** 2).sum(-1) / sst
        r2[sst <= 0] = np.nan                      # target identically zero: R² undefined
        m[f"R2@{s:g}"] = r2
    return m


def tune(batches: list[dict], clean_batches: list[dict], methods, ops: Operators, delta: bool) -> dict:
    """
    Fit the free parameter of each baseline on training molecules.
    Criterion (lower is better): median W1 for absolute PDs; for --delta, −median R2@0.1, because
    the trivial prediction ΔPD̂ = 0 already scores W1 = W1(ΔPD, 0) ≈ the best achievable, so W1
    cannot tell a useful ΔPD estimate from zero.
    batches       : tuning frames in the evaluation condition (with noise, if any)
    clean_batches : the same frames without noise — used for the DBT scale k only
    """
    cfg = {}
    X = np.concatenate([b["x"] for b in batches])
    Y = np.concatenate([b["y"] for b in batches])
    T = np.concatenate([b["tgt"] for b in batches])

    crit = "−median R2@0.1" if delta else "median W1 [Å]"
    print(f"  tuning criterion: {crit}")

    def med_w1(P):
        if delta:
            return -float(np.nanmedian(frame_metrics(P, T, [0.1], ops.dR)["R2@0.1"]))
        return float(np.median(w1_and_junk(P, T, ops.dR)[0]))

    if "dbt" in methods:
        Yc = np.concatenate([b["y"] for b in clean_batches])
        Tc = np.concatenate([b["tgt"] for b in clean_batches])
        cfg["k_dbt"] = float((Yc * Tc).sum() / (Yc * Yc).sum())
        print(f"  dbt      : k = {cfg['k_dbt']:.4g} (noise-free fit)   "
              f"({crit} in this condition {med_w1(cfg['k_dbt'] * Y):.4f})")
    if "tikhonov" in methods:
        scores = {lr: med_w1(X @ ops.tikhonov_matrix(lr).T) for lr in TIKHONOV_GRID}
        best = min(scores, key=scores.get)
        cfg["tik_lam_rel"], cfg["P_tik"] = float(best), ops.tikhonov_matrix(best)
        print("  tikhonov : " + "  ".join(f"λ={lr:.0e}:{v:.4f}" for lr, v in scores.items())
              + f"   → λ = {best:.0e}")
    if any(m in methods for m in NSK_METHODS):
        # one exact path per tuning frame serves every ε below (and the L-curve reference)
        lc, lc_eps = [], []
        for b in batches:
            p, eps, _ = nsk_predict(b, ops, delta, None)
            lc.append(p)
            lc_eps.append(eps)
        lc_eps = np.concatenate(lc_eps)
        print(f"  nsk_paper: dictionary ΔR = {ops.nsk.dR:.4f} Å ({len(ops.nsk.R)} kernels); L-curve "
              f"ε/ε_max median {np.median(lc_eps):.1e} (p25–p75 {np.percentile(lc_eps, 25):.1e}–"
              f"{np.percentile(lc_eps, 75):.1e}), {crit} {med_w1(np.concatenate(lc)):.4f}  (no tuning)")
    if "nsk_paper_tuned" in methods:
        scores = {lr: med_w1(np.concatenate([nsk_predict(b, ops, delta, lr)[0] for b in batches]))
                  for lr in NSK_PAPER_GRID}
        best = min(scores, key=scores.get)
        cfg["nsk_paper_lam_rel"] = float(best)
        print("  nsk_paper_tuned: " + "  ".join(f"ε={lr:.0e}:{v:.4f}" for lr, v in scores.items())
              + f"   → ε/ε_max = {best:.0e}")
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main(args):
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = set(methods) - set(ALL_METHODS)
    if bad:
        raise SystemExit(f"unknown method(s) {sorted(bad)}; choose from {ALL_METHODS}")
    sigmas = [float(s) for s in args.sigmas.split(",")]

    data_dir = ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    split_path = ROOT / args.split if not Path(args.split).is_absolute() else Path(args.split)
    val_files, train_files = split_files(data_dir, split_path)
    if args.n_mol > 0:
        val_files = [val_files[i] for i in rng.choice(len(val_files), min(args.n_mol, len(val_files)),
                                                      replace=False)]
    tune_files = [train_files[i] for i in rng.choice(len(train_files), args.n_tune, replace=False)]

    ref = np.load(val_files[0])
    q, R = ref["q"], ref["R_grid"]
    meas = Measurement(q.astype(np.float64), args.qmin, args.qmax)
    print(f"Condition: q ∈ [{args.qmin}, {args.qmax}] Å⁻¹ ({meas.mask.sum()} pts), "
          f"SNR = {'∞' if args.snr_db is None else f'{args.snr_db} dB'}, "
          f"task = {'ΔPD (frame t − frame 0)' if args.delta else 'absolute PD'}")
    print(f"Molecules: {len(val_files)} val × {args.frames_per_mol} frames;  "
          f"{len(tune_files)} train molecules for tuning")
    use_nsk = any(m in methods for m in NSK_METHODS)
    ops = Operators(q, R, meas, args.nsk_dr if use_nsk else None)

    model = None
    if "lpd" in methods:
        ckpt_path = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
        A_dbt, _ = build_dbt_operator(q, len(R) + 1, verbose=False)
        model = LearnedPrimalDual(A_dbt)
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        print(f"LPD checkpoint: {ckpt_path.relative_to(ROOT) if ckpt_path.is_relative_to(ROOT) else ckpt_path}"
              f" (epoch {ck.get('epoch', '?')})")
        if args.qmin > q[1] or args.qmax < q[-1] - 1e-6 or args.snr_db is not None:
            print("  note: LPD was trained noise-free on the full q-range — this condition is "
                  "out of distribution for it.")

    # Sanity check: the processing reproduces the stored arrays in the default condition.
    default_cond = args.qmin <= q[0] and args.qmax >= q[-1] - 1e-6 and args.snr_db is None
    if default_cond:
        s = load_samples(val_files[0], 4, np.random.default_rng(0), delta=False, with_stored=True)
        g = meas.process(s["S0"], s["self_scatter"], s["fe"], None, rng)
        e1 = np.abs(g - s["dS0_stored"]).max() / np.abs(s["dS0_stored"]).max()
        e2 = np.abs(ops.dbt(s["dS0_stored"]) - s["pdlim_stored"]).max() / np.abs(s["pdlim_stored"]).max()
        print(f"Sanity: max rel. diff  processed vs stored dS0 = {e1:.1e},  "
              f"DBT vs stored pd_limited = {e2:.1e}")

    print("Tuning on training molecules ...")
    tune_samples = [load_samples(f, args.frames_per_mol, rng, args.delta) for f in tune_files]
    tune_batches = [prepare(s, meas, ops, args.snr_db, rng, args.delta) for s in tune_samples]
    clean_batches = (tune_batches if args.snr_db is None else
                     [prepare(s, meas, ops, None, rng, args.delta) for s in tune_samples])
    cfg = tune(tune_batches, clean_batches, methods, ops, args.delta)
    del tune_batches, clean_batches                    # drop the cached NSK paths of the tuning frames
    if "lpd_v2" in methods:
        p2 = ROOT / args.checkpoint_v2 if not Path(args.checkpoint_v2).is_absolute() else Path(args.checkpoint_v2)
        ck2 = torch.load(p2, map_location="cpu")
        m2 = LearnedPrimalDualV2(ck2["q"], ck2["R_grid"], **ck2["config"])
        m2.load_state_dict(ck2["model_state_dict"])
        m2.eval()
        cfg["model_v2"], cfg["window"] = m2, meas.W
        print(f"LPD v2 checkpoint: {p2} (epoch {ck2.get('epoch', '?')})")

    print("Evaluating ...")
    per_frame = {m: [] for m in methods}
    nsk_info = {m: dict(eps_rel=[], nnz=[]) for m in methods if m in NSK_METHODS}
    for i, f in enumerate(val_files, 1):
        b = prepare(load_samples(f, args.frames_per_mol, rng, args.delta), meas, ops,
                    args.snr_db, rng, args.delta)
        for m in methods:
            per_frame[m].append(frame_metrics(predict(b, m, cfg, ops, model, args.delta),
                                              b["tgt"], sigmas, ops.dR))
        for m, d in nsk_info.items():
            for k in d:
                d[k].append(b["nsk_info"][m][k])
        if i % 5 == 0 or i == len(val_files):
            print(f"  {i}/{len(val_files)} molecules  ({time.time() - t_start:.0f} s)")

    keys = ["mass", "W1", "junk"] + [f"R2@{s:g}" for s in sigmas]
    summary = {}
    for m in methods:
        cat = {k: np.concatenate([d[k] for d in per_frame[m]]) for k in keys}
        summary[m] = {k: {"median": float(np.nanmedian(v)), "mean": float(np.nanmean(v)),
                          "p25": float(np.nanpercentile(v, 25)), "p75": float(np.nanpercentile(v, 75)),
                          "n": int(np.isfinite(v).sum())}
                      for k, v in cat.items()}

    n_frames = sum(len(d["W1"]) for d in per_frame[methods[0]])
    n_r2 = summary[methods[0]][keys[-1]]["n"]
    print(f"\nMedian over {n_frames} frames (R²: n_R2 = {n_r2} frames with non-zero target variance)."
          f"  R2@σ: prediction and truth both blurred by σ Å;  W1 in Å")
    print(f"{'method':>15} " + " ".join(f"{k:>9}" for k in keys))
    for m in methods:
        print(f"{m:>15} " + " ".join(f"{summary[m][k]['median']:9.4f}" for k in keys))

    # sparsity of the NSK solutions: ε/ε_max chosen (per frame for the L-curve) and stems kept
    nsk_stats = {}
    for m, d in nsk_info.items():
        nsk_stats[m] = {k: {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                            "p75": float(np.percentile(v, 75))}
                        for k, v in ((k, np.concatenate(v)) for k, v in d.items())}
        s = nsk_stats[m]
        print(f"{m}: ε/ε_max median {s['eps_rel']['median']:.1e} (p25–p75 {s['eps_rel']['p25']:.1e}–"
              f"{s['eps_rel']['p75']:.1e}),  non-zero stems median {s['nnz']['median']:.0f} "
              f"(p25–p75 {s['nnz']['p25']:.0f}–{s['nnz']['p75']:.0f}) of {len(ops.nsk.R)}")

    if args.out:
        out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tuned = {k: v for k, v in cfg.items() if k not in ("P_tik", "model_v2", "window")}
        with open(out_path, "w") as fh:
            json.dump({"args": vars(args), "tuned": tuned, "n_frames": n_frames,
                       "val_molecules": [f.stem for f in val_files],
                       "tune_molecules": [f.stem for f in tune_files],
                       "summary": summary, "nsk_stats": nsk_stats}, fh, indent=2)
        print(f"Wrote {out_path}")
    print(f"Done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/trajectory_data_cap200",
                   help="directory of mol_*_pd.npz files")
    p.add_argument("--split", default="data/splits/split_lpd.npy",
                   help="file permutation saved by models/lpd/training/train_lpd.py (first 10%% = validation)")
    p.add_argument("--checkpoint", default="models/lpd/code/lpd_final.pt", help="LPD checkpoint")
    p.add_argument("--checkpoint-v2", default="models/lpd/code/lpd_v2_final.pt",
                   help="LPD v2 checkpoint (used when lpd_v2 is in --methods)")
    p.add_argument("--methods", default="dbt,tikhonov,nsk_paper,nsk_paper_tuned,lpd,lpd_v2",
                   help=f"comma-separated subset of {','.join(ALL_METHODS)}")
    p.add_argument("--nsk-dr", type=float, default=0.1,
                   help="NSK dictionary resolution [Å]: spacing of the DBT grid on which the kernels "
                        "(stems) sit. Paper Fig. 5: ≈ 0.1 Å (M/N ≈ 3); Figs. 2–4: 0.157 Å")
    p.add_argument("--n-mol", type=int, default=20, help="validation molecules to evaluate (0 = all)")
    p.add_argument("--n-tune", type=int, default=4, help="training molecules used to tune baselines")
    p.add_argument("--frames-per-mol", type=int, default=20, help="random frames drawn per molecule")
    p.add_argument("--qmin", type=float, default=0.0, help="lowest observed q [Å⁻¹] (beamstop edge)")
    p.add_argument("--qmax", type=float, default=10.0, help="highest observed q [Å⁻¹] (detector edge)")
    p.add_argument("--snr-db", type=float, default=None,
                   help="per-frame SNR of added white noise on S0 − Σf² [dB]; omit for noise-free")
    p.add_argument("--delta", action="store_true", help="evaluate ΔPD = PD(t) − PD(0) instead of PD")
    p.add_argument("--sigmas", default="0,0.025,0.05,0.1,0.2",
                   help="Gaussian blur widths σ [Å] at which R² is reported")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional JSON path for the full summary")
    main(p.parse_args())
