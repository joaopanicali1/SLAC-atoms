"""
baseline_beamstop.py — Non-ML polynomial-extrapolation baseline for the
beamstop reconstruction task.

For each sample, fit a low-order polynomial in q² to the observed ΔS0_norm
in a local window above q_min, then extrapolate into the masked low-q region.

    p(q) = a₁q² + a₂q⁴ + a₃q⁶                  (parity="even", default)
    p(q) = a₁q  + a₂q² + a₃q³                  (parity="all")

Both forms drop the constant term, enforcing ΔS₀(q=0) = 0 exactly. The
"even" form additionally respects the parity of S₀(q) — S₀ depends on q only
through even functions (sinc(qR), cos(qR)·…), so ΔS₀ should too. This
removes a non-physical degree of freedom (nonzero slope at q=0) that "all"
permits.

Fit by ordinary least squares on q points inside [q_min, q_min + window].
Local windowing keeps the fit anchored to the data nearest the gap rather
than being dominated by high-q noise.

Usage
-----
    from analysis.baseline_beamstop import PolynomialBaseline
    bl = PolynomialBaseline(n_terms=3, parity="even", window=1.5)
    pred = bl.predict(q, delta_S0_masked, mask)  # numpy

Run as a script to evaluate on the same val split as training and (if a
checkpoint is provided) compare against a trained BeamstopModel.

    python analysis/baseline_beamstop.py \
        --checkpoint checkpoints/beamstop_128/best_model.pt \
        --plot beamstop_baseline_vs_model.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Suppress PyTorch C++ warnings (e.g. "Could not initialize NNPACK") on CPU
# nodes lacking the required ISA. Must be set BEFORE `import torch` because
# the c10 logger reads this env var at library init. The NNPACK warning is
# logged from C++, not Python's warnings module, so torch.backends.nnpack
# doesn't silence it.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch

torch.backends.nnpack.enabled = False  # belt-and-braces: skip NNPACK path


# ═════════════════════════════════════════════════════════════════════════════
# Polynomial baseline
# ═════════════════════════════════════════════════════════════════════════════

class PolynomialBaseline:
    """
    Per-sample polynomial extrapolation into the masked region.

    Parameters
    ----------
    n_terms : number of polynomial terms (default 3 → 3 free coefficients).
    parity  : "even" (default) → p(q) = a₁q² + a₂q⁴ + a₃q⁶ + ...
              "all"             → p(q) = a₁q  + a₂q² + a₃q³ + ...
              The "even" form respects ΔS₀'s parity in q (S₀ depends on q
              only through even functions of q·R) and gives zero slope at
              q=0, removing a non-physical degree of freedom. p(0)=0 is
              enforced exactly in both cases by omitting the constant term.
    window  : width of the fit window above q_min, in Å⁻¹ (default 1.5).
              Smaller → more local, more sensitive to noise. Larger →
              smoother, but biased by curvature far from the gap.
    """
    def __init__(self, n_terms: int = 3, parity: str = "even", window: float = 1.5):
        if n_terms < 1:
            raise ValueError("n_terms must be ≥ 1")
        if parity not in ("even", "all"):
            raise ValueError("parity must be 'even' or 'all'")
        self.n_terms = n_terms
        self.parity  = parity
        self.window  = float(window)
        if parity == "even":
            self._powers = [2 * k for k in range(1, n_terms + 1)]   # 2, 4, 6, ...
        else:
            self._powers = list(range(1, n_terms + 1))              # 1, 2, 3, ...

    def _fit_one(self, q_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        X = np.stack([q_fit ** k for k in self._powers], axis=1)
        coef, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        return coef

    def _eval_poly(self, q: np.ndarray, coef: np.ndarray) -> np.ndarray:
        return sum(c * q ** k for c, k in zip(coef, self._powers))

    def predict(self, q, delta_S0_masked, mask) -> np.ndarray:
        """
        Parameters
        ----------
        q                : (nq,)            q grid in Å⁻¹
        delta_S0_masked  : (..., nq)        observed values (zero in masked region)
        mask             : (..., nq)        1=observed, 0=masked

        Returns
        -------
        pred : (..., nq)  Full ΔS0_norm — observed values in the observed
               region, polynomial extrapolation in the masked region.
        """
        q     = np.asarray(q,               dtype=np.float64)
        y_all = np.asarray(delta_S0_masked, dtype=np.float64)
        m_all = np.asarray(mask,            dtype=np.float64)

        squeeze = y_all.ndim == 1
        if squeeze:
            y_all = y_all[None]
            m_all = m_all[None]

        out = y_all.copy()
        for b in range(y_all.shape[0]):
            m   = m_all[b]
            obs = m > 0.5
            if not obs.any():
                continue
            q_min     = q[obs].min()
            fit_in    = obs & (q <= q_min + self.window)
            if fit_in.sum() < self.n_terms:
                fit_in = obs
            coef      = self._fit_one(q[fit_in], y_all[b, fit_in])
            masked_in = ~obs
            out[b, masked_in] = self._eval_poly(q[masked_in], coef)

        return out[0] if squeeze else out


# ═════════════════════════════════════════════════════════════════════════════
# NSK / spherical-Bessel inversion baseline
# (Wang et al., chemRxiv 15002782_v2, §2.1, Eqs. 10–12)
# ═════════════════════════════════════════════════════════════════════════════

class NSKBaseline:
    """
    Inversion-based low-q extrapolation using a spherical-Bessel basis.

    The measured (truncated) ΔS₀(q) is represented as
        ΔS₀(q) ≈ Σ_m  w_m · j₀(q · R_m).

    Workflow per sample (matches the paper):
      1. Initial regularized inversion on the observed q region:
             w = argmin ‖B_obs · w − ΔS₀_obs‖² + ε ‖w‖²,    B[i,m] = j₀(q_i R_m)
      2. Refine to w_ext that simultaneously
           - matches measurement in the observed region,
           - stays close to the inversion prior (weight λ_R),
           - matches value + first + second derivatives at q_min between the
             extrapolated function and the measured data (weight λ_cont),
           - hard-enforces ΔS₀(0) = 0  ⇔  Σ_m w_ext_m · j₀(0·R_m) = Σ w_ext_m = 0.
      3. Extrapolate by evaluating Σ_m w_ext_m · j₀(q · R_m) for q < q_min;
         observed values are kept verbatim above q_min.

    Parameters
    ----------
    R_max       : upper bound of the real-space basis grid (Å). Default 10.
                  Truncating well above the largest interatomic distance avoids
                  fitting noise into long-range modes.
    dR          : real-space grid spacing (Å). Smaller → finer basis, more
                  parameters, slower per-sample solve. Default 0.1 (M ≈ 100).
    eps         : Tikhonov weight for the initial inversion.
    lambda_R    : penalty pulling w_ext toward the initial w.
    lambda_cont : penalty on the (value, ∂_q, ∂²_q) continuity mismatch at q_min.
    lambda_zero : penalty enforcing ΔS₀(0)=0 (effectively a hard constraint
                  when large). Default 1e8.
    """

    def __init__(self, R_max: float = 10.0, dR: float = 0.1,
                 eps: float = 1e-3, lambda_R: float = 1.0,
                 lambda_cont: float = 10.0, lambda_zero: float = 1e8):
        if dR <= 0 or R_max <= dR:
            raise ValueError("require dR > 0 and R_max > dR")
        self.R_grid      = np.arange(dR, R_max + dR / 2, dR, dtype=np.float64)
        self.eps         = float(eps)
        self.lambda_R    = float(lambda_R)
        self.lambda_cont = float(lambda_cont)
        self.lambda_zero = float(lambda_zero)
        self._q_cache: np.ndarray | None = None
        self._B_cache: np.ndarray | None = None

    @staticmethod
    def _spherical_jn(n: int, x: np.ndarray) -> np.ndarray:
        from scipy.special import spherical_jn
        return spherical_jn(n, x)

    def _build_B(self, q: np.ndarray) -> np.ndarray:
        """B[i, m] = j₀(q[i] · R_m)."""
        return self._spherical_jn(0, np.outer(q, self.R_grid))

    def _continuity_vectors(self, q_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (v0, v1, v2) such that for any weight vector w,
            f_ext(q_min)     = v0 · w
            f_ext'(q_min)    = v1 · w
            f_ext''(q_min)   = v2 · w
        with f_ext(q) = Σ_m w_m · j₀(q R_m).
        Uses dj₀/dx = -j₁(x) and d²j₀/dx² = -j₀(x) + (2/x) j₁(x).
        """
        R   = self.R_grid
        qR  = q_min * R
        j0  = self._spherical_jn(0, qR)
        j1  = self._spherical_jn(1, qR)
        v0  = j0
        v1  = -R * j1
        # Limit of -j0 + 2 j1/x at x→0 is -1/3 (j0=1, j1≈x/3).
        d2j0 = np.where(qR > 1e-10, -j0 + 2.0 * j1 / np.where(qR > 1e-10, qR, 1.0), -1.0 / 3.0)
        v2  = R * R * d2j0
        return v0, v1, v2

    def predict(self, q, delta_S0_masked, mask) -> np.ndarray:
        """See PolynomialBaseline.predict for the I/O contract."""
        q     = np.asarray(q,               dtype=np.float64)
        y_all = np.asarray(delta_S0_masked, dtype=np.float64)
        m_all = np.asarray(mask,            dtype=np.float64)

        squeeze = y_all.ndim == 1
        if squeeze:
            y_all = y_all[None]
            m_all = m_all[None]

        # Cache the (nq, M) synthesis matrix on first call.
        if (self._B_cache is None or self._q_cache is None
                or self._q_cache.shape != q.shape
                or not np.allclose(self._q_cache, q)):
            self._B_cache = self._build_B(q)
            self._q_cache = q.copy()
        B      = self._B_cache
        M_dim  = B.shape[1]
        dq     = float(q[1] - q[0])
        Ieye   = np.eye(M_dim)
        v_zero = np.ones(M_dim)   # j₀(0)=1, so Σ w = ΔS₀(0)

        sqrt_lR    = float(np.sqrt(self.lambda_R))
        sqrt_lc    = float(np.sqrt(self.lambda_cont))
        sqrt_lzero = float(np.sqrt(self.lambda_zero))

        out = y_all.copy()
        for b in range(y_all.shape[0]):
            m   = m_all[b]
            obs = m > 0.5
            if obs.sum() < 3:
                continue   # need ≥3 observed points for forward FD of 2nd derivative

            B_obs = B[obs]
            y_obs = y_all[b, obs]

            # Step 1 — initial NSK-style inversion: (BᵀB + εI) w = Bᵀ y
            BtB   = B_obs.T @ B_obs
            Bty   = B_obs.T @ y_obs
            w0    = np.linalg.solve(BtB + self.eps * Ieye, Bty)

            # Boundary index = first observed q
            q_min_idx = int(np.argmax(obs))
            q_min     = q[q_min_idx]
            v0, v1, v2 = self._continuity_vectors(q_min)

            # Forward finite differences for measured-side derivatives at q_min.
            y_at  =  y_all[b, q_min_idx]
            yp_at = (y_all[b, q_min_idx + 1] - y_all[b, q_min_idx]) / dq
            ypp_at = (y_all[b, q_min_idx + 2]
                      - 2.0 * y_all[b, q_min_idx + 1]
                      + y_all[b, q_min_idx]) / (dq * dq)

            # Step 2 — refined LSQ with prior + continuity + ΔS₀(0)=0.
            A = np.vstack([
                B_obs,
                sqrt_lR    * Ieye,
                sqrt_lc    * v0[None, :],
                sqrt_lc    * v1[None, :],
                sqrt_lc    * v2[None, :],
                sqrt_lzero * v_zero[None, :],
            ])
            rhs = np.concatenate([
                y_obs,
                sqrt_lR    * w0,
                [sqrt_lc    * y_at],
                [sqrt_lc    * yp_at],
                [sqrt_lc    * ypp_at],
                [0.0],
            ])
            w_ext, *_ = np.linalg.lstsq(A, rhs, rcond=None)

            masked_in = ~obs
            out[b, masked_in] = B[masked_in] @ w_ext

        return out[0] if squeeze else out


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def masked_mse_per_sample(pred: np.ndarray, target: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """MSE inside the masked region only, per sample."""
    sq    = (pred - target) ** 2
    inv_m = 1.0 - mask
    n     = inv_m.sum(axis=-1).clip(min=1)
    return (sq * inv_m).sum(axis=-1) / n


def load_model(checkpoint_path: Path, B_sinc: torch.Tensor, n_channels: int,
                use_conditioning: bool, device: torch.device):
    from models.beamstop import BeamstopModel
    model = BeamstopModel(B_sinc, n_channels=n_channels,
                          use_conditioning=use_conditioning).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
    return model


def run_eval(args):
    from data.Dataset.dataset_beamstop import BeamstopDataset
    from models.beamstop import build_sinc_operator
    from training.train_beamstop import compute_dataset_R_max, build_truncated_R_grid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BeamstopDataset(args.data_dir, q_min_lo=args.q_min_lo,
                              q_min_hi=args.q_min_hi,
                              use_conditioning=False, seed=args.seed)
    q_np = dataset.q.numpy()
    print(
        f"Dataset: {len(dataset):,} samples from {len(dataset.loaded_files)} loaded "
        f"mols ({len(dataset.files)} found, {len(dataset.skipped_files)} skipped)"
    )

    # Same file-level split as training (seed 42).
    n_files = len(dataset.loaded_files)
    n_val   = max(1, int(0.1 * n_files))
    rng     = np.random.default_rng(42)
    file_indices = np.arange(n_files); rng.shuffle(file_indices)
    val_files    = set(file_indices[:n_val])
    val_idx      = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]
    if args.max_samples:
        val_idx = val_idx[:args.max_samples]
    print(f"Evaluating on {len(val_idx):,} val samples")

    if args.baseline == "nsk":
        baseline      = NSKBaseline(R_max=args.r_max, dR=args.dr_basis,
                                    eps=args.eps, lambda_R=args.lambda_R,
                                    lambda_cont=args.lambda_cont)
        baseline_desc = (f"NSK (R_max={args.r_max} Å, dR={args.dr_basis} Å, "
                         f"M={len(baseline.R_grid)}, λ_R={args.lambda_R}, "
                         f"λ_cont={args.lambda_cont}, ε={args.eps})")
    else:
        baseline      = PolynomialBaseline(n_terms=args.n_terms,
                                           parity=args.parity,
                                           window=args.window)
        baseline_desc = (f"polynomial (parity={args.parity}, "
                         f"n_terms={args.n_terms}, window={args.window} Å⁻¹)")

    # Optional model load.
    model = None
    if args.checkpoint:
        R_max_data       = compute_dataset_R_max(dataset.loaded_files, margin=args.R_margin)
        R_max            = min(R_max_data, args.R_cap)
        dR               = float(dataset.R_grid[1] - dataset.R_grid[0])
        R_grid_truncated = build_truncated_R_grid(R_max, dR=dR)
        B_sinc = build_sinc_operator(q_np, R_grid_truncated, verbose=False)
        print(f"R_max_data={R_max_data:.2f} Å  R_cap={args.R_cap:.2f} Å  "
              f"→ R_max={R_max:.2f} Å  B_sinc={tuple(B_sinc.shape)}")
        model  = load_model(Path(args.checkpoint), B_sinc, args.n_channels,
                            use_conditioning=False, device=device)

    bl_mse_all   = []
    mdl_mse_all  = []
    plot_cache   = []

    # Pick n_plot random positions in val_idx for plotting. Iteration order
    # is (mol, traj, frame) lex, so taking the first n_plot would put every
    # panel on the same molecule + trajectory + consecutive frames. Random
    # positions give variety across molecules and frame indices.
    plot_rng       = np.random.default_rng(args.plot_seed)
    n_to_plot      = min(args.n_plot, len(val_idx))
    plot_positions = set(plot_rng.choice(len(val_idx), size=n_to_plot,
                                         replace=False).tolist()) if args.plot else set()

    for k, idx in enumerate(val_idx):
        sample = dataset[idx]
        y_msk  = sample["delta_S0_masked"].numpy()
        y_tgt  = sample["delta_S0_norm"].numpy()
        m      = sample["mask"].numpy()

        y_bl   = baseline.predict(q_np, y_msk, m)
        bl_mse_all.append(masked_mse_per_sample(y_bl[None], y_tgt[None], m[None])[0])

        y_mdl = None
        if model is not None:
            with torch.no_grad():
                pred, _ = model(sample["delta_S0_masked"].unsqueeze(0).to(device),
                                sample["mask"].unsqueeze(0).to(device))
            y_mdl = pred.cpu().numpy()[0]
            mdl_mse_all.append(masked_mse_per_sample(y_mdl[None], y_tgt[None], m[None])[0])

        if k in plot_positions:
            plot_cache.append({
                "q":      q_np,
                "mask":   m,
                "target": y_tgt,
                "baseline": y_bl,
                "model":    y_mdl,
                "label":   f"mol={sample['mol_idx']} traj={sample['traj_idx']} fr={sample['frame_idx']}",
            })

    bl_mse_all  = np.asarray(bl_mse_all)
    print(f"\nBaseline: {baseline_desc}")
    print(f"  masked-MSE  mean = {bl_mse_all.mean():.6e}")
    print(f"             median = {np.median(bl_mse_all):.6e}")
    print(f"             p95    = {np.quantile(bl_mse_all, 0.95):.6e}")

    if mdl_mse_all:
        mdl_mse_all = np.asarray(mdl_mse_all)
        print(f"\nTrained model:")
        print(f"  masked-MSE  mean = {mdl_mse_all.mean():.6e}")
        print(f"             median = {np.median(mdl_mse_all):.6e}")
        print(f"             p95    = {np.quantile(mdl_mse_all, 0.95):.6e}")
        ratio = bl_mse_all.mean() / mdl_mse_all.mean()
        print(f"\nBaseline / Model masked-MSE ratio (mean): {ratio:.2f}×")

    if args.plot:
        _plot_panels(plot_cache, args.plot)
        print(f"\nWrote {args.plot}")


def _plot_panels(samples, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, samples):
        q      = s["q"]
        m      = s["mask"]
        target = s["target"]
        bl     = s["baseline"]
        mdl    = s["model"]
        q_min  = q[m > 0.5].min()

        ax.axvspan(q.min(), q_min, color="khaki", alpha=0.4, label="masked")
        ax.plot(q, target, "k-",  lw=1.0, label="target")
        ax.plot(q, bl,    "C0-",  lw=1.2, label="baseline")
        if mdl is not None:
            ax.plot(q, mdl, "C3-", lw=1.0, alpha=0.85, label="model")
        ax.set_title(s["label"], fontsize=9)
        ax.set_ylabel(r"$\Delta S_0^{norm}$")
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$q$ (Å$^{-1}$)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data/trajectory_data_cap200")
    p.add_argument("--checkpoint", default=None, help="Optional model checkpoint .pt for side-by-side comparison")
    p.add_argument("--n-channels", type=int, default=128, help="Must match the checkpoint's training config")
    p.add_argument("--baseline",   choices=["poly", "nsk"], default="nsk",
                   help="Which baseline to evaluate. 'nsk' = spherical-Bessel "
                        "inversion + boundary-matched extrapolation (Wang et al. 2026).")
    # Polynomial baseline params
    p.add_argument("--n-terms",    type=int, default=3, help="(poly) number of polynomial terms")
    p.add_argument("--parity",     choices=["even", "all"], default="even",
                   help="(poly) 'even' → q²,q⁴,q⁶,...  'all' → q,q²,q³,...")
    p.add_argument("--window",     type=float, default=1.5, help="(poly) fit window above q_min, Å⁻¹")
    # NSK baseline params
    p.add_argument("--r-max",      type=float, default=10.0, help="(nsk) upper R for basis grid, Å")
    p.add_argument("--dr-basis",   type=float, default=0.1,  help="(nsk) basis grid spacing, Å")
    p.add_argument("--eps",        type=float, default=1e-3, help="(nsk) Tikhonov weight for initial inversion")
    p.add_argument("--lambda-R",   type=float, default=1.0,  help="(nsk) weight on ‖w_ext − w‖²")
    p.add_argument("--lambda-cont", type=float, default=10.0, help="(nsk) weight on (value, ∂_q, ∂²_q) continuity")
    p.add_argument("--q-min-lo",   type=float, default=0.5)
    p.add_argument("--q-min-hi",   type=float, default=1.0)
    p.add_argument("--R-margin",   type=float, default=1.2)
    p.add_argument("--R-cap",     type=float, default=10.0, help="Hard cap on R_max (Å) — must match training config")
    p.add_argument("--max-samples", type=int, default=500, help="Limit val samples for quick checks")
    p.add_argument("--n-plot",     type=int, default=6, help="Number of sample panels to plot")
    p.add_argument("--plot",       default=None, help="Output path for comparison plot (PNG)")
    p.add_argument("--seed",       type=int, default=0, help="Seed for dataset q_min sampling")
    p.add_argument("--plot-seed",  type=int, default=0, help="Seed for selecting which val samples to plot")
    args = p.parse_args()
    run_eval(args)
