"""
nsk.py — Natural-scattering-kernel (NSK) ℓ1 deconvolution as published in
A. Natan, Phys. Rev. A 107, 023105 (2023), Secs. II B–E and IV.

Steps (paper equation numbers):
  1. Inversion (Eqs. 5–7). PD0 = DBT(x) on the DBT's own real-space grid R_m = j0_m / q_end, where
     j0_m is the m-th root of j0 and q_end = (M−1)Δq. M = ceil(π / (ΔR Δq)) + 1 sets the grid spacing
     ΔR ≈ π/q_end — the dictionary resolution (0.157 Å in Figs. 2–4, ≈ 0.1 Å in Fig. 5).
  2. Dictionary (Eqs. 8–9, thin-sample limit j̃0 ≈ h·j0). D[:, m] = DBT(h(q) · j0(q R_m)): the
     distorted inversion of a δ pair density at R_m, for every grid point R_m.
  3. Deconvolution (Eqs. 10–11, ℓ1 case). w = argmin ½‖PD0 − D w‖² + ε‖w‖₁, with ε at the corner of the
     L-curve (Sec. II E, Fig. 2d: ‖w‖₁ vs residual norm on linear axes; located automatically as the
     knee of the low-residual part, see NSK.lcurve).
  4. Optional (Sec. IV B, Fig. 5): the stems w are blurred by the minimum separation distance δR
     (≈ 0.15 Å in Fig. 5) for display.

Conventions of this repository: x is the processed signal dS0 = (S0 − Σf²)/fe · h(q), and a pair
density p (fraction of pair weight at each distance) gives x = Σ_m 2 p_m h j0(q R_m), so
PD0 ≈ D (2p) and the pair density is p = w / 2. Dictionary columns are normalised to unit norm before
the ℓ1 problem is solved (as in the group's port of the author's code, processing/analyse_mol_v2
.decon_TVL1); the returned weights are in physical units. positive=True adds w ≥ 0 (absolute pair
densities); positive=False is the signed problem the paper solves for pair-density differences.

Solver: the exact lasso homotopy (LARS with the lasso modification; Efron et al., Ann. Stat. 32, 407
(2004)), in Gram form. It returns every breakpoint of the piecewise-linear solution path w(ε), so
the L-curve is evaluated on exact solutions (the paper used CVX at a grid of ε values). The ADMM in
decon_TVL1 with its 500-iteration cap does not converge on these highly correlated dictionaries
and returns a smoothed, non-sparse iterate instead (see admm_lasso below).

Usage
-----
    from models.nsk.code.nsk import NSK
    nsk = NSK(q, h, dR_dict=0.1)                    # dictionary for this q grid / window h(q)
    p, info = nsk.solve(x, positive=False)          # x: (n, nq); p: (n, K) at nsk.R; ε by L-curve
    p_fine = nsk.to_grid(p, R_fine)                 # onto another uniform grid (bin centres R_fine)

    # several ε for one frame from a single path (what solve does internally, per frame):
    y = nsk.transform(x)[0]
    lams, Z = nsk.path(y, positive=True)
    p_lc, eps_rel, nnz, curve = nsk.weights(y, lams, Z)            # L-curve corner
    p_1e3 = nsk.weights(y, lams, Z, lam_rel=1e-3)[0]                # fixed ε = 1e-3 · ε_max
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import spherical_jn

from models.lpd.code.learned_primal_dual import _dbt


# ═════════════════════════════════════════════════════════════════════════════
# Exact lasso path
# ═════════════════════════════════════════════════════════════════════════════

def lasso_path(G: np.ndarray, c0: np.ndarray, positive: bool, lam_min: float,
               max_steps: int = 5000, ridge: float = 1e-11):
    """
    Solution path of  min_z ½‖y − A z‖² + λ‖z‖₁  (z ≥ 0 if positive) from λ_max down to lam_min.

    G = AᵀA (K, K) and c0 = Aᵀy (K,) — the problem only enters through them.
    Returns lams (n_bp,) decreasing and Z (n_bp, K): the solutions at the breakpoints; between two
    breakpoints the solution is linear in λ (see path_at).
    ridge (relative to mean diag G) regularises the active-set solves when columns are nearly
    collinear; it shifts the path by a negligible amount.
    """
    K = len(c0)
    z = np.zeros(K)
    c = c0.copy()
    j = int(np.argmax(c if positive else np.abs(c)))
    lam = float(c[j] if positive else abs(c[j]))
    if lam <= lam_min:
        return np.array([max(lam, lam_min)]), np.zeros((1, K))
    tol = 1e-12 * lam
    eps = ridge * float(np.trace(G)) / K
    active = [j]
    sign = np.zeros(K)
    sign[j] = 1.0 if positive else float(np.sign(c[j]))
    lams, Zs = [lam], [z.copy()]
    for _ in range(max_steps):
        A = np.array(active)
        try:
            d = np.linalg.solve(G[np.ix_(A, A)] + eps * np.eye(len(A)), sign[A])
        except np.linalg.LinAlgError:
            break
        a = G[:, A] @ d                       # dc/dγ = −a; on the active set a = sign → |c_A| = λ − γ
        inactive = np.ones(K, bool)
        inactive[A] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            cands = [((lam - c) / (1 - a), 1.0)]                 # c_j reaches +level
            if not positive:
                cands.append(((lam + c) / (1 + a), -1.0))        # c_j reaches −level
            g_add, j_add, s_add = np.inf, -1, 0.0
            for g, s in cands:
                g = np.where(inactive & (g > tol), g, np.inf)
                jj = int(np.argmin(g))
                if g[jj] < g_add:
                    g_add, j_add, s_add = float(g[jj]), jj, s
            gd = np.where((-z[A] / d) > tol, -z[A] / d, np.inf)  # an active coefficient hits 0
        kd = int(np.argmin(gd))
        g_drop = float(gd[kd])
        g_end = lam - lam_min
        gam = min(g_add, g_drop, g_end)
        z[A] += gam * d
        lam -= gam
        c = c0 - G @ z
        if gam == g_end:
            lams.append(lam)
            Zs.append(z.copy())
            break
        if g_drop <= g_add:
            jd = int(A[kd])
            z[jd] = 0.0
            active.remove(jd)
            sign[jd] = 0.0
        else:
            active.append(j_add)
            sign[j_add] = s_add
        lams.append(lam)
        Zs.append(z.copy())
        if not active:
            break
    return np.array(lams), np.array(Zs)


def path_at(lams: np.ndarray, Z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Solutions at arbitrary λ (array, within the path range) by linear interpolation in λ."""
    lam = np.atleast_1d(lam)
    lr = lams[::-1]                            # increasing
    k = np.clip(np.searchsorted(lr, lam) - 1, 0, len(lr) - 2) if len(lr) > 1 else np.zeros(len(lam), int)
    if len(lr) == 1:
        return np.repeat(Z[:1], len(lam), 0)
    lo, hi = lr[k], lr[k + 1]                  # bracketing breakpoints (λ_lo < λ ≤ λ_hi)
    t = np.where(hi > lo, (lam - lo) / (hi - lo), 0.0)[:, None]
    Zr = Z[::-1]
    return (1 - t) * Zr[k] + t * Zr[k + 1]


# ═════════════════════════════════════════════════════════════════════════════
# NSK dictionary and the paper's deconvolution
# ═════════════════════════════════════════════════════════════════════════════

class NSK:
    """
    Dictionary for one measurement: q grid (uniform, starting at ~0, as the DBT requires), window
    h(q) (zero outside the measured range — beamstop / detector edge, the analysis window inside)
    and dictionary resolution dR_dict [Å].

    Attributes: M (DBT order), R (K,) dictionary positions = the DBT real-space grid (minus points
    whose kernel the inversion cannot see), D (M−1, K) dictionary, col (K,) its column norms.
    """

    def __init__(self, q: np.ndarray, h: np.ndarray, dR_dict: float):
        self.q = np.asarray(q, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        dq = float(self.q[1] - self.q[0])
        self.M = int(np.ceil(np.pi / (dR_dict * dq))) + 1
        _, self.R = _dbt(0, np.zeros((len(self.q), 1)), self.q, self.M)
        self.dR = float(self.R[1] - self.R[0])
        D = _dbt(0, self.h[:, None] * spherical_jn(0, np.outer(self.q, self.R)), self.q, self.M)[0]
        col = np.linalg.norm(D, axis=0)
        # Drop kernels the inversion cannot see (column norm ≈ 0: the last DBT grid point, ≈ 1e-6 of
        # the others). Normalising such a column would turn round-off into arbitrarily large weights.
        keep = col > 1e-3 * np.median(col)
        self.R_dbt = self.R                    # grid of transform(): PD0 lives here (M−1 points)
        self.R, D, col = self.R[keep], D[:, keep], col[keep]
        self.D, self.col = D, col
        self.Dn = D / col
        self.G = self.Dn.T @ self.Dn

    def transform(self, x: np.ndarray) -> np.ndarray:
        """PD0 = DBT(x) on the dictionary grid. x: (n, nq) → (n, K)."""
        return _dbt(0, np.atleast_2d(x).T.astype(np.float64), self.q, self.M)[0].T

    def path(self, y: np.ndarray, positive: bool, lam_min_rel: float = 1e-6):
        """Exact path for one PD0 vector y (K,). Returns lams, Z (normalised-column units)."""
        c0 = self.Dn.T @ y
        lam_max = float(np.max(c0) if positive else np.max(np.abs(c0)))
        return lasso_path(self.G, c0, positive, lam_min_rel * max(lam_max, 1e-300))

    def lcurve(self, y: np.ndarray, lams: np.ndarray, Z: np.ndarray, n_grid: int = 300,
               res_window: float = 0.05):
        """
        L-curve corner, as drawn in the paper (Fig. 2d): solution norm ‖w‖₁ versus residual norm
        ‖PD0 − Dw‖ on LINEAR axes, over the low-residual part of the path where the paper shows it
        (relative residual ≤ res_window). There the curve is L-shaped: first the residual falls at
        nearly constant ‖w‖₁, then ‖w‖₁ grows at nearly zero residual. The corner is located as the
        knee — the point farthest from the chord between the window's end points, with both axes
        normalised to [0, 1] over the window. Exact solutions along the path are sampled on a
        log-spaced λ grid. (Maximum curvature on log–log axes, the other common recipe, picks a bend
        at much larger λ on noise-free data and misses sub-diffraction features in the paper's Fig. 2
        example — see analysis/pd_recovery/replicate_nsk_paper.py --case i2.)
        Returns (λ*, dict of the curve for plotting: lam, residual, norm, k = selected index,
        window = boolean mask of the points used).
        """
        hi, lo = lams[1] if len(lams) > 1 else lams[0], lams[-1]
        grid = np.exp(np.linspace(np.log(hi), np.log(lo), n_grid))
        Zg = path_at(lams, Z, grid)
        res = np.linalg.norm(y[None] - Zg @ self.Dn.T, axis=1)
        nrm = np.abs(Zg / self.col).sum(1)
        win = res <= res_window * max(np.linalg.norm(y), 1e-300)
        if win.sum() < 10:                     # residual never gets that small: use the whole path
            win = np.ones(n_grid, bool)
        idx = np.nonzero(win)[0]
        xr, yr = res[idx], nrm[idx]
        xn = (xr - xr.min()) / max(np.ptp(xr), 1e-300)
        yn = (yr - yr.min()) / max(np.ptp(yr), 1e-300)
        # chord from the first (largest-residual) to the last point of the window; knee = max distance
        p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-300)
        dist = np.abs((xn - p0[0]) * u[1] - (yn - p0[1]) * u[0])
        k = int(idx[np.argmax(dist)])
        return float(grid[k]), dict(lam=grid, residual=res, norm=nrm, k=k, window=win)

    def weights(self, y: np.ndarray, lams: np.ndarray, Z: np.ndarray, lam_rel: float | None = None):
        """
        Pick ε on an already computed path (self.path(y, ...)) of the PD0 vector y: at the L-curve
        corner if lam_rel is None, else ε = lam_rel · ε_max (ε_max = max|Dnᵀy|, where the path starts;
        lam_rel must lie inside the computed path, i.e. ≥ its lam_min_rel).
        Returns p (K,) — pair density at self.R, p = w/2, physical units — ε/ε_max, the number of
        non-zero stems, and the L-curve dict (None when lam_rel is given).
        """
        if lam_rel is None:
            lam, cur = self.lcurve(y, lams, Z)
        else:
            lam, cur = lam_rel * lams[0], None
        z = path_at(lams, Z, np.array([lam]))[0]
        nnz = int((np.abs(z) > 1e-9 * max(np.abs(z).max(), 1e-300)).sum())
        return 0.5 * z / self.col, lam / lams[0], nnz, cur

    def solve(self, x: np.ndarray, positive: bool, lam_rel: float | None = None,
              lam_min_rel: float = 1e-6, return_curves: bool = False):
        """
        Paper NSK-ℓ1 for each row of x (n, nq). ε by the L-curve unless lam_rel is given
        (ε = lam_rel · max|Dnᵀ y|, the convention of the harness's ADMM baseline).
        Returns p (n, K) — pair density at self.R, p = w/2 — and info (per-frame ε, sparsity, curves).
        """
        Y = self.transform(x)
        P = np.zeros((len(Y), len(self.R)))
        info = dict(lam=[], nnz=[], curves=[])
        for i, y in enumerate(Y):
            lams, Z = self.path(y, positive, lam_min_rel if lam_rel is None else min(lam_min_rel, lam_rel))
            P[i], lr, nnz, cur = self.weights(y, lams, Z, lam_rel)
            info["lam"].append(lr)
            info["nnz"].append(nnz)
            if return_curves:
                info["curves"].append(cur)
        return P, info

    def to_grid(self, p: np.ndarray, Rc: np.ndarray) -> np.ndarray:
        """
        Stems at self.R → histogram on another uniform grid with bin centres Rc, splitting each stem
        linearly between the two nearest centres (conserves weight and its first moment, so W1 is
        unaffected by the regridding). p: (n, K) → (n, len(Rc)).
        """
        p = np.atleast_2d(p)
        d = float(Rc[1] - Rc[0])
        u = (self.R - Rc[0]) / d
        i0 = np.floor(u).astype(int)
        f = u - i0
        out = np.zeros((p.shape[0], len(Rc)))
        for idx, wgt in ((i0, 1 - f), (i0 + 1, f)):
            ok = (idx >= 0) & (idx < len(Rc))
            np.add.at(out, (slice(None), idx[ok]), p[:, ok] * wgt[ok])
        return out


def admm_lasso(Y: np.ndarray, Dn: np.ndarray, lam_frac: float, nonneg: bool, max_iter: int = 500,
               rho: float = 1.0, alpha: float = 1.25, mu: float = 10.0, tau_rho: float = 2.0,
               rho_update: int = 10, tol_abs: float = 1e-6, tol_rel: float = 1e-3) -> np.ndarray:
    """
    The pipeline's solver for comparison: ADMM for the same lasso, with the settings of
    processing/analyse_mol_v2.decon_TVL1 as called in run_pipeline (500 iterations, tol_rel 1e-3,
    adaptive ρ, over-relaxation α) — identical to the former nsk_l1 baseline of analysis/pd_recovery/eval_pd_recovery.py (removed).
    Y: (K, n) PD0 columns; Dn: column-normalised dictionary; λ = lam_frac · max|Dnᵀy| per column.
    Returns the last iterate z (K, n) in normalised-column units. With the highly correlated NSK
    dictionaries it stops long before convergence, returning a smooth, non-sparse iterate.
    """
    ev, V = np.linalg.eigh(Dn.T @ Dn)
    ev = np.clip(ev, 0, None)
    DtY = Dn.T @ Y
    lam = lam_frac * np.abs(DtY).max(axis=0, keepdims=True)
    X = np.zeros_like(DtY)
    Z = np.zeros_like(DtY)
    U = np.zeros_like(DtY)
    sqrt_n = np.sqrt(X.size)
    for k in range(1, max_iter + 1):
        Z_old = Z
        X = V @ ((V.T @ (DtY + rho * (Z - U))) / (ev + rho)[:, None])
        Xh = alpha * X + (1 - alpha) * Z_old
        Vv = Xh + U
        Z = np.maximum(Vv - lam / rho, 0) if nonneg else np.sign(Vv) * np.maximum(np.abs(Vv) - lam / rho, 0)
        U = U + Xh - Z
        r_norm, s_norm = np.linalg.norm(X - Z), rho * np.linalg.norm(Z - Z_old)
        if (r_norm < sqrt_n * tol_abs + tol_rel * max(np.linalg.norm(X), np.linalg.norm(Z))
                and s_norm < sqrt_n * tol_abs + tol_rel * rho * np.linalg.norm(U)):
            break
        if k % rho_update == 0:
            rho_new = rho * tau_rho if r_norm > mu * s_norm else rho / tau_rho if s_norm > mu * r_norm else rho
            U *= rho / rho_new
            rho = rho_new
    return Z


def blur_fwhm(p: np.ndarray, fwhm: float, dR: float) -> np.ndarray:
    """Gaussian blur with the given FWHM [Å] (area-preserving) along the last axis — the paper's
    minimum-separation smoothing of the ℓ1 stems (Fig. 5, δR ≈ 0.15 Å)."""
    return gaussian_filter1d(p, fwhm / (2 * np.sqrt(2 * np.log(2))) / dR, axis=-1, mode="constant")
