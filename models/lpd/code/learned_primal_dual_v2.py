"""
learned_primal_dual_v2.py — Physics-consistent Learned Primal-Dual for pair-density recovery.

Maps the processed scattering signal of one frame to its NORMALISED pair density

    dS0(q), W(q)  →  p(R) = pd(R) / Σ_{a<b} Z_a Z_b          (p ≥ 0, Σ_R p = 1)

where dS0 = (S0 − Σf²)/fe · W is processed exactly as in processing/analyse_mol_v2.py and
W(q) is the analysis (Lorch) window over the measured q-range (zero below a beamstop q_min).

Changes relative to learned_primal_dual.py (v1), each motivated by the baseline evaluation
(analysis/pd_recovery/eval_pd_recovery.py, results in analysis/pd_recovery/results/lpd_v2/):

  1. Forward operator = Debye equation, not the DBT transpose:
         dS0(q_i) ≈ W(q_i) · Σ_m 2 j0(q_i · Rc_m) · p_m        (point-atom fe approximation)
     v1's T(f) = f @ A_dbt left 50–80 % of dS0 unexplained even for the true PD.
  2. The window W is an input, so data consistency is only enforced at measured q. One model
     handles any beamstop q_min (v1 collapsed at q_min = 0.8 Å⁻¹: R² < 0).
  3. Warm start = Tikhonov solution for each sample's own window, f0 = T*(T T* + λI)⁻¹ g
     (alone: R² ≈ 0.91 at 0.1 Å on clean data). v1's warm start was ~10⁻⁶ of the target.
  4. Output is non-negative and sums to one: p̂ = relu(f¹) / Σ relu(f¹). dS0 cannot determine
     Σ Z_a Z_b (it is divided by fe ∝ ΣZZ), and positivity is what keeps NSK-ℓ1 robust to the
     beamstop. v1 put 9 % of its weight beyond the molecule's longest pair distance.
  5. Coordinate channels (q, R, observed-q mask) and dilated R-space convolutions give the
     CNNs position information and a receptive field of ±1.6 Å per iteration (v1: ±0.15 Å,
     and no way to apply a q- or R-dependent correction).

Units inside the network: f = κ·p (κ = `kappa`, default 100), and both operators are scaled by
1/σ_max(F0) so that ‖T‖ = ‖T*‖ ≤ 1 — required for stability of the unrolled iterations
(Adler & Öktem, IEEE TMI 2018). The measured data are rescaled to the same units:
g = κ · (dS0 / max W) / σ_max(F0).

Usage
-----
    from models.lpd.code.learned_primal_dual_v2 import LearnedPrimalDualV2, process_scattering
    model = LearnedPrimalDualV2(q, R_grid)                       # q (nq,), R_grid (nR,) numpy
    dS0, W = process_scattering(S0, self_scatter, fe, q_t, qmin) # batch processing on device
    p_hat, f_raw = model(dS0, W)                                 # (B, nR) each
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import spherical_jn


# ═════════════════════════════════════════════════════════════════════════════
# Measurement processing (batched, on device)
# ═════════════════════════════════════════════════════════════════════════════

def lorch_window_batch(q: torch.Tensor, qmin: torch.Tensor) -> torch.Tensor:
    """
    Lorch window over the observed points q ≥ q_min, normalised to unit sum; one row per sample.
    Identical to analyse_mol_v2.lorch() for q_min = 0 and to eval_pd_recovery.lorch_window().

    q : (nq,) ascending,  qmin : (B,)  →  W : (B, nq)
    """
    nq = q.shape[0]
    obs = q[None, :] >= qmin[:, None]                          # (B, nq)
    i0 = (~obs).sum(-1, keepdim=True)                          # index of first observed point
    n_obs = (nq - i0).clamp(min=2).to(q.dtype)
    k = (torch.arange(nq, device=q.device)[None, :] - i0).clamp(min=0).to(q.dtype)
    W = torch.sin(math.pi * k / (n_obs - 1)) * obs
    return W / W.sum(-1, keepdim=True)


def process_scattering(S0, self_scatter, fe, q, qmin):
    """
    dS0 = (S0 − Σf²)/fe · W_qmin  — the pipeline's normalisation plus a beamstop at q_min.

    S0 : (B, nq);  self_scatter, fe : (B, nq) or (nq,);  q : (nq,);  qmin : (B,)
    Returns dS0 (B, nq) and the window W (B, nq).
    """
    W = lorch_window_batch(q, qmin)
    return (S0 - self_scatter) / fe * W, W


# ═════════════════════════════════════════════════════════════════════════════
# Operator
# ═════════════════════════════════════════════════════════════════════════════

def build_debye_operator(q: np.ndarray, R_grid: np.ndarray) -> torch.Tensor:
    """F0[i, m] = 2 · j0(q_i · Rc_m), Rc = centres of the pd_true histogram bins.  (nq, nR)"""
    q = np.asarray(q, dtype=np.float64)
    R = np.asarray(R_grid, dtype=np.float64)
    Rc = R + (R[1] - R[0]) / 2          # pd_true uses floor binning: bin m covers [R_m, R_m + ΔR)
    return torch.tensor(2 * spherical_jn(0, np.outer(q, Rc)), dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Per-iteration networks
# ═════════════════════════════════════════════════════════════════════════════

class DualNetV2(nn.Module):
    """
    q-space update  h ← h + Λᵈ(h, T f², g, T f² − g, observed mask, q coordinate).
    Last layer zero-initialised, so each iteration starts as the identity.
    """
    N_EXTRA = 5

    def __init__(self, n_dual: int, n_channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_dual + self.N_EXTRA, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_dual, kernel_size, padding=pad),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return h + self.net(torch.cat([h, extra], dim=1))


class DilatedBlock(nn.Module):
    def __init__(self, n_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.PReLU(n_channels)
        self.mix = nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(self.act(self.conv(x)))


class PrimalNetV2(nn.Module):
    """
    R-space update  f ← f + Λᵖ(f, T* h¹, warm start f0, R coordinate).
    Dilated residual stack: receptive field ±(2 + Σ dilations) bins = ±65 bins ≈ ±1.6 Å.
    Last layer zero-initialised.
    """
    N_EXTRA = 3

    def __init__(self, n_primal: int, n_channels: int, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(n_primal + self.N_EXTRA, n_channels, 3, padding=1)
        self.act = nn.PReLU(n_channels)
        self.blocks = nn.Sequential(*[DilatedBlock(n_channels, d) for d in dilations])
        self.out = nn.Conv1d(n_channels, n_primal, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        x = self.act(self.inp(torch.cat([f, extra], dim=1)))
        return f + self.out(self.blocks(x))


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

class LearnedPrimalDualV2(nn.Module):
    """
    Parameters
    ----------
    q, R_grid    : (nq,), (nR,) grids of the data [Å⁻¹], [Å] (numpy or tensor)
    n_primal     : R-space memory channels
    n_dual       : q-space memory channels
    n_iter       : unrolled iterations (separate weights per iteration)
    n_channels   : hidden channels of the per-iteration CNNs
    kappa        : internal scale, f = κ·p (keeps the CNN inputs O(0.1–10))
    tikhonov_rel : λ of the warm start, relative to σ_max(F0)² (baseline tuning: 1e-6 clean)
    """

    def __init__(self, q, R_grid, n_primal: int = 5, n_dual: int = 5, n_iter: int = 10,
                 n_channels: int = 32, kappa: float = 100.0, tikhonov_rel: float = 1e-6):
        super().__init__()
        q = np.asarray(q, dtype=np.float64)
        R = np.asarray(R_grid, dtype=np.float64)
        F0 = build_debye_operator(q, R)
        s = float(torch.linalg.matrix_norm(F0.double(), ord=2))
        F0n = (F0.double() / s)
        self.register_buffer("F0", F0n.float())                    # (nq, nR), ‖F0‖ = 1
        self.register_buffer("K", F0n @ F0n.T)                     # (nq, nq) float64
        self.register_buffer("q_coord", torch.tensor(q / q.max(), dtype=torch.float32))
        self.register_buffer("R_coord", torch.tensor(R / R.max(), dtype=torch.float32))
        self.sigma_F0 = s
        self.kappa = float(kappa)
        self.tikhonov_rel = float(tikhonov_rel)
        self.n_primal, self.n_dual, self.n_iter = n_primal, n_dual, n_iter
        self.config = dict(n_primal=n_primal, n_dual=n_dual, n_iter=n_iter, n_channels=n_channels,
                           kappa=kappa, tikhonov_rel=tikhonov_rel)

        self.dual_nets = nn.ModuleList([DualNetV2(n_dual, n_channels) for _ in range(n_iter)])
        self.primal_nets = nn.ModuleList([PrimalNetV2(n_primal, n_channels) for _ in range(n_iter)])

    # ── operators (Wt = window normalised to max 1; zero outside the measured range) ──
    def T(self, f: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
        """(B, nR) → (B, nq)"""
        return (f @ self.F0.T) * Wt

    def T_star(self, h: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
        """(B, nq) → (B, nR); exact adjoint of T"""
        return (h * Wt) @ self.F0

    def scale_data(self, dS0: torch.Tensor, W: torch.Tensor):
        """Measured (dS0, W) → (g in f-units, Wt). T(κ·p) ≈ g for the true p."""
        Wmax = W.amax(-1, keepdim=True)
        Wt = W / Wmax
        g = self.kappa * (dS0 / Wmax) / self.sigma_F0
        return g, Wt

    def warm_start(self, g: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
        """Tikhonov f0 = T*(T T* + λI)⁻¹ g, solved per sample in q-space (float64)."""
        Wd = Wt.double()
        M = Wd[:, :, None] * self.K[None] * Wd[:, None, :]
        M = M + self.tikhonov_rel * torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        sol = torch.linalg.solve(M, g.double()).to(g.dtype)
        return self.T_star(sol, Wt)

    def data_residual(self, f: torch.Tensor, g: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
        """Mean squared data misfit over measured q, relative to the data power. (B,)"""
        obs = (Wt > 0).to(g.dtype)
        r2 = ((self.T(f, Wt) - g) ** 2 * obs).sum(-1)
        return r2 / ((g ** 2).sum(-1) + 1e-12)

    def forward(self, dS0: torch.Tensor, W: torch.Tensor):
        """
        dS0 : (B, nq) processed signal;  W : (B, nq) the window used to process it.
        Returns p_hat (B, nR) — normalised PD prediction — and f_raw (B, nR), the
        pre-activation primal channel in f-units (used for the data-consistency loss).
        """
        B = dS0.shape[0]
        nq, nR = self.F0.shape
        g, Wt = self.scale_data(dS0, W)
        obs = (Wt > 0).to(g.dtype)
        f0 = self.warm_start(g, Wt)

        f = dS0.new_zeros(B, self.n_primal, nR)
        f[:, 0] = f0
        f[:, 1] = f0
        h = dS0.new_zeros(B, self.n_dual, nq)
        qc = self.q_coord.expand(B, nq)
        Rc = self.R_coord.expand(B, nR)

        for i in range(self.n_iter):
            Tf = self.T(f[:, 1], Wt)
            h = self.dual_nets[i](h, torch.stack([Tf, g, Tf - g, obs, qc], dim=1))
            Th = self.T_star(h[:, 0], Wt)
            f = self.primal_nets[i](f, torch.stack([Th, f0, Rc], dim=1))

        f_raw = f[:, 0]
        p = F.relu(f_raw)
        p_hat = p / (p.sum(-1, keepdim=True) + 1e-8)
        return p_hat, f_raw


# ═════════════════════════════════════════════════════════════════════════════
# Losses / metrics (torch)
# ═════════════════════════════════════════════════════════════════════════════

def w1_loss(p_hat: torch.Tensor, p: torch.Tensor, dR: float) -> torch.Tensor:
    """1-D Wasserstein distance per sample, Σ_R |CDF(p̂) − CDF(p)|·ΔR  [Å].  (B,)
    For unit-mass p̂ the tail beyond the molecule charges stray weight by its distance."""
    return torch.cumsum(p_hat - p, dim=-1).abs().sum(-1) * dR


def gaussian_blur(x: torch.Tensor, sigma: float, dR: float) -> torch.Tensor:
    """Zero-padded Gaussian blur along R (std σ Å, truncated at 4σ, like scipy's default)."""
    if sigma <= 0:
        return x
    s = sigma / dR
    r = int(4 * s + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / s) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    return F.conv1d(x.unsqueeze(1), k, padding=r).squeeze(1)


def r2_blurred(p_hat: torch.Tensor, p: torch.Tensor, sigma: float, dR: float) -> torch.Tensor:
    """R² after blurring prediction and truth by σ (same metric as eval_pd_recovery).  (B,)"""
    a, b = gaussian_blur(p_hat, sigma, dR), gaussian_blur(p, sigma, dR)
    sst = ((b - b.mean(-1, keepdim=True)) ** 2).sum(-1)
    return 1 - ((a - b) ** 2).sum(-1) / sst.clamp_min(1e-30)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "data/trajectory_data_cap200").glob("mol_*_pd.npz"))
    d = np.load(files[0])
    q, R = d["q"], d["R_grid"]
    model = LearnedPrimalDualV2(q, R)
    print(f"σ_max(F0) = {model.sigma_F0:.2f}   parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    S0 = torch.tensor(d["S0"][0, :4]); ss = torch.tensor(d["self_scatter"]); fe = torch.tensor(d["fe"])
    pd = torch.tensor(d["pd_true"][0, :4]); p = pd / pd.sum(-1, keepdim=True)
    qt = torch.tensor(q)
    for qmin in (0.0, 0.8):
        dS0, W = process_scattering(S0, ss, fe, qt, torch.full((4,), qmin))
        p_hat, f_raw = model(dS0, W)
        g, Wt = model.scale_data(dS0, W)
        print(f"q_min={qmin}: W1(init) = {w1_loss(p_hat, p, float(R[1]-R[0])).mean():.4f} Å   "
              f"data residual(true p) = {model.data_residual(model.kappa * p, g, Wt).mean():.3f}")
    sys.exit(0)
