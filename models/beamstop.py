"""
beamstop.py — Model for predicting the missing low-q scattering signal
blocked by the beamstop in TRXSS experiments.

Task: ΔS0_norm(q ≥ q_min) → ΔS0_norm(all q)

where ΔS0_norm = (S0(t) − S0_ref) / fe(q) is the normalized differential
scattering signal and S0_ref = S0[ti, 0] is the first frame of the same
trajectory, analogous to the pre-pump ground-state measurement in TRXSS.

Physics constraints enforced by construction:
  1. ΔS0_norm(q=0) = 0  — because S0(q=0) = (ΣZ_i)² is geometry-independent,
                           enforced by zero-centering the R-space intermediate
  2. Smoothness            — enforced via second-derivative loss penalty in training

Architecture (physics-guided encoder-decoder):
  Encoder : Conv1d in q-space → linear interpolation to nR → Conv1d in R-space
            → ΔPD_eff (batch, nR)   [learned]
  Decoder : zero-center(ΔPD_eff) @ B_sinc → ΔS0_norm_pred (batch, nq)  [fixed physics]

The forward operator B_sinc (nR × nq) is the plain sinc/Debye matrix:
  B_sinc[j, i] = 2 · sin(q[i] · R[j]) / (q[i] · R[j])
No Lorch window — ΔS0_norm is not windowed.

Usage
-----
    from models.beamstop import BeamstopModel, build_sinc_operator
    B_sinc = build_sinc_operator(q_np, R_grid_np)
    model  = BeamstopModel(B_sinc)
    pred   = model(delta_S0_masked, mask)   # (batch, nq)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# Sinc / Debye forward operator  (no Lorch window)
# ═════════════════════════════════════════════════════════════════════════════

def build_sinc_operator(q: np.ndarray, R_grid: np.ndarray, verbose: bool = True) -> torch.Tensor:
    """
    Build the plain sinc/Debye forward operator B_sinc of shape (nR, nq).

    B_sinc[j, i] = 2 · sin(q[i] · R[j]) / (q[i] · R[j])

    Satisfies: ΔS0_norm(q) ≈ ΔPD_eff @ B_sinc  (approximate, ignoring form-factor
    q-dependence which is weak at low q where the model operates).

    The factor 2 reflects the Debye double-counting convention:
        (S0 − Σf²) / fe = 2 Σ_{k<l} (f_k f_l / fe) sinc(q R_kl)

    At q ≈ 0: B_sinc[:, 0] → 2 for all R, so zero-centering ΔPD_eff before
    projecting guarantees ΔS0_norm(q=0) = 2 · sum(ΔPD_eff) = 0.

    Parameters
    ----------
    q      : (nq,) q grid in Å⁻¹
    R_grid : (nR,) R grid in Å

    Returns
    -------
    B_sinc : torch.Tensor (nR, nq) float32
    """
    qR = np.outer(R_grid, q)
    B  = np.where(np.abs(qR) < 1e-10, 2.0, 2.0 * np.sin(qR) / qR)

    if verbose:
        print(f'B_sinc : {B.shape}  ({B.nbytes / 1e6:.1f} MB)')

    return torch.tensor(B, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Encoder — q-space signal → R-space representation
# ═════════════════════════════════════════════════════════════════════════════

class BeamstopEncoder(nn.Module):
    """
    Maps masked ΔS0_norm (nq,) to ΔPD_eff (nR,) via:
      1. Conv1d feature extraction in q-space   (length nq preserved)
      2. Linear interpolation from nq to nR     (no learned params)
      3. Conv1d refinement in R-space           (length nR preserved)

    The mask is concatenated as a second input channel so the encoder knows
    which q values are observed vs. missing.
    """

    def __init__(self, nq: int, nR: int, n_channels: int = 32, kernel_size: int = 5):
        super().__init__()
        self.nR = nR
        pad = kernel_size // 2

        self.q_conv = nn.Sequential(
            nn.Conv1d(2, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, 1, kernel_size, padding=pad),
        )

        self.R_conv = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, 1, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (batch, nq)  — ΔS0_norm with q < q_min zeroed
        mask : (batch, nq)  — 1.0 where observed, 0.0 where masked
        Returns: (batch, nR)
        """
        h = torch.stack([x, mask], dim=1)                                          # (batch, 2, nq)
        h = self.q_conv(h)                                                          # (batch, 1, nq)
        h = F.interpolate(h, size=self.nR, mode='linear', align_corners=True)      # (batch, 1, nR)
        h = self.R_conv(h)                                                          # (batch, 1, nR)
        return h.squeeze(1)                                                         # (batch, nR)


# ═════════════════════════════════════════════════════════════════════════════
# BeamstopModel
# ═════════════════════════════════════════════════════════════════════════════

class BeamstopModel(nn.Module):
    """
    Predicts the missing low-q region of ΔS0_norm blocked by the beamstop.

    Parameters
    ----------
    B_sinc     : (nR, nq) sinc/Debye forward operator (from build_sinc_operator)
    n_channels : hidden channels in encoder CNNs  (default: 32)
    kernel_size: Conv1d kernel size               (default: 5)
    """

    def __init__(self, B_sinc: torch.Tensor, n_channels: int = 32, kernel_size: int = 5):
        super().__init__()
        self.register_buffer('B_sinc', B_sinc.float())  # (nR, nq) — fixed
        nR, nq = B_sinc.shape
        self.encoder = BeamstopEncoder(nq, nR, n_channels, kernel_size)

    def forward(self, delta_S0_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        delta_S0_masked : (batch, nq)  — ΔS0_norm with q < q_min set to 0
        mask            : (batch, nq)  — 1.0 where observed, 0.0 where masked

        Returns
        -------
        delta_S0_pred : (batch, nq)  — predicted ΔS0_norm at all q
        """
        delta_pd = self.encoder(delta_S0_masked, mask)                 # (batch, nR)

        # Enforce ΔS0(q=0) = 0: at q=0, B_sinc[:,0] ≈ 2 for all R, so
        # ΔS0(0) ∝ 2·sum(ΔPD_eff). Zero-centering sets sum=0, hence ΔS0(0)=0.
        delta_pd = delta_pd - delta_pd.mean(dim=-1, keepdim=True)

        return delta_pd @ self.B_sinc                                   # (batch, nq)


# ═════════════════════════════════════════════════════════════════════════════
# Smoothness loss (imported by training script)
# ═════════════════════════════════════════════════════════════════════════════

def smoothness_loss(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Second-derivative penalty applied only to the predicted (masked) low-q region.

    x    : (batch, nq)  — predicted ΔS0_norm
    mask : (batch, nq)  — 1=observed, 0=predicted
    """
    d2          = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]       # (batch, nq-2)
    pred_region = (mask[:, 1:-1] == 0).float()
    return (d2.pow(2) * pred_region).sum() / (pred_region.sum() + 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# Sanity check
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import glob

    npz_files = sorted(glob.glob('data/trajectory_data*/**/*.npz', recursive=True))

    if npz_files:
        ref  = np.load(npz_files[0])
        q_np = ref['q']
        R_np = ref['R_grid']
        print(f'Building B_sinc from {npz_files[0]}')
    else:
        print('No data found — using synthetic grids')
        q_np = np.arange(1e-4, 10.1, 0.1, dtype=np.float32)
        R_np = np.linspace(0.025, 31.4, 1258, dtype=np.float32)

    B_sinc = build_sinc_operator(q_np, R_np)
    model  = BeamstopModel(B_sinc, n_channels=32, kernel_size=5)

    nq    = len(q_np)
    batch = 4
    dq    = float(q_np[1] - q_np[0])

    # Simulate beamstop at q_min = 0.75 Å⁻¹ (midpoint of [0.5, 1.0] training range)
    q_min     = 0.75
    q_min_idx = int(q_min / dq)
    mask      = torch.ones(batch, nq)
    mask[:, :q_min_idx] = 0.0

    delta_S0        = torch.randn(batch, nq) * 0.1
    delta_S0_masked = delta_S0 * mask

    pred = model(delta_S0_masked, mask)

    print(f'Input  delta_S0_masked : {tuple(delta_S0_masked.shape)}')
    print(f'Output delta_S0_pred   : {tuple(pred.shape)}')
    print(f'pred[:, 0] (q≈0)       : {pred[:, 0].detach().tolist()}  ← should be ~0')
    print(f'Parameters             : {sum(p.numel() for p in model.parameters()):,}')
    print(f'B_sinc                 : {tuple(B_sinc.shape)}  ({B_sinc.nbytes / 1e6:.1f} MB)')
