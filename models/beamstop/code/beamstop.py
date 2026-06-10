"""Beamstop reconstruction model for low-q TRXSS signals.

The model receives a normalized differential scattering curve with the low-q
beamstop region zeroed out and predicts the complete curve:

    ΔS0_norm(q observed) -> ΔS0_norm(all q)

where ΔS0_norm = (S0(t) - S0_ref) / fe(q), and S0_ref is the first frame of the
same trajectory.

The decoder is a fixed Debye/sinc projection from an R-space intermediate back
to q-space. Zero-centering that intermediate enforces ΔS0_norm(q=0) = 0, while
the truncated R grid acts as an architectural smoothness prior. The encoder can
optionally condition on scalar molecular features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# Sinc / Debye forward operator
# ═════════════════════════════════════════════════════════════════════════════

def build_sinc_operator(q: np.ndarray, R_grid: np.ndarray, verbose: bool = True) -> torch.Tensor:
    """
    Build the sinc/Debye forward operator B_sinc of shape (nR, nq).

    B_sinc[m, i] = 2 · sin(q[i] · R[m]) / (q[i] · R[m]) = 2 · j₀(q[i] · R[m])

    This is identical in structure to simulate_scattering's Debye sum:
        bessel = sinc(q * r_ab / π) = j₀(q * r_ab)
        ΔS0 = 2 · Σ_pairs fafb · bessel
    with R_grid bins playing the role of pair distances (uniform fafb weights
    because ΔS0_norm is already normalized by fe).

    Pre-building the matrix is more efficient than recomputing sinc on every
    forward pass — the (nR, nq) matrix is computed once and reused as a buffer.

    At q ≈ 0: B_sinc[:, 0] → 2 for all R. Zero-centering ΔPD before projecting
    guarantees ΔS0_norm(q=0) = 0 by construction (Σ_m delta_pd[m] = 0 →
    Σ_m delta_pd[m] · 2·j₀(0) = 0).

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
# FiLM — Feature-wise Linear Modulation
# ═════════════════════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """
    Per-channel affine conditioning: out = (1 + γ(cond)) * x + β(cond).

    Initialized to identity (all weights/biases zero -> scale=1, shift=0) so the
    conditioning has no effect at the start of training.

    Parameters
    ----------
    n_channels : channels in the feature map to be modulated
    d_cond     : dimension of the conditioning vector
    """
    def __init__(self, n_channels: int, d_cond: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_ch, L)  cond: (batch, d_cond)"""
        gb    = self.proj(cond)
        n_ch  = gb.shape[1] // 2
        gamma = gb[:, :n_ch].unsqueeze(-1) + 1.0
        beta  = gb[:, n_ch:].unsqueeze(-1)
        return gamma * x + beta


class MolFeatureEmbedding(nn.Module):
    """Map normalized scalar molecular features to the FiLM context vector."""

    def __init__(self, n_features: int, d_cond: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_cond),
            nn.LayerNorm(d_cond),
        )

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        return self.net(mol_features)


# ═════════════════════════════════════════════════════════════════════════════
# Encoder — q-space signal → R-space representation (FiLM-conditioned)
# ═════════════════════════════════════════════════════════════════════════════

class BeamstopEncoder(nn.Module):
    """
    Maps masked ΔS0_norm (nq,) to ΔPD_eff (nR,) via FiLM-conditioned CNNs.

    Architecture:
      [signal, mask] → q-space Conv1d (FiLM) → interpolate to nR
                     → R-space Conv1d (FiLM) → ΔPD_eff

    The mask channel tells the encoder which q values are observed vs. missing.
    Last Conv1d in each block is zero-initialized to prevent blow-up at init.
    """
    def __init__(self, nR: int, n_channels: int = 32,
                 d_cond: int = 64, kernel_size: int = 5):
        super().__init__()
        self.nR = nR
        pad = kernel_size // 2

        self.q_conv1 = nn.Conv1d(2,          n_channels, kernel_size, padding=pad)
        self.q_act1  = nn.PReLU(n_channels)
        self.q_film1 = FiLM(n_channels, d_cond)
        self.q_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad)
        self.q_act2  = nn.PReLU(n_channels)
        self.q_film2 = FiLM(n_channels, d_cond)
        self.q_conv3 = nn.Conv1d(n_channels, 1,          kernel_size, padding=pad)

        self.R_conv1 = nn.Conv1d(1,          n_channels, kernel_size, padding=pad)
        self.R_act1  = nn.PReLU(n_channels)
        self.R_film1 = FiLM(n_channels, d_cond)
        self.R_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad)
        self.R_act2  = nn.PReLU(n_channels)
        self.R_film2 = FiLM(n_channels, d_cond)
        self.R_conv3 = nn.Conv1d(n_channels, 1,          kernel_size, padding=pad)

        nn.init.zeros_(self.q_conv3.weight); nn.init.zeros_(self.q_conv3.bias)
        nn.init.zeros_(self.R_conv3.weight); nn.init.zeros_(self.R_conv3.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mol_cond: torch.Tensor) -> torch.Tensor:
        """
        x        : (batch, nq)     — masked ΔS0_norm
        mask     : (batch, nq)     — 1=observed, 0=masked
        mol_cond : (batch, d_cond)

        Returns  : (batch, nR)
        """
        h = torch.stack([x, mask], dim=1)
        h = self.q_film1(self.q_act1(self.q_conv1(h)), mol_cond)
        h = self.q_film2(self.q_act2(self.q_conv2(h)), mol_cond)
        h = self.q_conv3(h)

        h = F.interpolate(h, size=self.nR, mode='linear', align_corners=True)

        h = self.R_film1(self.R_act1(self.R_conv1(h)), mol_cond)
        h = self.R_film2(self.R_act2(self.R_conv2(h)), mol_cond)
        h = self.R_conv3(h)
        return h.squeeze(1)


# ═════════════════════════════════════════════════════════════════════════════
# BeamstopModel
# ═════════════════════════════════════════════════════════════════════════════

class BeamstopModel(nn.Module):
    """
    Predict the missing low-q region of ΔS0_norm.

    The decoder is a pre-built Debye/sinc operator (B_sinc) stored as a buffer.
    Pre-building is more efficient than recomputing sinc every forward pass.

    Set use_feature_conditioning=True to FiLM-condition on scalar molecular
    features. With it disabled, the same encoder runs without molecular context.

    forward() returns a tuple (pred, delta_pd):
      pred     : (batch, nq) — full predicted ΔS0_norm
      delta_pd : (batch, nR) — R-space intermediate (zero-centred)
                               used for Tikhonov regularisation in training.

    Parameters
    ----------
    B_sinc     : (nR, nq) sinc/Debye operator  (from build_sinc_operator)
    d_cond     : FiLM conditioning vector dim           (default 64)
    n_channels : hidden channels in encoder CNNs        (default 32)
    kernel_size: Conv1d kernel size                     (default 5)
    """
    def __init__(self, B_sinc: torch.Tensor, d_cond: int = 64,
                 n_channels: int = 32, kernel_size: int = 5,
                 use_feature_conditioning: bool = False,
                 mol_feature_dim: int = 13):
        super().__init__()
        self.register_buffer('B_sinc', B_sinc.float())
        self.use_feature_conditioning = use_feature_conditioning
        nR, _ = B_sinc.shape
        self.d_cond    = d_cond
        self.feature_embed = (
            MolFeatureEmbedding(mol_feature_dim, d_cond)
            if use_feature_conditioning else None
        )
        self.encoder   = BeamstopEncoder(nR, n_channels, d_cond, kernel_size)

    def forward(self, delta_S0_masked: torch.Tensor, mask: torch.Tensor,
                mol_features: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        delta_S0_masked : (batch, nq)              — ΔS0_norm with q < q_min = 0
        mask            : (batch, nq)              — 1=observed, 0=masked
        mol_features    : (batch, mol_feature_dim)  required if use_feature_conditioning

        Returns
        -------
        pred     : (batch, nq) — full ΔS0_norm prediction
        delta_pd : (batch, nR) — zero-centred R-space intermediate,
                                 exposed for Tikhonov regularisation
        """
        if self.use_feature_conditioning:
            if mol_features is None:
                raise ValueError("mol_features is required when use_feature_conditioning=True")
            mol_cond = self.feature_embed(mol_features)
        else:
            mol_cond = delta_S0_masked.new_zeros(delta_S0_masked.shape[0], self.d_cond)

        delta_pd = self.encoder(delta_S0_masked, mask, mol_cond)
        # Zero-centre so ΔS0_norm(q=0) = 0:
        # B_sinc[:, q=0] = 2 for all R  →  Σ_m delta_pd[m]·2 = 0 when mean=0.
        delta_pd = delta_pd - delta_pd.mean(dim=-1, keepdim=True)

        pred_raw = delta_pd @ self.B_sinc   # (batch, nq)

        # Hard-constrain the observed region to the measured input. The model
        # only fills the masked gap; observed q's pass through unchanged.
        pred = mask * delta_S0_masked + (1.0 - mask) * pred_raw
        return pred, delta_pd


# ═════════════════════════════════════════════════════════════════════════════
# Loss
# ═════════════════════════════════════════════════════════════════════════════

def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                lambda_smooth: float = 0.0) -> torch.Tensor:
    """
    Masked-region MSE with optional smoothness.

    BeamstopModel.forward hard-pastes observed q values into the output, so the
    observed-region residual is identically zero. The useful data term is only
    the masked low-q region.
    Smoothness          — second-derivative penalty over the full q range.

    Parameters
    ----------
    pred, target  : (batch, nq)
    mask          : (batch, nq)  1=observed, 0=masked
    lambda_smooth : smoothness penalty weight          (default 0)
    """
    residual = pred - target

    n_mask = (1 - mask).sum(dim=-1, keepdim=True).clamp(min=1)

    loss = (residual.pow(2) * (1 - mask) / n_mask).sum(dim=-1).mean()

    if lambda_smooth > 0:
        d2     = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        smooth = d2.pow(2).mean()
        loss   = loss + lambda_smooth * smooth
    return loss


def boundary_continuity_loss(pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """
    Slope + curvature continuity at q_min via finite differences.

    BeamstopModel.forward hard-pastes target values into the observed region,
    so value continuity at q_min is already exact. This loss penalizes
    mismatch in the FIRST and SECOND derivative across the gap-edge by
    constraining the two predicted pixels just below q_min to match a
    quadratic Taylor extrapolation of the measured signal from above.

    Per-sample derivation (with t = target, p = pred, qmi = q_min_idx, dq = grid spacing):
      slope (meas, fwd FD) = (t[qmi+1] - t[qmi]) / dq
      slope (ext,  bwd FD) = (p[qmi]   - p[qmi-1]) / dq
                           = (t[qmi]   - p[qmi-1]) / dq   (paste makes p[qmi]=t[qmi])

      curv  (meas, fwd FD) = (t[qmi+2] - 2 t[qmi+1] + t[qmi]) / dq²
      curv  (ext,  bwd FD) = (p[qmi]   - 2 p[qmi-1] + p[qmi-2]) / dq²

    Setting ext = meas and solving for the two unknown pixels gives:
      p[qmi-1] = 2 t[qmi] -   t[qmi+1]
      p[qmi-2] = 4 t[qmi] - 4 t[qmi+1] + t[qmi+2]

    The dq factors cancel out so the result is a unit-free MSE on the two
    boundary pixels.

    Parameters
    ----------
    pred, target : (batch, nq)
    mask         : (batch, nq)  1=observed, 0=masked

    Returns
    -------
    scalar loss. Zero if no sample has ≥2 masked + ≥3 observed pixels at the
    boundary (rare; only edge-case masks).
    """
    qmi   = mask.argmax(dim=-1)                          # (B,) first obs index
    B, nq = pred.shape
    valid = (qmi >= 2) & (qmi + 2 < nq)
    if not valid.any():
        return pred.new_zeros(())

    b   = torch.arange(B, device=pred.device)[valid]
    qmi = qmi[valid]

    t0 = target[b, qmi]
    t1 = target[b, qmi + 1]
    t2 = target[b, qmi + 2]

    desired_m1 = 2.0 * t0 -       t1
    desired_m2 = 4.0 * t0 - 4.0 * t1 + t2

    return ((pred[b, qmi - 1] - desired_m1).pow(2)
          + (pred[b, qmi - 2] - desired_m2).pow(2)).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Sanity check
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import glob

    npz_files = sorted(glob.glob('data/trajectory_data*/**/*.npz', recursive=True))

    if npz_files:
        ref  = np.load(npz_files[0])
        q_np = ref['q'];  R_np = ref['R_grid']
        print(f'Building B_sinc from {npz_files[0]}')
    else:
        print('No data found — using synthetic data')
        q_np      = np.arange(1e-4, 10.1, 0.1, dtype=np.float32)
        R_np      = np.linspace(0.025, 31.4, 1257, dtype=np.float32)

    B_sinc = build_sinc_operator(q_np, R_np)
    model  = BeamstopModel(B_sinc, n_channels=32)

    batch     = 4
    nq        = len(q_np)
    dq        = float(q_np[1] - q_np[0])
    q_min_idx = int(0.75 / dq)

    mask          = torch.ones(batch, nq)
    mask[:, :q_min_idx] = 0.0
    delta_S0      = torch.randn(batch, nq) * 0.1
    delta_S0_masked = delta_S0 * mask

    pred, delta_pd = model(delta_S0_masked, mask)
    loss = masked_loss(pred, delta_S0, mask)

    print(f'pred shape   : {tuple(pred.shape)}')
    print(f'pred[:,0]    : {pred[:, 0].detach().tolist()}  ← should be ~0')
    print(f'loss         : {loss.item():.6f}  nan={loss.isnan().item()}')
    print(f'Parameters   : {sum(p.numel() for p in model.parameters()):,}')
