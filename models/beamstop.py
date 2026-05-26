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
  2. Smoothness            — enforced architecturally via R-cap band-limiting
                             (R_max controls fastest q-space oscillation period)

Architecture (physics-guided encoder-decoder with mode-embedding FiLM conditioning):
  MolEmbedding : V_vib + omegas + masks → mol_cond (d_cond,)
                   per-atom MLP per mode → masked atom pool → cat ω
                   → per-mode embed → masked mode pool → single vector
  Encoder      : Conv1d in q-space, FiLM-conditioned on mol_cond
                   → interpolate to nR
                   → Conv1d in R-space, FiLM-conditioned on mol_cond
                   → ΔPD_eff (batch, nR)
  Decoder      : Differentiable Debye sum (mirrors simulate_scattering):
                   zero-center(ΔPD_eff) → Σ_m ΔPD[m] · 2·j₀(q·R[m])
                   → ΔS0_norm_pred (batch, nq)

                 This is identical in structure to simulate_scattering's:
                   bessel = sinc(q·r_ab / π)
                   ΔS0 = 2 · Σ_pairs(fafb · bessel)
                 but applied to the continuous R-grid pair density rather than
                 discrete atom pairs (uniform form-factor weights, since ΔS0 is
                 already normalized by fe).

FiLM (Feature-wise Linear Modulation) applies per-channel scale γ and shift β
derived from mol_cond after each activation: out = γ(mol_cond) * act + β(mol_cond).
Initialized to identity (γ=1, β=0) so conditioning starts neutral.

Loss (masked_loss):
  Masked upweighting  — masked (low-q) MSE upweighted by lambda_masked
                         relative to observed MSE, both per-q-point normalized

Usage
-----
    from models.beamstop import BeamstopModel
    model = BeamstopModel(q_tensor, R_grid_tensor, max_atoms=75, max_modes=219)
    pred  = model(delta_S0_masked, mask, V_vib, omegas, atom_mask, mode_mask)
"""

import math
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

    Initialized to identity (all weights/biases zero → γ=1, β=0) so the
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


# ═════════════════════════════════════════════════════════════════════════════
# MolEmbedding — normal modes → single conditioning vector
# ═════════════════════════════════════════════════════════════════════════════

class MolEmbedding(nn.Module):
    """
    Encodes normal-mode eigenvectors and frequencies into a single conditioning
    vector used by FiLM layers throughout the encoder.

    Pipeline per molecule:
      1. V_vib (3*max_atoms, max_modes) → (max_modes, max_atoms, 3)
      2. atom_mlp applied to the last dim (3,): → (max_modes, max_atoms, d_atom)
      3. Masked mean-pool over atoms (using atom_mask) → (max_modes, d_atom)
      4. Concatenate ω_k → (max_modes, d_atom+1)
      5. Linear out_proj → (max_modes, d_mode)
      6. Thermal-amplitude-weighted pool over modes:
         weight_k ∝ 1/ω_k² (proportional to A_k² = 2kBT/ω_k²),
         masked by mode_mask → (d_mode,)
      7. Linear final + LayerNorm → (d_cond,)

    The 1/ω² weighting encodes the physical fact that low-frequency soft
    modes have larger thermal amplitudes and dominate the structural signal,
    especially in the low-q region the model is trying to recover. Plain
    mean-pooling would dilute these dominant modes with stiff C-H stretches.

    Parameters
    ----------
    max_atoms : padded atom count (must match V_vib first dim // 3)
    d_atom    : hidden dim of per-atom MLP
    d_mode    : per-mode embedding dim before final projection
    d_cond    : output conditioning dim (fed to FiLM layers)
    """
    def __init__(self, max_atoms: int = 75, d_atom: int = 64,
                 d_mode: int = 128, d_cond: int = 64):
        super().__init__()
        self.max_atoms = max_atoms

        self.atom_mlp = nn.Sequential(
            nn.Linear(3, d_atom),
            nn.ReLU(),
            nn.Linear(d_atom, d_mode),
        )
        self.out_proj = nn.Linear(d_mode + 1, d_mode)
        self.final    = nn.Sequential(
            nn.Linear(d_mode, d_cond),
            nn.LayerNorm(d_cond),
        )

    def forward(self, V_vib: torch.Tensor, omegas: torch.Tensor,
                atom_mask: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
        """
        V_vib     : (batch, 3*max_atoms, max_modes)
        omegas    : (batch, max_modes)
        atom_mask : (batch, max_atoms)  bool — True for real atoms
        mode_mask : (batch, max_modes)  bool — True for real modes

        Returns   : (batch, d_cond)
        """
        batch, _, max_modes = V_vib.shape

        # (batch, max_atoms, 3, max_modes) → (batch, max_modes, max_atoms, 3)
        x = V_vib.reshape(batch, self.max_atoms, 3, max_modes).permute(0, 3, 1, 2)

        # Per-atom embedding: (batch, max_modes, max_atoms, d_mode)
        x = self.atom_mlp(x)

        # Masked mean-pool over atoms: (batch, max_modes, d_mode)
        atom_m = atom_mask.unsqueeze(1).unsqueeze(-1).float()   # (batch, 1, max_atoms, 1)
        x = (x * atom_m).sum(dim=2) / atom_m.sum(dim=2).clamp(min=1)

        # Concatenate frequency, project: (batch, max_modes, d_mode)
        x = self.out_proj(torch.cat([x, omegas.unsqueeze(-1)], dim=-1))

        # Thermal-amplitude-weighted pool over modes: weight ∝ 1/ω², masked.
        # Soft (low-ω) modes have larger A_k = sqrt(2kBT)/ω_k and dominate the
        # structural signal at low q — give them more weight in the embedding.
        weight = 1.0 / (omegas.pow(2) + 1e-6)                    # (batch, max_modes)
        weight = (weight * mode_mask.float()).unsqueeze(-1)      # (batch, max_modes, 1)
        x = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1e-6)

        return self.final(x)                                    # (batch, d_cond)


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
    def __init__(self, nq: int, nR: int, n_channels: int = 32,
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
    Predicts the missing low-q region of ΔS0_norm, conditioned on the
    molecule's normal modes and vibrational frequencies via FiLM.

    The decoder is a pre-built Debye/sinc operator (B_sinc) stored as a buffer.
    Pre-building is more efficient than recomputing sinc every forward pass.

    forward() returns a tuple (pred, delta_pd):
      pred     : (batch, nq) — full predicted ΔS0_norm
      delta_pd : (batch, nR) — R-space intermediate (zero-centred)
                               used for Tikhonov regularisation in training.

    Parameters
    ----------
    B_sinc     : (nR, nq) sinc/Debye operator  (from build_sinc_operator)
    max_atoms  : padded atom count in V_vib             (default 75)
    max_modes  : padded mode count in V_vib / omegas    (default 219)
    d_atom     : hidden dim in per-atom MLP             (default 64)
    d_mode     : per-mode embedding dim                 (default 128)
    d_cond     : FiLM conditioning vector dim           (default 64)
    n_channels : hidden channels in encoder CNNs        (default 32)
    kernel_size: Conv1d kernel size                     (default 5)
    """
    def __init__(self, B_sinc: torch.Tensor, max_atoms: int = 75,
                 max_modes: int = 219, d_atom: int = 64, d_mode: int = 128,
                 d_cond: int = 64, n_channels: int = 32, kernel_size: int = 5,
                 use_conditioning: bool = True):
        super().__init__()
        self.register_buffer('B_sinc', B_sinc.float())
        self.use_conditioning = use_conditioning
        nR, nq = B_sinc.shape
        self.d_cond    = d_cond
        self.mol_embed = MolEmbedding(max_atoms, d_atom, d_mode, d_cond) if use_conditioning else None
        self.encoder   = BeamstopEncoder(nq, nR, n_channels, d_cond, kernel_size)

    def forward(self, delta_S0_masked: torch.Tensor, mask: torch.Tensor,
                V_vib: torch.Tensor | None = None, omegas: torch.Tensor | None = None,
                atom_mask: torch.Tensor | None = None,
                mode_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        delta_S0_masked : (batch, nq)              — ΔS0_norm with q < q_min = 0
        mask            : (batch, nq)              — 1=observed, 0=masked
        V_vib           : (batch, 3*max_atoms, max_modes)  required if use_conditioning
        omegas          : (batch, max_modes)                required if use_conditioning
        atom_mask       : (batch, max_atoms)  bool          required if use_conditioning
        mode_mask       : (batch, max_modes)  bool          required if use_conditioning

        Returns
        -------
        pred     : (batch, nq) — full ΔS0_norm prediction
        delta_pd : (batch, nR) — zero-centred R-space intermediate,
                                 exposed for Tikhonov regularisation
        """
        if self.use_conditioning:
            mol_cond = self.mol_embed(V_vib, omegas, atom_mask, mode_mask)
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
                lambda_masked: float = 10.0, lambda_smooth: float = 0.0) -> torch.Tensor:
    """
    Combined loss with masked-region upweighting and optional smoothness.

    Masked upweighting  — masked (low-q) MSE normalized per q-point and
                          multiplied by lambda_masked vs observed region.
    Smoothness          — second-derivative penalty over the full q range.
                          BeamstopModel.forward hard-pastes observed values
                          into the output, so smoothness in the observed
                          region is a constant w.r.t. parameters (no gradient
                          there). The gradient signal lands on the masked
                          region and on the boundary between the two —
                          enforcing both intra-gap smoothness and continuity
                          with the measurement at q = q_min.

    Parameters
    ----------
    pred, target  : (batch, nq)
    mask          : (batch, nq)  1=observed, 0=masked
    lambda_masked : upweight factor for masked region (default 10)
    lambda_smooth : smoothness penalty weight          (default 0)
    """
    residual = pred - target

    n_obs  = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    n_mask = (1 - mask).sum(dim=-1, keepdim=True).clamp(min=1)

    mse_obs  = (residual.pow(2) * mask      / n_obs ).sum(dim=-1).mean()
    mse_mask = (residual.pow(2) * (1-mask)  / n_mask).sum(dim=-1).mean()

    loss = mse_obs + lambda_masked * mse_mask
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
        omegas_np = ref['omegas_ASE'];  V_vib_np = ref['V_vib']
        print(f'Building B_sinc from {npz_files[0]}')
        print(f'omegas: {omegas_np.shape}, V_vib: {V_vib_np.shape}')
    else:
        print('No data found — using synthetic data')
        q_np      = np.arange(1e-4, 10.1, 0.1, dtype=np.float32)
        R_np      = np.linspace(0.025, 31.4, 1257, dtype=np.float32)
        omegas_np = np.linspace(50, 3500, 30, dtype=np.float32)
        V_vib_np  = np.random.randn(30, 30).astype(np.float32)

    MAX_ATOMS = 75; MAX_MODES = 219
    B_sinc = build_sinc_operator(q_np, R_np)
    model  = BeamstopModel(B_sinc, max_atoms=MAX_ATOMS, max_modes=MAX_MODES,
                           n_channels=32)

    batch     = 4
    nq        = len(q_np)
    dq        = float(q_np[1] - q_np[0])
    q_min_idx = int(0.75 / dq)

    # Build padded inputs
    n_modes = len(omegas_np);  n_atoms = V_vib_np.shape[0] // 3
    V_pad   = np.zeros((3 * MAX_ATOMS, MAX_MODES), dtype=np.float32)
    V_pad[:3*n_atoms, :n_modes] = V_vib_np[:3*n_atoms, :n_modes]
    om_pad  = np.zeros(MAX_MODES, dtype=np.float32)
    om_pad[:n_modes] = omegas_np

    V_vib_t   = torch.tensor(V_pad).unsqueeze(0).expand(batch, -1, -1)
    omegas_t  = torch.tensor(om_pad).unsqueeze(0).expand(batch, -1)
    atom_mask = torch.zeros(batch, MAX_ATOMS, dtype=torch.bool)
    atom_mask[:, :n_atoms] = True
    mode_mask = torch.zeros(batch, MAX_MODES, dtype=torch.bool)
    mode_mask[:, :n_modes] = True

    mask          = torch.ones(batch, nq)
    mask[:, :q_min_idx] = 0.0
    delta_S0      = torch.randn(batch, nq) * 0.1
    delta_S0_masked = delta_S0 * mask

    pred, delta_pd = model(delta_S0_masked, mask, V_vib_t, omegas_t, atom_mask, mode_mask)
    loss = masked_loss(pred, delta_S0, mask)

    print(f'pred shape   : {tuple(pred.shape)}')
    print(f'pred[:,0]    : {pred[:, 0].detach().tolist()}  ← should be ~0')
    print(f'loss         : {loss.item():.6f}  nan={loss.isnan().item()}')
    print(f'Parameters   : {sum(p.numel() for p in model.parameters()):,}')
