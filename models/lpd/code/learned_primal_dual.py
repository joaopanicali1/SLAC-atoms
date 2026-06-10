"""
learned_primal_dual.py — Learned Primal-Dual model for pair density inversion.

Implements Algorithm 3 from Adler & Öktem (IEEE TMI 2018), adapted for the
X-ray scattering inverse problem:

    dS0(q) → pd_true(R)

where dS0 is the normalized q-space scattering signal and pd_true is the
ground-truth pair density from atomic positions.

Input/output convention (matches LPDDataset):
    dS0     : (batch, nq)  — normalized scattering signal  [input]
    pd_pred : (batch, nR)  — reconstructed pair density    [output]

Physics operators (fixed, not learned):
    T  : R-space → q-space  via A_dbt        (forward / synthesis)
    T* : q-space → R-space  via A_dbt.T      (adjoint / DBT inversion)

The primal variable f is warm-started from T*(dS0) — the naive DBT estimate —
rather than zero, matching the measurement in the first iteration.

Usage
-----
    from models.lpd.dataset.dataset_lpd import LPDDataset
    from models.lpd.code.learned_primal_dual import LearnedPrimalDual, build_dbt_operator

    ds = LPDDataset('data/trajectory_data_cap200_new')
    A_dbt, Rgrid = build_dbt_operator(ds.q.numpy(), M)
    model = LearnedPrimalDual(A_dbt)
    pd_pred = model(dS0_batch)   # (batch, nR)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn


# ═════════════════════════════════════════════════════════════════════════════
# DBT — Discrete Bessel Transform (port of Natan group MATLAB DBT.m)
# ═════════════════════════════════════════════════════════════════════════════

def _bessel_zeros_spherical(n: int, k: int) -> np.ndarray:
    """First k positive zeros of the spherical Bessel function j_n."""
    coeff1 = [0.411557013144507, 0.999986723293410, 0.698028985524484, 1.06977507291468]
    exp1   = [0.335300369843979, 0.339671493811664]
    coeff2 = [1.93395115137444, 1.00007656297072, -0.805720018377132, 3.38764629174694]
    exp2   = [0.456215294517928, 0.388380341189200]
    coeff3 = [5.40770803992613, 1.00093850589418, 2.66926179799040, -0.174925559314932]
    exp3   = [0.429702214054531, 0.633480051735955]

    zeros = np.zeros(k)

    def _guess1(n): return coeff1[0] + coeff1[1]*n + coeff1[2]*(n+1)**exp1[0] + coeff1[3]*(n+1)**exp1[1]
    def _guess2(n): return coeff2[0] + coeff2[1]*n + coeff2[2]*(n+1)**exp2[0] + coeff2[3]*(n+1)**exp2[1]
    def _guess3(n): return coeff3[0] + coeff3[1]*n + coeff3[2]*(n+1)**exp3[0] + coeff3[3]*(n+1)**exp3[1]

    def halley(n_ord, x0):
        x = x0
        TOL = 1e4 * np.finfo(float).eps
        for _ in range(100):
            a = spherical_jn(n_ord, x)
            b = spherical_jn(n_ord + 1, x)
            x2 = x * x
            denom = 2*b*b*x2 - a*b*x*(4*n_ord+1) + (n_ord*(n_ord+1) + x2)*a*a
            if abs(denom) < 1e-300:
                break
            err = 2*a*x*(n_ord*a - b*x) / denom
            x -= err
            if abs(err) < TOL * abs(x):
                break
        return x

    zeros[0] = halley(n, _guess1(n))
    if k >= 2: zeros[1] = halley(n, _guess2(n))
    if k >= 3: zeros[2] = halley(n, _guess3(n))
    for i in range(3, k):
        zeros[i] = halley(n, 2*zeros[i-1] - zeros[i-2])
    return zeros


def _dbt(n: int, data: np.ndarray, q: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Bessel Transform of order n. Returns (PD, Rgrid)."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze_out = True
    else:
        squeeze_out = False

    dq     = q[1] - q[0]
    qp_end = q[0] + dq * (M - 1)
    BZ     = _bessel_zeros_spherical(n, M)

    qgrid = qp_end * BZ[:M-1] / BZ[M-1]
    Rgrid = BZ[:M-1] / qp_end

    in_range = (qgrid >= q[0]) & (qgrid <= q[-1])
    data_s = np.zeros((M-1, data.shape[1]))
    if np.any(in_range):
        interp = PchipInterpolator(q, data, axis=0, extrapolate=False)
        data_s[in_range] = interp(qgrid[in_range])

    scalingFactor = qp_end**2 / BZ[M-1]
    bz_inner = BZ[:M-1]
    denom = BZ[M-1] * spherical_jn(n+1, bz_inner)**2
    G = 2 * spherical_jn(n, np.outer(bz_inner, bz_inner) / BZ[M-1]) / denom[:, np.newaxis]

    PD = G @ (data_s * qgrid[:, np.newaxis]) * scalingFactor
    if squeeze_out:
        PD = PD[:, 0]
    return PD, Rgrid


# ═════════════════════════════════════════════════════════════════════════════
# DBT operator matrix  (precomputed once, used as fixed buffers in the model)
# ═════════════════════════════════════════════════════════════════════════════

def build_dbt_operator(q: np.ndarray, M: int, normalize: bool = True,
                       verbose: bool = True):
    """
    Build the DBT operator as a dense matrix A_dbt of shape (nR, nq).

    A_dbt[:, j] = _dbt(e_j, q, M)  where e_j is the j-th standard basis vector.

    If normalize=True (default), divide A_dbt by its spectral norm so that
    ‖T‖ = ‖T*‖ = 1. This is essential for stability in unrolled iterative
    schemes (Adler & Öktem 2018): a perturbation δ in iteration k propagates
    as δ·‖A‖^k through the loop, so anything ‖A‖ > 1 causes exponential
    blowup over n_iter steps.

    Parameters
    ----------
    q         : (nq,) array of q values in Å⁻¹
    M         : DBT order — controls R-space grid density.
                Typically M = ceil(π / (dR · dq)) + 1.
    normalize : if True, scale A by 1/σ_max so spectral norm = 1.

    Returns
    -------
    A_dbt : torch.Tensor  (nR, nq)  float32
    Rgrid : torch.Tensor  (nR,)     float32  [Å]
    """
    nq = len(q)
    e0 = np.zeros(nq); e0[0] = 1.0
    col0, Rgrid_np = _dbt(0, e0, q, M)
    nR = len(col0)

    A = np.zeros((nR, nq), dtype=np.float32)
    for j in range(nq):
        e_j = np.zeros(nq); e_j[j] = 1.0
        A[:, j], _ = _dbt(0, e_j, q, M)

    sigma_max = float(np.linalg.norm(A, ord=2))
    if normalize:
        A = A / sigma_max

    if verbose:
        print(f'A_dbt      : {A.shape}  ({A.nbytes / 1e6:.1f} MB)')
        print(f'Rgrid      : [{Rgrid_np[0]:.3f}, {Rgrid_np[-1]:.2f}] Å  ({nR} pts)')
        print(f'σ_max(A)   : {sigma_max:.4f}  ({"normalized to 1" if normalize else "kept as-is"})')

    return torch.tensor(A, dtype=torch.float32), \
           torch.tensor(Rgrid_np, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Per-iteration networks
# ═════════════════════════════════════════════════════════════════════════════

class DualNet(nn.Module):
    """
    Dual update CNN — operates in q-space.

    Takes the current dual state, the forward projection of the current primal
    estimate, and the measurement signal, and returns an updated dual state.

    Input  : concat(h_prev, T(f²), g)  →  (n_dual + 2) channels
    Output : h_new  (residual update)  →  n_dual channels
    """
    def __init__(self, n_dual: int, n_channels: int = 32, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_dual + 2, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels, n_dual,     kernel_size, padding=pad),
        )
        # Zero-init last layer so residual path starts as identity at init,
        # preventing exponential blow-up across the unrolled iterations.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, Tf: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        h  : (batch, n_dual, nq)  — current dual state
        Tf : (batch, 1,      nq)  — T applied to second primal channel
        g  : (batch, 1,      nq)  — fixed measurement signal
        """
        return h + self.net(torch.cat([h, Tf, g], dim=1))


class PrimalNet(nn.Module):
    """
    Primal update CNN — operates in R-space.

    Takes the current primal state and the back-projection of the first dual
    channel, and returns an updated primal state.

    Input  : concat(f_prev, T*(h¹))  →  (n_primal + 1) channels
    Output : f_new (residual update) →  n_primal channels
    """
    def __init__(self, n_primal: int, n_channels: int = 32, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_primal + 1, n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels,   n_channels, kernel_size, padding=pad),
            nn.PReLU(n_channels),
            nn.Conv1d(n_channels,   n_primal,   kernel_size, padding=pad),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f: torch.Tensor, Tstar_h: torch.Tensor) -> torch.Tensor:
        """
        f       : (batch, n_primal, nR)  — current primal state
        Tstar_h : (batch, 1,        nR)  — T* applied to first dual channel
        """
        return f + self.net(torch.cat([f, Tstar_h], dim=1))


# ═════════════════════════════════════════════════════════════════════════════
# Learned Primal-Dual model
# ═════════════════════════════════════════════════════════════════════════════

class LearnedPrimalDual(nn.Module):
    """
    Learned Primal-Dual (Adler & Öktem, IEEE TMI 2018) for pair density inversion.

    Maps normalized scattering signal dS0(q) → ground-truth pair density pd_true(R).

    The physics operators use the precomputed DBT matrix A_dbt:
      T (forward)  : pd(R) → dS0_approx(q)  via  pd @ A_dbt
      T* (adjoint) : dS0(q) → pd(R)         via  dS0 @ A_dbt.T

    The primal variable f is warm-started from T*(dS0) — the naive DBT
    inversion — so the first iteration refines an already meaningful estimate.

    Parameters
    ----------
    A_dbt      : (nR, nq) precomputed DBT operator matrix
    n_primal   : memory channels in R-space  (paper default: 5)
    n_dual     : memory channels in q-space  (paper default: 5)
    n_iter     : number of unrolled iterations (paper default: 10)
    n_channels : CNN hidden channels           (paper default: 32)
    kernel_size: Conv1d kernel size            (default: 5)
    """
    def __init__(
        self,
        A_dbt: torch.Tensor,
        n_primal: int = 5,
        n_dual: int = 5,
        n_iter: int = 10,
        n_channels: int = 32,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.register_buffer('A_dbt', A_dbt.float())   # (nR, nq) — fixed
        self.n_primal = n_primal
        self.n_dual   = n_dual
        self.n_iter   = n_iter

        # Separate networks per iteration — key improvement over basic LISTA
        self.dual_nets = nn.ModuleList(
            [DualNet(n_dual, n_channels, kernel_size) for _ in range(n_iter)]
        )
        self.primal_nets = nn.ModuleList(
            [PrimalNet(n_primal, n_channels, kernel_size) for _ in range(n_iter)]
        )

    # ── Physics operators ──────────────────────────────────────────────────

    def T(self, f: torch.Tensor) -> torch.Tensor:
        """Forward: pd(R) → dS0_approx(q).  (batch, nR) → (batch, nq)"""
        return f @ self.A_dbt          # (batch, nR) @ (nR, nq) = (batch, nq)

    def T_star(self, h: torch.Tensor) -> torch.Tensor:
        """Adjoint / DBT: dS0(q) → pd(R).  (batch, nq) → (batch, nR)"""
        return h @ self.A_dbt.T        # (batch, nq) @ (nq, nR) = (batch, nR)

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, dS0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        dS0 : (batch, nq)  — normalized q-space scattering signal

        Returns
        -------
        pd_pred : (batch, nR)  — reconstructed ground-truth pair density
        """
        batch  = dS0.shape[0]
        nR, nq = self.A_dbt.shape

        g    = dS0.unsqueeze(1)                                  # (batch, 1, nq)
        h    = dS0.new_zeros(batch, self.n_dual, nq)             # dual   — q-space

        # Warm-start primal from naive DBT inversion T*(dS0).
        f    = dS0.new_zeros(batch, self.n_primal, nR)           # primal — R-space
        f[:, 0, :] = self.T_star(dS0)                            # first channel = pd_limited

        for i in range(self.n_iter):
            # Dual update: hᵢ = Λᵈᵢ(hᵢ₋₁, T(f²ᵢ₋₁), g)
            Tf  = self.T(f[:, 1, :]).unsqueeze(1)               # (batch, 1, nq)
            h   = self.dual_nets[i](h, Tf, g)

            # Primal update: fᵢ = Λᵖᵢ(fᵢ₋₁, T*(h¹ᵢ))
            Tsh = self.T_star(h[:, 0, :]).unsqueeze(1)          # (batch, 1, nR)
            f   = self.primal_nets[i](f, Tsh)

        return f[:, 0, :]              # (batch, nR) — first primal channel


# ═════════════════════════════════════════════════════════════════════════════
# Sanity check
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    import glob
    npz_files = sorted(glob.glob('data/trajectory_data*/**/*.npz', recursive=True))

    if npz_files:
        ref  = np.load(npz_files[0])
        q_np = ref['q']
        dR   = float(ref['R_grid'][1] - ref['R_grid'][0])
        dq   = float(q_np[1] - q_np[0])
        M    = int(np.ceil(np.pi / (dR * dq))) + 1
        print(f'Building A_dbt from {npz_files[0]}  (M={M})')
        A_dbt, Rgrid = build_dbt_operator(q_np, M)
        nq = len(q_np)
    else:
        print('No data found — using synthetic operator for sanity check')
        nq, nR = 101, 1257
        A_dbt  = torch.randn(nR, nq) * 0.01
        Rgrid  = torch.linspace(0.025, 31.4, nR)

    model = LearnedPrimalDual(
        A_dbt      = A_dbt,
        n_primal   = 5,
        n_dual     = 5,
        n_iter     = 10,
        n_channels = 32,
        kernel_size= 5,
    )

    nR    = A_dbt.shape[0]
    batch = 4
    dS0   = torch.randn(batch, nq)

    pd_pred = model(dS0)

    print(f'Input  dS0    : {tuple(dS0.shape)}')
    print(f'Output pd_pred: {tuple(pd_pred.shape)}')
    print(f'Parameters    : {sum(p.numel() for p in model.parameters()):,}')
    print(f'A_dbt         : {tuple(A_dbt.shape)}  ({A_dbt.nbytes / 1e6:.1f} MB)')
