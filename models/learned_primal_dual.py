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
    from data.Dataset.dataset_lpd import LPDDataset
    from models.learned_primal_dual import LearnedPrimalDual, build_dbt_operator

    ds = LPDDataset('data/trajectory_data_cap200')
    A_dbt, Rgrid = build_dbt_operator(ds.q.numpy(), M)
    model = LearnedPrimalDual(A_dbt)
    pd_pred = model(dS0_batch)   # (batch, nR)
"""

import numpy as np
import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# DBT operator matrix  (precomputed once, used as fixed buffers in the model)
# ═════════════════════════════════════════════════════════════════════════════

def build_dbt_operator(q: np.ndarray, M: int, verbose: bool = True):
    """
    Build the DBT operator as a dense matrix A_dbt of shape (nR, nq).

    A_dbt[:, j] = DBT(e_j, q, M)  where e_j is the j-th standard basis vector.

    The DBT function is imported from processing/analyse_mol_dbt.py, which
    implements the exact MATLAB DBT.m methodology (Natan 2020).

    Parameters
    ----------
    q : (nq,) array of q values in Å⁻¹
    M : DBT order — controls the R-space grid density.
        Typically M = ceil(π / (dR · dq)) + 1, matching analyse_mol_v2.py.

    Returns
    -------
    A_dbt : torch.Tensor  (nR, nq)  float32
    Rgrid : torch.Tensor  (nR,)     float32  [Å]
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))
    from analyse_mol_v2 import DBT

    nq = len(q)
    e0 = np.zeros(nq); e0[0] = 1.0
    col0, Rgrid_np = DBT(0, e0, q, M)
    nR = len(col0)

    A = np.zeros((nR, nq), dtype=np.float32)
    for j in range(nq):
        e_j = np.zeros(nq); e_j[j] = 1.0
        A[:, j], _ = DBT(0, e_j, q, M)

    if verbose:
        print(f'A_dbt  : {A.shape}  ({A.nbytes / 1e6:.1f} MB)')
        print(f'Rgrid  : [{Rgrid_np[0]:.3f}, {Rgrid_np[-1]:.2f}] Å  ({nR} pts)')

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
