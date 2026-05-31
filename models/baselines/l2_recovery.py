import numpy as np
from scipy.special import spherical_jn


def build_D_dict(q: np.ndarray, R_grid: np.ndarray, detector: np.ndarray, M: int, DBT_fn):
    """
    Build the NSK dictionary D used by both L1 and L2 recovery.
    Each column is a sinc basis function at position R_i, transformed
    through the same DBT pipeline as the data.

    Parameters
    ----------
    q         : (nq,) momentum transfer grid
    R_grid    : (nR,) real-space grid from DBT
    detector  : (nq,) detector window (e.g. Lorch)
    M         : int, DBT output size
    DBT_fn    : callable, the DBT function from your pipeline

    Returns
    -------
    D_dict : (nR, nR) dictionary matrix
    """
    nq = len(q)
    nR = len(R_grid)

    # sinc basis: each column is j_0(q * R_i) = sinc(q*R_i / pi)
    bes = np.zeros((nq, nR))
    for i, Ri in enumerate(R_grid):
        bes[:, i] = spherical_jn(0, q * Ri)

    # transform through DBT with detector window (same as data)
    D_dict, _ = DBT_fn(0, bes * detector[:, np.newaxis], q, M)
    return D_dict  # (nR, nR)


def recover_l2(pd_limited: np.ndarray, D: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """
    L2 (ridge regression) recovery of pair density.

    Solves: min_x ||D x - y||^2 + lam * ||x||^2
    Closed form: x = (D^T D + lam * I)^{-1} D^T y

    Parameters
    ----------
    pd_limited : (nR,) or (nR, nRHS) bandlimited pair density (input signal)
    D          : (nR, nCoef) dictionary matrix
    lam        : float, ridge penalty (default 1e-2)

    Returns
    -------
    x_recovered : (nCoef,) or (nCoef, nRHS) recovered coefficients
    """
    squeeze = pd_limited.ndim == 1
    if squeeze:
        pd_limited = pd_limited[:, np.newaxis]

    nObs, nCoef = D.shape
    Y = pd_limited  # (nObs, nRHS)

    # column normalisation (same as decon_TVL1 for fair comparison)
    col_scale = np.sqrt(np.sum(D ** 2, axis=0))
    col_scale[col_scale == 0] = 1.0
    Dn = D / col_scale[np.newaxis, :]

    DtD = Dn.T @ Dn
    DtY = Dn.T @ Y

    # closed-form ridge solve
    A = DtD + lam * np.eye(nCoef)
    A = 0.5 * (A + A.T)  # symmetrise for numerical stability
    x_norm = np.linalg.solve(A, DtY)

    # undo column normalisation
    x = x_norm / col_scale[:, np.newaxis]

    if squeeze:
        x = x[:, 0]

    return x


def auto_lam_l2(pd_limited: np.ndarray, D: np.ndarray, lam_frac: float = 1e-2) -> float:
    """
    Set ridge lambda proportional to the max of D^T y,
    matching the auto-lambda logic in decon_TVL1.
    """
    y = pd_limited if pd_limited.ndim == 2 else pd_limited[:, np.newaxis]
    col_scale = np.sqrt(np.sum(D ** 2, axis=0))
    col_scale[col_scale == 0] = 1.0
    Dn = D / col_scale[np.newaxis, :]
    DtY = Dn.T @ y
    return lam_frac * np.max(np.abs(DtY))