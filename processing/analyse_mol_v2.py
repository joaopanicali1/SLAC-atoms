import numpy as np
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.special import spherical_jn
from scipy.io import loadmat 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from form_factors.atomic_form_factors import ATOMIC_FORM_FACTORS
from ase import units as _ase_units
from ase.db import connect as _ase_connect

import re
from pathlib import Path
import os 

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DB_PATH = os.path.join(PROJECT_ROOT, "train_4M", "data0000.aselmdb")

_DB = _ase_connect(DB_PATH)


def atomic_form_factor(element: str, q: np.ndarray) -> np.ndarray:
    """
    WK form factors (from ATOMIC_FORM_FACTORS). 
    f(q) = sum a_i * exp(-b_i * (q / 4pi)^2) + c
    """
    entry = ATOMIC_FORM_FACTORS.get(element)
    Z = entry['Z']
    a = entry['a']
    b = entry['b']
    c = entry['c']

    s2 = (q / (4 * np.pi)) ** 2
    return np.exp(-np.outer(s2, b)) @ a + c

def build_scattering_context(
    atom_names: list[str],
    q: np.ndarray,
):
    """
    Precompute everything that depends only on atoms and q grid. 
    """
    f0 = np.column_stack([
        atomic_form_factor(el, q)
        for el in atom_names
    ])

    all_pairs = np.array(
        list(combinations(range(len(atom_names)), 2))
    )

    fafb = (
        f0[:, all_pairs[:, 0]]
        * f0[:, all_pairs[:, 1]]
    )

    fe = np.sum(fafb, axis=1)
    self_scatter = np.sum(f0**2, axis=1)

    Z = np.array([
        ATOMIC_FORM_FACTORS[el]["Z"]
        for el in atom_names
    ])

    pair_weights = (
        Z[all_pairs[:, 0]]
        * Z[all_pairs[:, 1]]
    )

    return {
        "f0": f0,
        "all_pairs": all_pairs,
        "fafb": fafb,
        "fe": fe,
        "self_scatter": self_scatter,
        "pair_weights": pair_weights,
    }

def get_atom_names(sample_idx):
    """
    Look up atom symbols from the ASE database.
    Falls back to reading 'symbols' from the npz if sample_idx is None
    (e.g. mol_water_physics.npz created outside the database pipeline).
    """
    if sample_idx is not None:
        row = _DB.get(int(sample_idx) + 1)
        return list(row.toatoms().get_chemical_symbols())

def simulate_scattering(q: np.ndarray, atom_names: list, xyz: np.ndarray):
    """
    Compute S_0(q) from q (momentum transfer), list of atoms + their positions
    """
    # form factors
    f0 = np.column_stack([atomic_form_factor(el, q) for el in atom_names])
    
    all_pairs = np.array(list(combinations(range(len(atom_names)), 2)))

    # pair distances
    diff = xyz[all_pairs[:, 1]] - xyz[all_pairs[:, 0]]
    r_ab = np.sqrt(np.sum(diff ** 2, axis=1))

    # pair form-factor products
    fafb = f0[:, all_pairs[:, 0]] * f0[:, all_pairs[:, 1]]

    # Debye Scattering equation sum
    S0 = np.sum(f0**2, axis=1)
    qr = q[:, None] * r_ab[None, :]
    bessel = np.sinc(qr / np.pi)
    S0 = S0 + 2 * np.sum(fafb * bessel, axis=1)

    # effective form factor
    fe = np.sum(fafb, axis=1)

    pair_labels = [f"{atom_names[i]}-{atom_names[j]}" for i, j in all_pairs]

    info = dict(f0=f0, all_pairs=all_pairs, fafb=fafb, fe=fe, pair_labels=pair_labels)

    return S0, info

def simulate_scattering_batch(
    q: np.ndarray,
    xyz_traj: np.ndarray,
    scattering_ctx: dict,
):
    """
    Vectorized Debye scattering simulation.

    Parameters
    ----------
    xyz_traj:
        (n_traj, n_frames, N, 3)

    Returns
    -------
    S0:
        (n_traj, n_frames, nq)
    r_ab:
        (n_traj, n_frames, n_pairs)
    """

    all_pairs = scattering_ctx["all_pairs"]
    fafb = scattering_ctx["fafb"]
    self_scatter = scattering_ctx["self_scatter"]

    diff = (
        xyz_traj[:, :, all_pairs[:, 1], :]
        - xyz_traj[:, :, all_pairs[:, 0], :]
    )

    r_ab = np.linalg.norm(diff, axis=-1)

    qr = (
        q[None, None, :, None]
        * r_ab[:, :, None, :]
    )

    bessel = np.sinc(qr / np.pi)

    pair_sum = np.sum(
        fafb[None, None, :, :] * bessel,
        axis=-1,
    )

    S0 = (
        self_scatter[None, None, :]
        + 2 * pair_sum
    )

    return S0, r_ab

def bessel_zeros_spherical(n: int, k: int) -> np.ndarray:
    """
    Computes first k positive zeros of the spherical Bessel function
    j_n. (port of Adi's matlab code)
    """
    # Initial guess coefficients fitted for n=0 to 10000 (from MATLAB source)
    coeff1 = [0.411557013144507, 0.999986723293410,
              0.698028985524484, 1.06977507291468]
    exp1   = [0.335300369843979, 0.339671493811664]
 
    coeff2 = [1.93395115137444, 1.00007656297072,
              -0.805720018377132, 3.38764629174694]
    exp2   = [0.456215294517928, 0.388380341189200]
 
    coeff3 = [5.40770803992613, 1.00093850589418,
              2.66926179799040, -0.174925559314932]
    exp3   = [0.429702214054531, 0.633480051735955]
 
    zeros = np.zeros(k)
 
    def _guess1(n): return (coeff1[0] + coeff1[1]*n + coeff1[2]*(n+1)**exp1[0] + coeff1[3]*(n+1)**exp1[1])
    def _guess2(n): return (coeff2[0] + coeff2[1]*n + coeff2[2]*(n+1)**exp2[0] + coeff2[3]*(n+1)**exp2[1])
    def _guess3(n): return (coeff3[0] + coeff3[1]*n + coeff3[2]*(n+1)**exp3[0] + coeff3[3]*(n+1)**exp3[1])
 
    def halley(n_ord, x0):
        ITER_MAX = 100
        TOL = 1e4 * np.finfo(float).eps
        x = x0
        for _ in range(ITER_MAX):
            # scipy.special.spherical_jn handles x=0 and derivatives cleanly
            a = spherical_jn(n_ord,     x)
            b = spherical_jn(n_ord + 1, x)
            x2 = x * x
            denom = (2*b*b*x2
                     - a*b*x*(4*n_ord + 1)
                     + (n_ord*(n_ord+1) + x2) * a*a)
            if abs(denom) < 1e-300:
                break
            err = 2*a*x*(n_ord*a - b*x) / denom
            x -= err
            if abs(err) < TOL * abs(x):
                break
        return x
 
    zeros[0] = halley(n, _guess1(n))
    if k >= 2:
        zeros[1] = halley(n, _guess2(n))
    if k >= 3:
        zeros[2] = halley(n, _guess3(n))
    if k >= 4:
        for i in range(3, k):
            guess = 2*zeros[i-1] - zeros[i-2]
            zeros[i] = halley(n, guess)
    return zeros

def DBT(n: int, data: np.ndarray, q: np.ndarray, M: int=None):
    """
    Discrete Bessel Transform of order n (spherical version)
    params:
        - n: transform order, where 0 corresponds to isotropic scattering
        - data
        - q
        - M = number of output points, zero padding of len(q) < M
    """
    if M is None:
        M = len(q)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
        squeeze_out = True
    else:
        squeeze_out = False
    
    dq = q[1] - q[0]
    qp_end = q[0] + dq * (M - 1)

    BZ = bessel_zeros_spherical(n, M)

    # resampled grids using bessel zeros as node locations
    qgrid = qp_end * BZ[:M-1] / BZ[M-1]
    Rgrid = BZ[:M-1] / qp_end

    # resample data onto bessel-zero q grid
    in_range = (qgrid >= q[0]) & (qgrid <= q[-1])
    data_s = np.zeros((M-1, data.shape[1]))
    if np.any(in_range):
        # interp = Akima1DInterpolator(q, data, axis=0)
        interp = PchipInterpolator(q, data, axis=0, extrapolate=False)
        data_s[in_range, :] = interp(qgrid[in_range])

    # jacobian
    scalingFactor = qp_end**2 / BZ[M-1]

    # DBT kernel G
    bz_inner = BZ[:M-1]
    denom = BZ[M-1] * spherical_jn(n+1, bz_inner)**2
    arg = np.outer(bz_inner, bz_inner) / BZ[M-1]
    numer = 2 * spherical_jn(n, arg)
    G = numer / denom[:, np.newaxis]

    # apply jacobian 
    data_s = data_s * qgrid[:, np.newaxis]

    PD = G @ data_s * scalingFactor

    # fixing original shape
    if squeeze_out: 
        PD = PD[:, 0]
    
    return PD, Rgrid

#####################################################################
# L1 signal recovery stuff 
#####################################################################

def soft_threshold(V: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft-thresholding (real or complex)."""
    if np.isrealobj(V):
        return np.sign(V) * np.maximum(np.abs(V) - tau, 0)
    else:
        mag = np.abs(V)
        scale = np.maximum(1 - tau / np.maximum(mag, np.finfo(float).eps), 0)
        return scale * V
 
 
def decon_TVL1(dPD: np.ndarray,
               D: np.ndarray,
               lam: float = None,
               opts: dict = None):
    """
    L1-penalised least squares via ADMM (Lasso). Python version of Adi's Code. 
 
    Parameters
        - dPD: delta PD data
        - D: dictionary 
        - lam: L1 penalty
        - opts: dict of options
 
    Returns
        - dPD_TV: recovered coefficients
        - info: diagnostics dict
    """
    defaults = dict(normalizeColumns=True, lambdaFrac=1e-2,
                    rho=1.0, adaptiveRho=True, rhoUpdateInterval=10,
                    mu=10.0, tauRho=2.0, alpha=1.25,
                    maxIter=500, tolAbs=1e-6, tolRel=1e-4,
                    nonnegative=False, verbose=False,
                    debias=True, debiasRelTol=1e-3,
                    debiasAbsTol=0.0, debiasRidge=0.0,
                    storeDualProxy=True, trackObjective=False)
    if opts:
        defaults.update(opts)
    opts = defaults
 
    nObs, nCoef = D.shape
    orig_shape = dPD.shape
    trailing = orig_shape[1:]
    Y = dPD.reshape(nObs, -1)
    nRHS = Y.shape[1]
 
    # Column normalisation
    if opts['normalizeColumns']:
        colScale = np.sqrt(np.sum(np.abs(D)**2, axis=0))
        colScale[colScale == 0] = 1.0
        Dn = D / colScale[np.newaxis, :]
    else:
        colScale = np.ones(nCoef)
        Dn = D
 
    DtD = Dn.T @ Dn
    DtY = Dn.T @ Y
 
    if lam is None:
        lam = opts['lambdaFrac'] * np.max(np.abs(DtY))
 
    rho = opts['rho']
    I = np.eye(nCoef)
 
    def factor(rho_val):
        A = DtD + rho_val * I
        A = 0.5 * (A + A.T)
        return np.linalg.cholesky(A)       # lower triangular
 
    L = factor(rho)
 
    X = np.zeros((nCoef, nRHS))
    Z = np.zeros((nCoef, nRHS))
    U = np.zeros((nCoef, nRHS))
 
    nIter = 0
    for k in range(1, opts['maxIter'] + 1):
        nIter = k
        Zold = Z.copy()
 
        # X update (Cholesky solve)
        RHS = DtY + rho * (Z - U)
        X = np.linalg.solve(L.T, np.linalg.solve(L, RHS))
 
        # Over-relaxation
        Xhat = opts['alpha'] * X + (1 - opts['alpha']) * Zold
 
        # Z update (proximal / soft threshold)
        V = Xhat + U
        if opts['nonnegative']:
            Z = np.maximum(V - lam/rho, 0)
        else:
            Z = soft_threshold(V, lam/rho)
 
        # Dual update
        U = U + Xhat - Z
 
        # Residuals
        r_norm = np.linalg.norm(X - Z, 'fro')
        s_norm = rho * np.linalg.norm(Z - Zold, 'fro')
        eps_pri  = (np.sqrt(X.size) * opts['tolAbs']
                    + opts['tolRel'] * max(np.linalg.norm(X,'fro'),
                                           np.linalg.norm(Z,'fro')))
        eps_dual = (np.sqrt(X.size) * opts['tolAbs']
                    + opts['tolRel'] * rho * np.linalg.norm(U,'fro'))
 
        if opts['verbose'] and (k % 100 == 0 or k == 1):
            print(f"iter {k:4d}: r={r_norm:.3e}, s={s_norm:.3e}, "
                  f"eps_p={eps_pri:.3e}, eps_d={eps_dual:.3e}, rho={rho:.3g}")
 
        if r_norm < eps_pri and s_norm < eps_dual:
            break
 
        # Adaptive rho
        if opts['adaptiveRho'] and k % opts['rhoUpdateInterval'] == 0:
            rhoNew = rho
            if r_norm > opts['mu'] * s_norm:
                rhoNew = rho * opts['tauRho']
            elif s_norm > opts['mu'] * r_norm:
                rhoNew = rho / opts['tauRho']
            if rhoNew != rho:
                U = (rho / rhoNew) * U
                rho = rhoNew
                L = factor(rho)
 
    Xnorm = Z
    Xorig = Xnorm / colScale[:, np.newaxis]
    dPD_TV = Xorig.reshape((nCoef,) + trailing)
 
    info = dict(lam=lam, nIter=nIter, nObs=nObs, nCoef=nCoef, nRHS=nRHS,
                colScale=colScale)
 
    # Optional debias (LS refit on recovered support)
    if opts['debias']:
        Xdeb = np.zeros_like(Xorig)
        for j in range(nRHS):
            xj = Xorig[:, j]
            maxAbs = np.max(np.abs(xj))
            if maxAbs == 0:
                continue
            thresh = max(opts['debiasAbsTol'], opts['debiasRelTol'] * maxAbs)
            supp = np.where(np.abs(xj) > thresh)[0]
            if len(supp) == 0:
                continue
            Dj = D[:, supp]
            yj = Y[:, j]
            if opts['debiasRidge'] > 0:
                Aj = Dj.T @ Dj + opts['debiasRidge'] * np.eye(len(supp))
                cj = np.linalg.solve(Aj, Dj.T @ yj)
            else:
                cj, *_ = np.linalg.lstsq(Dj, yj, rcond=None)
            Xdeb[supp, j] = cj
        info['dPD_TV_debiased'] = Xdeb.reshape((nCoef,) + trailing)
 
    return dPD_TV, info


############################################
# Run the full pipeline
############################################

def _ensure_numpy(x):
    """
    Make sure the MATLAB stuff work as actual numpy arrays 
    """
    return np.array(x).squeeze()

def lorch(q):
    detector = np.ones_like(q)
    ndet = np.sum(detector > 0)
    lorch = np.sin(np.linspace(0, np.pi, ndet))
    detector[detector > 0] = lorch
    detector /= np.sum(detector)
    return detector

def run_pipeline(mat_path: str, output_path: str):
    """
    Scattering + signal recovery pipeline 
    """
    print("Loading .mat file...")
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Extract variables
    xyzt_mode = _ensure_numpy(mat["xyzt_mode"])
    atom_name = mat["atom_name"]
    t_common_fs = _ensure_numpy(mat["t_common_fs"])
    modeNumber = _ensure_numpy(mat["modeNumber"]).astype(int)

    nAtoms = len(atom_name)
    nTime = len(t_common_fs)
    nModes = len(modeNumber)

    # q grid
    dq = 0.1 
    q = np.arange(np.finfo(float).eps, 10 + dq, dq)

    # ground truth pair distances 
    allpairs_gt = np.array(list(combinations(range(nAtoms), 2)))
    diff_gt = (xyzt_mode[allpairs_gt[:, 1], :, :, :]
               - xyzt_mode[allpairs_gt[:, 0], :, :, :])
    Rab = np.sqrt(np.sum(diff_gt ** 2, axis=1))

    # simulate scattering
    print("Simulating X-ray scattering...")
    S0 = np.zeros((len(q), nTime, nModes))

    for nd in range(nModes):
        for nt in range(nTime):
            xyz = xyzt_mode[:, :, nt, nd]
            s, info = simulate_scattering(q, atom_name, xyz)
            S0[:, nt, nd] = s
    info_ref = info

    f0 = info_ref["f0"]
    all_pairs = info_ref["all_pairs"]
    fafb = info_ref["fafb"]
    fe = info_ref["fe"]
    pair_labels = info_ref["pair_labels"]

    # detector window
    detector = lorch(q)

    # background subtracted, normalized signal
    self_scattering = np.sum(f0**2, axis=1)
    dS0 = (S0 - self_scattering[:, np.newaxis, np.newaxis])
    dS0 *= detector[:, np.newaxis, np.newaxis]
    dS0 /= fe[:, np.newaxis, np.newaxis]

    # inverting via discrete bessel transform
    print("Inverting via Discrete Bessel Transform...")
    dR = 0.025
    N = int(np.ceil(np.pi / (dR * dq))) + 1 

    dPD = None
    Rvec = None

    for nd in range(nModes):
        PD_nd, Rvec = DBT(0, dS0[:, :, nd], q, N) 
        if dPD is None:
            dPD = np.zeros((len(Rvec), nTime, nModes))
        dPD[:, :, nd] = PD_nd

    # building NSK dictionary for ADMM 
    print("Building NSK dictionary...")
    nR = len(Rvec)
    nq = len(q)
    bes = np.zeros((nq, nR))
    for i, Ri in enumerate(Rvec):
        bes[:, i] = spherical_jn(0, q * Ri)
 
    # transform each NSK column through the same DBT
    D_dict, _ = DBT(0, bes * detector[:, np.newaxis], q, N) 

    # L1/TV deconvolution
    print("Running sparse L1/TV deconvolution/ADMM...")
    decon_opts = dict(normalizeColumns=True, lambdaFrac=1e-2,
                      rho=1.0, maxIter=500, tolRel=1e-3, tolAbs=1e-6,
                      adaptiveRho=True)
    lam = 1.0
 
    dPD_TV = None
    for nd in range(nModes):
        tv_nd, _ = decon_TVL1(dPD[:, :, nd], D_dict, lam, decon_opts)
        if dPD_TV is None:
            dPD_TV = np.zeros_like(dPD)
        dPD_TV[:, :, nd] = tv_nd
        print(f"mode {modeNumber[nd]} done")
    
    # Plotting
    print("Plotting...")
    cmap = plt.cm.turbo
 
    fig, axes = plt.subplots(nModes, 4,
                             figsize=(14, 3.5 * nModes),
                             constrained_layout=True)
    if nModes == 1:
        axes = axes[np.newaxis, :]
 
    pair_colors = ['tab:blue', 'tab:orange', 'tab:olive']
 
    for nd in range(nModes):
        mn = modeNumber[nd]
 
        # col 0 – S0 (q-space scattering)
        ax = axes[nd, 0]
        ax.imshow(S0[:, :, nd], aspect='auto', origin='lower', cmap=cmap,
                  extent=[t_common_fs[0], t_common_fs[-1], q[0], q[-1]])
        ax.set_ylim(0, 5)
        ax.set_title(f'$S_0$ mode {mn}')
        ax.set_ylabel('q (A^-1)')
        ax.set_xlabel('time (fs)')
 
        # col 1 – PD0 (real-space pair distribution)
        ax = axes[nd, 1]
        im = ax.imshow(dPD[:, :, nd], aspect='auto', origin='lower', cmap=cmap,
                       extent=[t_common_fs[0], t_common_fs[-1],
                                Rvec[0], Rvec[-1]])
        ax.set_ylim(1, 5)
        ax.set_title(f'$PD_0$ mode {mn}')
        ax.set_ylabel('R (A)')
        ax.set_xlabel('time (fs)')
        plt.colorbar(im, ax=ax)
 
        # col 2 – PD0 L1/TV deconvolved
        ax = axes[nd, 2]
        im = ax.imshow(dPD_TV[:, :, nd], aspect='auto', origin='lower', cmap=cmap,
                       extent=[t_common_fs[0], t_common_fs[-1],
                                Rvec[0], Rvec[-1]])
        ax.set_ylim(1, 5)
        ax.set_title(f'$PD_0$ mode {mn} L1/TV')
        ax.set_ylabel('R (A)')
        ax.set_xlabel('time (fs)')
        plt.colorbar(im, ax=ax)
 
        # col 3 – ground truth pair distances
        ax = axes[nd, 3]
        for p, lbl in enumerate(pair_labels):
            color = pair_colors[p % len(pair_colors)]
            ax.plot(t_common_fs, Rab[p, :, nd], color=color, label=lbl)
        ax.set_ylim(1, 5)
        ax.set_title(f'Ground Truth mode {mn}')
        ax.set_ylabel('R (A)')
        ax.set_xlabel('time (fs)')
        ax.legend(fontsize=7)
 
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    return fig


# -----------------------------------------------------------------------------------
# Computing the relevant trajectories and ground truth PD
# -----------------------------------------------------------------------------------

def generate_trajectory_batch(
    R0_flat: np.ndarray,   # (3N,)
    masses_amu: np.ndarray,   # (N,)
    V_vib: np.ndarray,   # (3N, n_modes)
    omegas_ASE: np.ndarray,   # (n_modes,)
    T_kelvin: float = 300.0,
    n_traj: int = 30,
    points_per_fast_cycle: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    time_fs  : (n_frames,)
    xyz_traj : (n_traj, n_frames, N, 3)
    """
    rng = np.random.default_rng(seed)
    N = len(R0_flat) // 3

    if len(omegas_ASE) == 0:
        raise ValueError("omegas_ASE is empty — no vibrational modes found.")

    periods_fs = (2 * np.pi / omegas_ASE) / _ase_units.fs
    t_total_fs = np.max(periods_fs)
    dt_fs = np.min(periods_fs) / points_per_fast_cycle
    time_fs = np.arange(0, t_total_fs + dt_fs, dt_fs)
    num_frames = len(time_fs)
    time_ASE = (time_fs * _ase_units.fs).reshape(-1, 1)

    # placeholder bc we're confused about this 
    A_k = np.sqrt(2 * _ase_units.kB * T_kelvin) / omegas_ASE
    inv_sqrt_m = 1.0 / np.sqrt(masses_amu)

    all_trajs = np.empty((n_traj, num_frames, N, 3), dtype=np.float32)
    for i in range(n_traj):
        phi = rng.uniform(0, 2 * np.pi, len(omegas_ASE))
        c_t = A_k * np.cos(omegas_ASE * time_ASE + phi)      # (frames, n_modes)
        dR = (V_vib * inv_sqrt_m[:, None]) @ c_t.T          # (3N, frames)
        all_trajs[i] = (R0_flat[:, None] + dR).T.reshape(num_frames, N, 3)

    return time_fs, all_trajs

def compute_ground_truth_pd(xyz: np.ndarray, atom_names: list, R_grid: np.ndarray):
    """
    Compute weighted pair distance histogram directly from trajectory.
    """
    all_pairs = np.array(list(combinations(range(len(atom_names)), 2)))
    diff = xyz[all_pairs[:, 1]] - xyz[all_pairs[:, 0]]
    r_ab = np.sqrt(np.sum(diff ** 2, axis=1))

    Z = np.array([ATOMIC_FORM_FACTORS[el]['Z'] for el in atom_names])
    pair_weights = Z[all_pairs[:, 0]] * Z[all_pairs[:, 1]]

    dR = R_grid[1] - R_grid[0]
    pd = np.zeros(len(R_grid))
    idx = np.floor(((r_ab - R_grid[0]) / dR)).astype(int)
    valid_idx = (idx >= 0) & (idx < len(R_grid))
    np.add.at(pd, idx[valid_idx], pair_weights[valid_idx])
    
    return pd

def compute_ground_truth_pd_batch(r_ab, pair_weights, R_grid):
    n_traj, n_frames, n_pairs = r_ab.shape
    n_R = len(R_grid)

    dR = R_grid[1] - R_grid[0]

    idx = np.floor((r_ab - R_grid[0]) / dR).astype(np.int32)
    valid = (idx >= 0) & (idx < n_R)

    pd = np.zeros((n_traj, n_frames, n_R), dtype=np.float32)

    flat_idx = idx[valid]
    flat_w = np.broadcast_to(pair_weights, r_ab.shape)[valid]

    traj_ids, frame_ids, _ = np.where(valid)

    np.add.at(
        pd,
        (traj_ids, frame_ids, flat_idx),
        flat_w
    )

    return pd

def process_molecule(
    npz_path: Path,
    out_dir: Path,
    dq: float = 0.1,
    qmax: float = 10.0,
    dR: float = 0.025,
    T_kelvin: float = 300.0,
    n_traj: int = 30,
    points_per_fast_cycle: int = 10):


    match = re.search(r'mol_(\d+)_physics', npz_path.stem)

    if match is None:
        return f"SKIP {npz_path.name}: cannot parse index"

    mol_idx = int(match.group(1))
    out_path = (out_dir/ f"mol_{mol_idx:06d}_pd.npz")

    if out_path.exists():
        return None

    data = np.load(npz_path)

    R0_flat = data["R_0"].ravel().astype(np.float64)
    masses_amu = data["masses_amu"].astype(np.float64)
    V_vib = data["V_vib"].astype(np.float64)
    omegas_ASE = data["omegas_ASE"].astype(np.float64)

    try:
        time_fs, xyz_traj = generate_trajectory_batch(
            R0_flat,
            masses_amu,
            V_vib,
            omegas_ASE,
            T_kelvin=T_kelvin,
            n_traj=n_traj,
            points_per_fast_cycle=points_per_fast_cycle,
            seed=mol_idx,
        )
    except ValueError as e:
        return f"SKIP {npz_path.name}: {e}"

    atom_names = get_atom_names(mol_idx)

    q = np.arange(np.finfo(float).eps, qmax + dq, dq)

    detector = lorch(q)

    M = int(np.ceil(np.pi / (dR * dq))) + 1

    ctx = build_scattering_context(atom_names, q)

    S0, r_ab = simulate_scattering_batch(q, xyz_traj, ctx)

    dS0 = (
        (S0 - ctx["self_scatter"][None, None, :])
        / ctx["fe"][None, None, :]
    )

    dS0 *= detector[None, None, :]

    # DBT expects:
    # (q, frames)

    n_traj_out, n_frames, nq = dS0.shape

    pd_limited = []
    R_grid = None

    for ti in range(n_traj_out):

        pd_l, R_grid = DBT(0, dS0[ti].T, q, M)
        pd_limited.append(pd_l.T.astype(np.float32))

    pd_limited = np.stack(pd_limited)
    pd_true = compute_ground_truth_pd_batch(r_ab, ctx["pair_weights"], R_grid)

    np.savez_compressed(
        out_path,

        pd_limited=pd_limited,
        pd_true=pd_true,

        V_vib=V_vib.astype(np.float32),
        omegas_ASE=omegas_ASE.astype(np.float32),

        R_grid=R_grid.astype(np.float32),
        time_fs=time_fs.astype(np.float32),

        mol_idx=np.array(mol_idx),
    )



if __name__ == "__main__":
    data_dir = Path("data/mol_data")
    out_dir = Path("data/trajectory_data")

    for npz_path in data_dir.glob("*.npz"):
        result = process_molecule(npz_path=npz_path, out_dir=out_dir)
        if result:
            print(result)

    # mat_path = "processing/DemoMatlabData/h2o_xyzt_minimal.mat"
    # output_path = "processing/figures/test.png"
    # fig = run_pipeline(mat_path, output_path)