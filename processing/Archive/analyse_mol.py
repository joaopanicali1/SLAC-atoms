#!/usr/bin/env python
"""
analyse_mol.py — static PD(R), ΔPD(R), S₀(q), and ΔPDF(R) from a mol_data .npz file.

ΔPD(R) = PD(R, t_end) − PD(R, 0): pair density at the end of the classical harmonic
trajectory minus the DFT equilibrium geometry (R_0). The trajectory spans one full
period of the slowest normal mode (ported from trajectory_pipeline_cluster.ipynb).

Usage
-----
  python processing/analyse_mol.py mol_data/mol_000080_physics.npz
  python processing/analyse_mol.py mol_data/mol_000080_physics.npz --T 300 --dR 0.02

Outputs (in PD_data/, or --outdir)
  mol_XXXXXX_PD_static.csv  : R [Å], PD [e²]             — equilibrium pair density
  mol_XXXXXX_PD_diff.csv    : R [Å], delta_PD_pct [%]    — ΔPD at end of trajectory
  mol_XXXXXX_SDS.csv        : q [Å⁻¹], S0_ground, S0_mean — scattering ground + traj mean
  mol_XXXXXX_dPDF.csv       : R [Å], delta_PDF            — ΔPDF at end of trajectory
  mol_XXXXXX_analysis.png   : 6-panel figure
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect
from ase.data import atomic_numbers

# ── Project paths ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DB_PATH      = os.path.join(PROJECT_ROOT, "data/train_4m", "data0000.aselmdb")

from ase import units as _ase_units

# ── Waasmaier-Kirfel / Cromer-Mann form-factor table ─────────────────────────
# Coefficients: [a1..a5, c0, b1..b5]  (ported from SDS_min.m)
_CM = {
"H":  [0.413048,0.294953,0.187491,0.080701,0.023736,4.9e-05,15.56995,32.39847,5.71140,61.88987,1.33412],
"He": [0.732354,0.753896,0.283819,0.190003,0.039139,0.000487,11.55392,4.59583,1.54630,26.46396,0.37752],
"Li": [0.974637,0.158472,0.811855,0.262416,0.790108,0.002542,4.33495,0.34245,97.10297,201.36383,1.40923],
"Be": [1.533712,0.638283,0.601052,0.106139,1.118414,0.002511,42.66208,0.59542,99.10650,0.15134,1.84309],
"B":  [2.085185,1.064580,1.062788,0.140515,0.641784,0.003823,23.49407,1.13789,61.23898,0.11489,0.39904],
"C":  [2.657506,1.078079,1.490909,-4.241070,0.713791,4.298983,14.78076,0.77678,42.08684,-0.00029,0.23954],
"N":  [11.893780,3.277479,1.858092,0.858927,0.912985,-11.804902,0.00016,10.23272,30.34469,0.65607,0.21729],
"O":  [2.960427,2.508818,0.637853,0.722838,1.142756,0.027014,14.18226,5.93686,0.11273,34.95848,0.39024],
"F":  [3.511943,2.772244,0.678385,0.915159,1.089261,0.032557,10.68786,4.38047,0.09398,27.25520,0.31307],
"Ne": [4.183749,2.905726,0.520513,1.135641,1.228065,0.025576,8.17546,3.25254,0.06330,21.81391,0.22495],
"Na": [4.910127,3.081783,1.262067,1.098938,0.560991,0.079712,3.28143,9.11918,0.10276,132.01395,0.40588],
"Mg": [4.708971,1.194814,1.558157,1.170413,3.239403,0.126842,4.87521,108.50608,0.11152,48.29241,1.92817],
"Al": [4.730796,2.313951,1.541980,1.117564,3.154754,0.139509,3.62893,43.05117,0.09596,108.93239,1.55592],
"Si": [5.275329,3.191038,1.511514,1.356849,2.519114,0.145073,2.63334,33.73073,0.08112,86.28864,1.17009],
"P":  [1.950541,4.146930,1.494560,1.522042,5.729711,0.155233,0.90814,27.04495,0.07128,67.52019,1.98117],
"S":  [6.372157,5.154568,1.473732,1.635073,1.209372,0.154722,1.51435,22.09253,0.06137,55.44518,0.64693],
"Cl": [1.446071,6.870609,6.151801,1.750347,0.634168,0.146773,0.05236,1.19317,18.34342,46.39840,0.40101],
"Ar": [7.188004,6.638454,0.454180,1.929593,1.523654,0.265954,0.95622,15.33988,15.33986,39.04382,0.06241],
"K":  [8.163991,7.146945,1.070140,0.877316,1.486434,0.253614,12.81632,0.80895,210.32701,39.59765,0.05282],
"Ca": [8.593655,1.477324,1.436254,1.182839,7.113258,0.196255,10.46064,0.04189,81.39038,169.84784,0.68810],
"Sc": [1.476566,1.487278,1.600187,9.177463,7.099750,0.157765,53.13102,0.03533,137.31949,9.09803,0.60210],
"Ti": [9.818524,1.522646,1.703101,1.768774,7.082555,0.102473,8.00188,0.02976,39.88542,120.15800,0.53241],
"V":  [10.473575,1.547881,1.986381,1.865616,7.056250,0.067744,7.08194,0.02604,31.90967,108.02284,0.47488],
"Cr": [11.007069,1.555477,2.985293,1.347855,7.034779,0.065510,6.36628,0.02399,23.24484,105.77450,0.42937],
"Mn": [11.709542,1.733414,2.673141,2.023368,7.003180,-0.147293,5.59712,0.01780,21.78842,89.51791,0.38305],
"Fe": [12.311098,1.876623,3.066177,2.070451,6.975185,-0.304931,5.00942,0.01446,18.74304,82.76788,0.34651],
"Co": [12.914510,2.481908,3.466894,2.106351,6.960892,-0.936572,4.50714,0.00913,16.43813,76.98732,0.31442],
"Ni": [13.521865,6.947285,3.866028,2.135900,4.284731,-2.762697,4.07728,0.28676,14.62263,71.96608,0.00444],
"Cu": [14.014192,4.784577,5.056806,1.457971,6.932996,-3.254477,3.73828,0.00374,13.03498,72.55479,0.26567],
"Zn": [14.741002,6.907748,4.642337,2.191766,38.424042,-36.915829,3.38823,0.24332,11.90369,63.31213,0.00040],
"Ga": [15.758946,6.841123,4.121016,2.714681,2.395246,-0.847395,3.12175,0.22606,12.48220,66.20362,0.00724],
"Ge": [16.540613,1.567900,3.727829,3.345098,6.785079,0.018726,2.86662,0.01220,13.43216,58.86605,0.21097],
"As": [17.025642,4.503441,3.715904,3.937200,6.790175,-2.984117,2.59774,0.00301,14.27212,50.43800,0.19302],
"Se": [17.354071,4.653248,4.259489,4.136455,6.749163,-3.160982,2.34979,0.00255,15.57946,45.18120,0.17743],
"Br": [17.550570,5.411882,3.937180,3.880645,6.707793,-2.492088,2.11923,16.55718,0.00248,42.16401,0.16212],
"Kr": [17.655279,6.848105,4.171004,3.446760,6.685200,-2.810592,1.90823,16.60624,0.00160,39.91747,0.14690],
"Rb": [8.123134,2.138042,6.761702,1.156051,17.679546,1.139548,15.14239,33.54267,0.12937,224.13251,1.71337],
"Sr": [17.730219,9.795867,6.099763,2.620025,0.600053,1.140251,1.56306,14.31087,0.12057,135.77132,0.12057],
"Y":  [17.792040,10.253252,5.714949,3.170516,0.918251,1.131787,1.42969,13.13282,0.11217,108.19703,0.11217],
"Zr": [17.859772,10.911038,5.821115,3.512513,0.746965,1.124859,1.31069,12.31929,0.10435,91.77754,0.10435],
"Nb": [17.958399,12.063054,5.007015,3.287667,1.531019,1.123452,1.21159,12.24669,0.09862,75.01195,0.09862],
"Mo": [6.236218,17.987711,12.973127,3.451426,0.210899,1.108770,0.09078,1.10831,11.46872,66.68415,0.09078],
"Tc": [17.840963,3.428236,1.373012,12.947364,6.335469,1.074784,1.00573,41.90138,119.32054,9.78154,0.08339],
"Ru": [6.271624,17.906738,14.123269,3.746008,0.908235,1.043992,0.07704,0.92822,9.55535,35.86068,123.55225],
"Rh": [6.216648,17.919739,3.854252,0.840326,15.173498,0.995452,0.07079,0.85612,33.88948,121.68669,9.02952],
"Pd": [6.121511,4.784063,16.631683,4.318258,13.246773,0.883099,0.06255,0.78403,8.75139,34.48998,0.78403],
"Ag": [6.073874,17.155437,4.173344,0.852238,17.988686,0.756603,0.05533,7.89651,28.44374,110.37611,0.71681],
"Cd": [6.080986,18.019468,4.018197,1.303510,17.974669,0.603504,0.04899,7.27365,29.11928,95.83121,0.66123],
"In": [6.196477,18.816183,4.050479,1.638929,17.962912,0.333097,0.04207,6.69567,31.00979,103.28435,0.61071],
"Sn": [19.325171,6.281571,4.498866,1.856934,17.917318,0.119024,6.11810,0.03692,32.52905,95.03719,0.56565],
"Sb": [5.394956,6.549570,19.650681,1.827820,17.867832,-0.290506,33.32652,0.03097,5.56493,87.13097,0.52399],
"Te": [6.660302,6.940756,19.847015,1.557175,17.802427,-0.806668,33.03165,0.02575,5.06555,84.10162,0.48766],
"I":  [19.884502,6.736593,8.110516,1.170953,17.548716,-0.448811,4.62859,0.02775,31.84910,84.40639,0.46355],
"Xe": [19.978920,11.774945,9.332182,1.244749,17.737501,-6.065902,4.14336,0.01014,28.79620,75.28069,0.41362],
"Cs": [17.418674,8.314444,10.323193,1.383834,19.876251,-2.322802,0.39983,0.01687,25.60583,233.33968,3.82692],
"Ba": [19.747343,17.368477,10.465718,2.592602,11.003653,-5.183497,3.48182,0.37122,21.22664,173.83427,0.01072],
"La": [19.966019,27.329655,11.018425,3.086696,17.335455,-21.745489,3.19741,0.00345,19.95549,141.38197,0.34182],
"Ce": [17.355122,43.988499,20.546650,3.130670,11.353665,-38.386017,0.32837,0.00205,3.08820,134.90765,18.83296],
"Pr": [21.551311,17.161730,11.903859,2.679103,9.564197,-3.871068,2.99568,0.31249,17.71671,152.19283,0.01047],
"Nd": [17.331244,62.783924,12.160097,2.663483,22.229950,-57.189842,0.30027,0.00132,17.02600,148.74899,2.91027],
"Sm": [23.700363,23.072214,12.777782,2.684217,17.204367,-17.452166,2.68954,0.00349,15.49544,139.86247,0.27454],
"Eu": [17.186195,37.156837,13.103387,2.707246,24.419271,-31.586687,0.26168,0.00200,14.78736,134.81630,2.58188],
"Gd": [24.898117,17.104952,13.222581,3.266152,48.995213,-43.505684,2.43503,0.24696,13.99633,110.86309,0.00138],
"Tb": [25.910013,32.344139,13.765117,2.751404,17.064405,-26.851971,2.37391,0.00203,13.48197,125.83651,0.23692],
"Dy": [26.671785,88.687576,14.064545,2.768497,17.067781,-83.279831,2.28259,0.00067,12.92023,121.93719,0.22553],
"Ho": [27.151900,16.999819,14.059334,3.386979,46.546471,-41.165253,2.16966,0.21541,12.21315,100.50678,0.00121],
"Er": [28.174887,82.493271,14.624002,2.802756,17.018515,-77.135223,2.12100,0.00064,11.91526,114.52994,0.20752],
"Tm": [28.925894,76.173798,14.904704,2.814812,16.998117,-70.839813,2.04620,0.00066,11.46538,111.41198,0.19938],
"Yb": [29.676760,65.624069,15.160854,2.830288,16.997850,-60.313812,1.97763,0.00072,11.04462,108.13915,0.19211],
"Lu": [30.122866,15.099346,56.314899,3.540980,16.943729,-51.049416,1.88309,10.34276,0.00078,89.55925,0.18385],
"Hf": [30.617033,15.145351,54.933548,4.096253,16.896156,-49.719837,1.79561,9.93447,0.00074,76.18971,0.17591],
"Ta": [31.066359,15.341823,49.278297,4.577665,16.828321,-44.119026,1.70873,9.61846,0.00076,66.34620,0.16800],
"W":  [31.507900,15.682498,37.960129,4.885509,16.792112,-32.864574,1.62949,9.44645,0.00090,59.98068,0.16080],
"Re": [31.888456,16.117104,42.390297,5.211669,16.767591,-37.412682,1.54924,9.23347,0.00069,54.51637,0.15282],
"Os": [32.210297,16.678440,48.559906,5.455839,16.735533,-43.677956,1.47353,9.04970,0.00052,50.21020,0.14577],
"Ir": [32.004436,1.975454,17.070105,15.939454,5.990003,4.018893,1.35377,81.01418,0.12809,7.66120,26.65940],
"Pt": [31.273891,18.445440,17.063745,5.555933,1.575270,4.050394,1.31699,8.79715,0.12474,40.17799,1.31700],
"Au": [16.777390,19.317156,32.979683,5.595453,10.576854,-6.279078,0.12274,8.62157,1.25690,38.00882,0.00060],
"Hg": [16.838990,20.023823,28.428564,5.881564,4.714706,4.076478,0.11591,8.25693,1.19525,39.24723,1.19525],
"Tl": [16.630795,19.386616,32.808571,1.747191,6.356862,4.066939,0.11070,7.18140,1.11973,90.66026,26.01498],
"Pb": [16.419567,32.738590,6.530247,2.342742,19.916475,4.049824,0.10550,1.05505,25.02589,80.90659,6.66445],
"Bi": [16.282274,32.725136,6.678302,2.694750,20.576559,4.040914,0.10118,1.00229,25.71415,77.05755,6.29188],
"Th": [15.784024,32.454899,21.849222,4.239077,11.736191,3.922533,0.07707,0.73514,4.09798,109.46411,20.51214],
"U":  [15.679275,32.824306,13.660459,3.687261,22.279434,3.854444,0.07121,0.68118,18.23616,112.50004,3.93033],
}


def _cm_form_factor(q, symbol):
    """Cromer-Mann X-ray form factor for one element at q values [Å⁻¹]. Returns (nq,)."""
    c = _CM.get(symbol)
    if c is None:
        return np.full(len(q), float(atomic_numbers.get(symbol, 1)))
    a  = np.array(c[:5])
    c0 = c[5]
    b  = np.array(c[6:])
    k2 = (q / (4 * np.pi)) ** 2          # (nq,)
    return np.exp(-k2[:, None] * b) @ a + c0  # (nq,)


def _pair_indices_and_weights(names):
    """Return (ii, jj, w) for all unique pairs, w = Z_a * Z_b."""
    N = len(names)
    Z = np.array([atomic_numbers[s] for s in names], dtype=float)
    ii, jj = np.tril_indices(N, k=-1)
    return ii, jj, Z[ii] * Z[jj]


def compute_PD(xyz_frames, names, dR=0.05, R_edges=None):
    """
    IAM-weighted pair density.
    xyz_frames: (N,3) or (N,3,T)
    Returns R [Å], PD (nR, T), R_edges used.
    """
    if xyz_frames.ndim == 2:
        xyz_frames = xyz_frames[:, :, None]
    N, _, T = xyz_frames.shape

    ii, jj, w = _pair_indices_and_weights(names)
    Rab = np.sqrt(np.sum((xyz_frames[ii] - xyz_frames[jj]) ** 2, axis=1))  # (nPair, T)

    if R_edges is None:
        R_edges = np.arange(0, Rab.max() + 2 * dR, dR)
    R = 0.5 * (R_edges[:-1] + R_edges[1:])
    nR = len(R)

    PD = np.zeros((nR, T))
    for t in range(T):
        PD[:, t], _ = np.histogram(Rab[:, t], bins=R_edges, weights=w)

    return R, PD, R_edges


def generate_dynamic_trajectory(R0_flat, masses_amu, V_vib, omegas_ASE,
                                T_kelvin=300.0, points_per_fast_cycle=10, seed=42):
    """
    Classical harmonic trajectory — ported from trajectory_pipeline_cluster.ipynb.

    Duration  = period of the slowest mode (full oscillation cycle).
    Time step = period of the fastest mode / points_per_fast_cycle.
    Amplitude per mode k: A_k = sqrt(2 kB T) / ω_k  (classical equipartition).
    Random initial phases ensure a representative phase-space snapshot.

    Returns (time_grid_fs [num_frames,], xyz_traj [num_frames, N, 3]).
    """
    rng = np.random.default_rng(seed)
    N   = len(R0_flat) // 3

    periods_fs   = (2 * np.pi / omegas_ASE) / _ase_units.fs
    t_total_fs   = np.max(periods_fs)
    dt_fs        = np.min(periods_fs) / points_per_fast_cycle
    time_grid_fs = np.arange(0, t_total_fs + dt_fs, dt_fs)
    num_frames   = len(time_grid_fs)

    time_ASE = (time_grid_fs * _ase_units.fs).reshape(-1, 1)  # (num_frames, 1)

    # Classical amplitude: A_k = sqrt(2 kB T) / ω_k
    A_k = np.sqrt(2 * _ase_units.kB * T_kelvin) / omegas_ASE  # (n_modes,)
    phi = rng.uniform(0, 2 * np.pi, len(omegas_ASE))

    # Normal coordinate displacement: Q_k(t) = A_k cos(ω_k t + φ_k)
    c_t = A_k * np.cos(omegas_ASE * time_ASE + phi)  # (num_frames, n_modes)

    # Cartesian: Δr(t) = (V_vib / √m) @ Q(t)
    inv_sqrt_m = 1.0 / np.sqrt(masses_amu)
    delta_R    = (V_vib * inv_sqrt_m[:, None]) @ c_t.T  # (3N, num_frames)

    xyz_traj = (R0_flat[:, None] + delta_R).T.reshape(num_frames, N, 3)

    print(f"  Slowest mode: {np.max(periods_fs):.2f} fs  |  "
          f"Fastest mode: {np.min(periods_fs):.2f} fs  |  "
          f"{num_frames} frames  (dt = {dt_fs:.2f} fs)")

    return time_grid_fs, xyz_traj


def compute_S0(q_arr, names, xyz,
               q_min=None, q_max=None,
               wavelength=None, two_theta_max_deg=None,
               window=None):
    """
    Isotropic Debye X-ray scattering S₀(q).
    S₀(q) = Σ_i f_i(q)² + 2 Σ_{i<j} f_i(q) f_j(q) j₀(q R_ij)
    xyz: (N, 3), q_arr: (nq,)  [Å⁻¹]

    Resolution constraints (all optional):
      q_min, q_max          : clip output to this q window [Å⁻¹]
      wavelength            : X-ray wavelength λ [Å]
      two_theta_max_deg     : detector half-angle 2θ_max [deg]
                              (if both wavelength and two_theta_max_deg given,
                               q_max = (4π/λ) sin(θ_max) overrides the kwarg)
      window                : None or 'hanning' — smooth taper near q_max to
                              suppress truncation ripples in the sine FT

    Returns (q_kept, S0_kept).
    """
    N  = len(names)
    f  = np.column_stack([_cm_form_factor(q_arr, s) for s in names])  # (nq, N)
    S0 = np.sum(f ** 2, axis=1)                                        # self terms

    ii, jj = np.tril_indices(N, k=-1)
    Rab = np.sqrt(np.sum((xyz[ii] - xyz[jj]) ** 2, axis=1))           # (nPair,)
    qR  = q_arr[:, None] * Rab[None, :]                                # (nq, nPair)
    j0  = np.where(qR > 1e-10, np.sin(qR) / qR, 1.0)
    S0 += 2 * np.sum(f[:, ii] * f[:, jj] * j0, axis=1)

    # ── Resolution-driven q-cutoff: q_max = (4π/λ) sin(θ_max) ────────────────
    if wavelength is not None and two_theta_max_deg is not None:
        theta_max = np.deg2rad(two_theta_max_deg) / 2.0
        q_max = (4 * np.pi / wavelength) * np.sin(theta_max)

    # ── Window selection ────────────────────────────────────────────────────
    lo = q_min if q_min is not None else q_arr.min()
    hi = q_max if q_max is not None else q_arr.max()
    keep = (q_arr >= lo) & (q_arr <= hi)
    q_out  = q_arr[keep]
    S0_out = S0[keep]

    # ── Optional smooth taper near q_max (Hanning half-window on upper end) ──
    if window == "hanning" and len(q_out) > 1:
        # half-Hanning: 1 at q_min, 0 at q_max
        u = (q_out - q_out.min()) / (q_out.max() - q_out.min())
        taper = 0.5 * (1 + np.cos(np.pi * u))   # cos² lobe, =1 at lo, =0 at hi
        S0_out = S0_out * taper

    return q_out, S0_out


def compute_difference_signal(S0_excited, S0_ground, fraction_excited=1.0):
    """
    Differential X-ray scattering signal.

      ΔS(q) = (1 / γ) · [S₀(q, excited-mix) − S₀(q, ground)]

    Both inputs must be on the same q-grid. fraction_excited (γ) rescales for the
    fact that only a fraction of molecules in the beam are pumped; pass γ=1 if
    your S0_excited is already a pure-excited-state signal (the default for our
    classical-trajectory snapshot).

    Returns ΔS on the same q-grid.
    """
    if S0_excited.shape != S0_ground.shape:
        raise ValueError("S0_excited and S0_ground must have the same shape.")
    if fraction_excited <= 0:
        raise ValueError("fraction_excited must be > 0.")
    return (S0_excited - S0_ground) / fraction_excited


def compute_differential_PDF(q_arr, delta_S, names,
                             r_grid=None, window="hanning"):
    """
    Differential pair distribution function ΔP(r) by sine Fourier transform.

      ΔsM(q) = q · ΔS(q) / Σ_i f_i(q)²
      ΔP(r) = (2/π) ∫₀^{q_max} ΔsM(q) sin(q r) dq

    Steps:
      1. Build the modified differential signal ΔsM(q) — multiply by q,
         normalize by the atomic self-term to flatten the f(q)-decay envelope.
      2. Optionally taper near q_max (Hanning) to reduce truncation ripples.
      3. Numerically integrate the sine FT on a user-supplied r-grid.

    Inputs
      q_arr   : (nq,) [Å⁻¹], assumed sorted ascending
      delta_S : (nq,) — ΔS(q), same grid as q_arr
      names   : list of atom symbols (for the self-term denominator)
      r_grid  : (nR,) [Å]; if None, uses 0..2π/Δq with reasonable spacing
      window  : None or 'hanning' (applied to ΔsM(q) before the FT)

    Returns (r_grid, delta_PDF).
    """
    q_arr   = np.asarray(q_arr, dtype=float)
    delta_S = np.asarray(delta_S, dtype=float)
    if q_arr.shape != delta_S.shape:
        raise ValueError("q_arr and delta_S must have the same shape.")

    # Self-term denominator Σ_i f_i(q)² on the input q-grid
    f = np.column_stack([_cm_form_factor(q_arr, s) for s in names])    # (nq, N)
    self_term = np.sum(f ** 2, axis=1)                                  # (nq,)

    # ΔsM(q) = q · ΔS(q) / Σ f_i²(q)  — guard against q=0
    sM = np.where(self_term > 0, q_arr * delta_S / self_term, 0.0)

    if window == "hanning" and len(q_arr) > 1:
        u = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min())
        sM = sM * 0.5 * (1 + np.cos(np.pi * u))     # half-Hanning, 1→0

    # Default r-grid spans up to a Nyquist-like bound
    if r_grid is None:
        dq    = np.mean(np.diff(q_arr))
        r_max = np.pi / dq                          # Nyquist: r_max = π/Δq
        r_grid = np.linspace(0.01, min(r_max, 15.0), 600)

    # Sine FT by trapezoid: ΔP(r) = (2/π) ∫ ΔsM(q) sin(q r) dq
    # Integrand shape: (nR, nq)
    integrand = sM[None, :] * np.sin(np.outer(r_grid, q_arr))
    delta_PDF = (2.0 / np.pi) * np.trapezoid(integrand, q_arr, axis=1)

    return r_grid, delta_PDF



def parse_sample_idx(path):
    m = re.search(r"mol_(\d+)_physics", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse sample_idx from: {path}")
    return int(m.group(1))


def get_atom_names(sample_idx):
    db  = connect(DB_PATH)
    row = db.get(int(sample_idx) + 1)
    return list(row.toatoms().get_chemical_symbols())


def hill_formula(names):
    """
    Return a matplotlib math-mode string in Hill notation (C first, H second,
    then remaining elements alphabetically), with subscripts for counts > 1.
    e.g. ['C','C','N','H','H','H','Cl'] → '$C_2H_3ClN$'
    """
    from collections import Counter
    counts = Counter(names)
    order = []
    for el in ["C", "H"]:
        if el in counts:
            order.append(el)
    for el in sorted(counts):
        if el not in ("C", "H"):
            order.append(el)
    parts = []
    for el in order:
        n = counts[el]
        parts.append(el if n == 1 else f"{el}_{{{n}}}")
    return "$" + "".join(parts) + "$"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path",                                    help="Path to mol_XXXXXX_physics.npz")
    parser.add_argument("--T",                    type=float, default=300,  help="Temperature in K (default 300)")
    parser.add_argument("--points-per-fast-cycle",type=int,   default=10,   help="Frames per fastest-mode period (default 10)")
    parser.add_argument("--dR",                   type=float, default=0.005, help="PD bin width Å (default 0.05)")
    parser.add_argument("--q-max",                type=float, default=15, help="Max q Å⁻¹ (default 10)")
    parser.add_argument("--q-min",                type=float, default=0, help="Min q Å⁻¹ (default 0.01)")
    parser.add_argument("--wavelength",           type=float, default=None, help="X-ray wavelength λ [Å]; if set with --two-theta-max overrides --q-max")
    parser.add_argument("--two-theta-max",        type=float, default=None, help="Detector half-angle 2θ_max [deg]; with --wavelength sets q_max = (4π/λ) sin(θ_max)")
    parser.add_argument("--window",               default="hanning", choices=["none", "hanning"], help="Taper near q_max for ΔS / ΔPDF (default hanning)")
    parser.add_argument("--gamma",                type=float, default=1.0,  help="Excited-state fraction γ for ΔS rescaling (default 1.0)")
    parser.add_argument("--outdir",               default=None,             help="Output directory")
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    data       = np.load(args.npz_path)
    R0_flat    = data["R_0"]
    masses_amu = data["masses_amu"]
    V_vib      = data["V_vib"]
    omegas_ASE = data["omegas_ASE"]

    N    = len(R0_flat) // 3
    xyz0 = R0_flat.reshape(N, 3)

    sample_idx = parse_sample_idx(args.npz_path)
    names      = get_atom_names(sample_idx)
    formula    = hill_formula(names)

    print(f"mol_{sample_idx:06d}  {formula}  ({N} atoms, {len(omegas_ASE)} modes)")
    print(f"T={args.T} K  dR={args.dR} Å")

    # ── Classical harmonic trajectory ─────────────────────────────────────────
    # ΔPD(R,τ) = PD(R,τ) − PD(R,0) 
    # PD(R,0) = static equilibrium; PD(R,τ) = trajectory snapshot at τ
    print("Generating classical harmonic trajectory...")
    time_grid_fs, xyz_traj = generate_dynamic_trajectory(
        R0_flat, masses_amu, V_vib, omegas_ASE,
        T_kelvin=args.T, points_per_fast_cycle=args.points_per_fast_cycle,
    )
    num_frames = len(time_grid_fs)

    # ── PD — shared R_edges from max range across static + all traj frames ───
    print("Computing PD(R)...")
    ii, jj, _ = _pair_indices_and_weights(names)
    static_Rab = np.sqrt(np.sum((xyz0[ii] - xyz0[jj]) ** 2, axis=1))
    traj_Rab   = np.sqrt(np.sum(
        (xyz_traj[:, ii, :] - xyz_traj[:, jj, :]) ** 2, axis=2
    ))  # (num_frames, nPair)
    R_edges = np.arange(0, max(static_Rab.max(), traj_Rab.max()) + 2 * args.dR, args.dR)

    R, PD_static, _ = compute_PD(xyz0, names, dR=args.dR, R_edges=R_edges)
    PD_static_1d    = PD_static[:, 0]

    # Compute PD for every trajectory frame
    xyz_traj_transposed = xyz_traj.transpose(1, 2, 0)  # (N, 3, num_frames)
    _, PD_traj, _ = compute_PD(xyz_traj_transposed, names, dR=args.dR, R_edges=R_edges)

    delta_PD_all = PD_traj - PD_static_1d[:, None]            # (nR, num_frames)
    PD_max         = PD_static_1d.max()
    delta_PD_pct = delta_PD_all[:, -1] / PD_max * 100           # ΔPD (%) at end of trajectory

    # ── S₀(q) — ground state ────────────────────────────────────────────────
    # q-window constrains S0 to the measurable range. Hanning taper is NOT
    # applied here — it belongs in the sine FT step to suppress truncation
    # ripples. The raw S0(q) saved here is what an experimentalist would measure.
    print("Computing S₀(q)...")
    q_dense = np.linspace(args.q_min, args.q_max, 600)
    win     = None if args.window == "none" else args.window

    q_arr, S0_ground = compute_S0(
        q_dense, names, xyz0,
        q_min=args.q_min, q_max=args.q_max,
        wavelength=args.wavelength, two_theta_max_deg=args.two_theta_max,
        window=None,
    )

    if args.wavelength is not None and args.two_theta_max is not None:
        print(f"  λ = {args.wavelength} Å, 2θ_max = {args.two_theta_max}°  →  q_max = {q_arr.max():.3f} Å⁻¹")
    print(f"  q-range used: [{q_arr.min():.3f}, {q_arr.max():.3f}] Å⁻¹  ({len(q_arr)} points)")

    # ── S₀(q, τ), ΔS(q, τ), ΔPDF(R) — computed across full trajectory ──────
    print("Computing ΔPDF(R) and ΔPDF(R, τ) over trajectory...")
    f_q     = np.column_stack([_cm_form_factor(q_arr, s) for s in names])  # (nq, N)
    st_q    = np.sum(f_q ** 2, axis=1)                                      # (nq,)
    fp_q    = f_q[:, ii] * f_q[:, jj]                                      # (nq, nPair)

    batch   = max(1, int(200e6 / (len(q_arr) * len(ii) * 8)))  # frames per batch
    S0_all  = np.empty((len(q_arr), num_frames))
    for b0 in range(0, num_frames, batch):
        b1    = min(b0 + batch, num_frames)
        qR_b  = q_arr[:, None, None] * traj_Rab[b0:b1].T[None, :, :]  # (nq, nPair, bs)
        j0_b  = np.where(np.abs(qR_b) > 1e-10, np.sin(qR_b) / qR_b, 1.0)
        S0_all[:, b0:b1] = st_q[:, None] + 2 * np.einsum("qp,qpt->qt", fp_q, j0_b)

    dS_all  = (S0_all - S0_ground[:, None]) / args.gamma          # (nq, num_frames)
    S0_mean = S0_all.mean(axis=1)                                  # (nq,) time-averaged

    sM_all  = np.where(st_q[:, None] > 0,
                       q_arr[:, None] * dS_all / st_q[:, None], 0.0)
    if win == "hanning":
        u       = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min())
        sM_all *= (0.5 * (1 + np.cos(np.pi * u)))[:, None]

    dq      = np.diff(q_arr).mean()
    trap_w  = np.ones(len(q_arr)); trap_w[0] = trap_w[-1] = 0.5
    sin_mat = np.sin(np.outer(R, q_arr))                          # (nR, nq)

    # ΔPDF at end of trajectory
    delta_PDF = (2 / np.pi) * dq * (sin_mat @ (sM_all[:, -1] * trap_w))

    # Full ΔPDF(R, τ) heatmap
    rPDF_traj = (2 / np.pi) * dq * (sin_mat @ (sM_all * trap_w[:, None]))

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    outdir = args.outdir or os.path.join(PROJECT_ROOT, "PD_data")
    os.makedirs(outdir, exist_ok=True)
    stem   = f"mol_{sample_idx:06d}"

    paths = {
        "static": os.path.join(outdir, f"{stem}_PD_static.csv"),
        "diff":   os.path.join(outdir, f"{stem}_PD_diff.csv"),
        "sds":    os.path.join(outdir, f"{stem}_SDS.csv"),
        "dPDF":   os.path.join(outdir, f"{stem}_dPDF.csv"),
        "fig":    os.path.join(outdir, f"{stem}_analysis.png"),
    }

    pd.DataFrame({"R_Ang": R, "PD": PD_static_1d}).to_csv(paths["static"], index=False)
    pd.DataFrame({"R_Ang": R, "delta_PD_pct": delta_PD_pct}).to_csv(paths["diff"], index=False)
    pd.DataFrame({"q_inv_Ang": q_arr, "S0_ground": S0_ground,
                  "S0_mean": S0_mean}).to_csv(paths["sds"], index=False)
    pd.DataFrame({"R_Ang": R, "delta_PDF": delta_PDF}).to_csv(paths["dPDF"], index=False)

    for label, p in paths.items():
        if label != "fig":
            print(f"Saved: {p}")

    # ── Plot (2 rows × 3 cols) ───────────────────────────────────────────────
    # Col 0: static/ground     Col 1: end-of-trajectory 1D     Col 2: time evolution 2D
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # (0,0) Static PD(R)
    ax = axes[0, 0]
    ax.plot(R, PD_static_1d, "k-", lw=1.5)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("Weighted pair density (e²)")
    ax.set_title("Static PD(R)")
    ax.set_xlim(0, R.max())

    # (0,1) ΔPD at end of trajectory vs equilibrium
    ax = axes[0, 1]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(R, delta_PD_pct, color="steelblue", lw=1.5,
            label=f"T = {args.T} K  (t = T)")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("ΔI (%)")
    ax.set_title("ΔPD  (end of trajectory)")
    ax.set_xlim(0, R.max())
    ax.legend(fontsize=8)

    # (0,2) ΔPD(R, τ) heatmap
    ax = axes[0, 2]
    delta_pct_2d = delta_PD_all / PD_max * 100
    vmax_pd = np.percentile(np.abs(delta_pct_2d), 99)
    im = ax.imshow(
        delta_pct_2d, aspect="auto", origin="lower",
        extent=[time_grid_fs[0], time_grid_fs[-1], R[0], R[-1]],
        cmap="RdBu_r", vmin=-vmax_pd, vmax=vmax_pd,
    )
    ax.set_xlabel("τ (fs)")
    ax.set_ylabel("R (Å)")
    ax.set_title("ΔPD(R, τ)  (%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔI (%)")

    # (1,0) S₀(q) ground vs time-averaged
    ax = axes[1, 0]
    ax.plot(q_arr, S0_ground, color="firebrick", lw=1.5, label="ground (t=0)")
    ax.plot(q_arr, S0_mean,   color="navy",      lw=1.0, ls="--",
            label="time-averaged")
    ax.set_xlabel("q (Å⁻¹)")
    ax.set_ylabel("S₀(q) (e²)")
    ax.set_title("Isotropic scattering S₀(q)")
    ax.set_xlim(q_arr.min(), q_arr.max())
    ax.legend(fontsize=8)

    # (1,1) ΔPDF time-averaged — beneath ΔPD for direct comparison
    ax = axes[1, 1]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(R, delta_PDF, color="purple", lw=1.5,
            label=f"T = {args.T} K  (t = T)")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("ΔPDF (a.u.)")
    ax.set_title("ΔPDF  (end of trajectory, from scattering)")
    ax.set_xlim(0, R.max())
    ax.legend(fontsize=8)

    # (1,2) ΔPDF(R, τ) heatmap — beneath ΔPD(R, τ)
    ax = axes[1, 2]
    vmax_pdf = np.percentile(np.abs(rPDF_traj), 99)
    im2 = ax.imshow(
        rPDF_traj, aspect="auto", origin="lower",
        extent=[time_grid_fs[0], time_grid_fs[-1], R[0], R[-1]],
        cmap="RdBu_r", vmin=-vmax_pdf, vmax=vmax_pdf,
    )
    ax.set_xlabel("τ (fs)")
    ax.set_ylabel("R (Å)")
    ax.set_title("ΔPDF(R, τ)  (from scattering)")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="ΔPDF")

    fig.suptitle(f"mol_{sample_idx:06d}  —  {formula}", fontsize=12)
    fig.tight_layout()
    fig.savefig(paths["fig"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {paths['fig']}")


if __name__ == "__main__":
    main()