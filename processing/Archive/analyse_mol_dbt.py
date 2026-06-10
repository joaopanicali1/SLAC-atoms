#!/usr/bin/env python
"""
analyse_mol_dbt.py — PD(R,t) extraction via Discrete Bessel Transform (DBT).

Input: mol_XXXXXX_physics.npz produced by the ORCA/PySCF pipeline, containing
the DFT-optimised geometry, atomic masses, and normal-mode eigenvectors.
A classical harmonic trajectory is generated from those modes and fed through
the full Natan-group analysis pipeline (S0 → Lorch → DBT → PD).

Methodology (follows DBT.m / Untitled.m):
  1. S₀(q,t) computed via the Debye formula for every trajectory frame
  2. Normalized windowed signal: dS₀ = [S₀ − Σf²] · Lorch / fe
       fe = Σ_{i<j} f_i(q)·f_j(q)   (effective form factor)
       Lorch = sin(linspace(0,π,N)) / sum(...)   (normalized window)
  3. DBT (Discrete Bessel Transform, order 0) inverts dS₀ → PD(R,t)
  4. ΔPD(R,t) = PD(R,t) − PD(R,0)

Usage
-----
  # Combined thermal trajectory — all modes superimposed at temperature T:
  python processing/analyse_mol_dbt.py mol_data/mol_water_physics.npz

  # Per-mode trajectories — one panel per mode, each mode oscillates alone:
  python processing/analyse_mol_dbt.py mol_data/mol_water_physics.npz --per-mode

Tunable parameters
------------------
  --T FLOAT             Temperature [K] for thermal mode amplitudes
                        A_k = sqrt(2 kB T) / ω_k  (default: 300).
                        Higher T gives larger displacements and a stronger ΔPD
                        signal. Ignored when --amplitude is set.

  --dq FLOAT            q-grid spacing [Å⁻¹] (default: 0.1).
                        Finer dq gives denser q sampling but increases compute.

  --qmax FLOAT          Maximum q [Å⁻¹] (default: 10).
                        Sets the detector q-range. Real-space resolution is
                        determined by this value: ΔR ≈ π / qmax ≈ 0.31 Å at 10 Å⁻¹.

  --dR FLOAT            Target R-grid spacing [Å] (default: 0.025).
                        Sets the DBT output sampling density via M = π/(dR·dq).
                        Does not improve resolution beyond ΔR ≈ π/qmax.

  --per-mode            Generate one independent trajectory per normal mode
                        instead of the combined thermal trajectory. Each mode
                        oscillates alone on a shared time grid so its individual
                        contribution to PD(R,t) is isolated. Produces one
                        figure row per mode.

  --n-cycles FLOAT      Per-mode only: number of slowest-mode periods to span
                        (default: 3). All modes share the same time axis so
                        faster modes show proportionally more oscillation cycles.

  --points-per-fast-cycle INT
                        Frames per period of the fastest mode (default: 10).
                        Higher values give smoother trajectories at the cost of
                        more frames to compute.

  --amplitude FLOAT     Fix the maximum Cartesian displacement [Å] for every
                        mode instead of using the thermal amplitude.
                        For each mode k, the normal-coordinate amplitude is:
                          A_k = amplitude / max_i(|V_vib[:,k] · 1/sqrt(m_i)|)
                        so every mode displaces atoms by the same peak distance.
                        Use 0.0794 Å (= 0.15 Bohr) to match the Natan-group
                        .mat convention. When unset, thermal amplitudes are used.

  --no-plot             Skip figure generation and saving. Useful when running
                        in batch mode or on a headless cluster where only the
                        .npz output is needed.

  --outdir PATH         Output directory for saved .npz and .png files
                        (default: PD_data/ in the project root).

Output files
------------
Combined mode (default):   <stem>_dbt.npz  +  <stem>_dbt_analysis.png
Per-mode (--per-mode):     <stem>_dbt_permode.npz  +  <stem>_permode_dbt_analysis.png

Arrays saved in the combined .npz:
  q          (nq,)            q-grid [Å⁻¹]
  Rgrid      (M-1,)           real-space grid from the DBT [Å]
  time_fs    (nT,)            trajectory time axis [fs]
  detector   (nq,)            Lorch window (normalized, sum=1)
  S0_all     (nq, nT)         total isotropic scattering S₀(q,t) per frame [e²]
  dS0_all    (nq, nT)         normalized windowed signal fed to the DBT:
                                dS₀ = [S₀(t) − Σf²] · Lorch / fe
  PD_dbt     (M-1, nT)        pair density PD(R,t) extracted by DBT [a.u.]
  dPD_dbt    (M-1, nT)        differential pair density ΔPD(R,t) = PD(t) − PD(0)
  traj_Rab   (nT, nPair)      ground-truth interatomic distances Rᵢⱼ(t) [Å]

Arrays saved in the per-mode .npz (one set per mode k = 1..N):
  q          (nq,)            q-grid [Å⁻¹]
  Rgrid      (M-1,)           real-space grid from the DBT [Å]
  freq_cm1   (N_modes,)       normal-mode frequencies [cm⁻¹]
  time_fs_mode<k>  (nT,)      shared time axis for all modes [fs]
  S0_mode<k>       (nq, nT)   S₀(q,t) for mode k
  dS0_mode<k>      (nq, nT)   dS₀(q,t) for mode k
  PD_mode<k>       (M-1, nT)  PD(R,t) for mode k
  dPD_mode<k>      (M-1, nT)  ΔPD(R,t) for mode k
  Rab_mode<k>      (nT, nPair) ground-truth Rᵢⱼ(t) for mode k [Å]
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import jv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.db import connect as _ase_connect
from ase import units as _ase_units
from ase.data import atomic_numbers

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DB_PATH      = os.path.join(PROJECT_ROOT, "data/train_4m", "data0000.aselmdb")


# ═════════════════════════════════════════════════════════════════════════════
# Waasmaier–Kirfel / Cromer–Mann form-factor table  (ported from SDS_min.m)
# Coefficients: [a1..a5, c0, b1..b5]
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# Form factors
# ═════════════════════════════════════════════════════════════════════════════

def _cm_form_factor(q, symbol):
    """Cromer-Mann X-ray form factor for one element at q [Å⁻¹]. Returns (nq,)."""
    c = _CM.get(symbol)
    if c is None:
        return np.full(len(q), float(atomic_numbers.get(symbol, 1)))
    a  = np.array(c[:5])
    c0 = c[5]
    b  = np.array(c[6:])
    k2 = (q / (4 * np.pi)) ** 2
    return np.exp(-k2[:, None] * b) @ a + c0


def form_factors_and_fe(q, names):
    """
    Compute per-atom form factors, self-term Σf², and effective form factor fe.

    fe(q) = Σ_{i<j} f_i(q)·f_j(q)   — matches Untitled.m normalization.

    Returns f (nq,N), self_term (nq,), fe (nq,), ii, jj.
    """
    N         = len(names)
    f         = np.column_stack([_cm_form_factor(q, s) for s in names])
    ii, jj    = np.tril_indices(N, k=-1)
    self_term = np.sum(f ** 2, axis=1)
    fe        = np.sum(f[:, ii] * f[:, jj], axis=1)
    return f, self_term, fe, ii, jj


# ═════════════════════════════════════════════════════════════════════════════
# Classical harmonic trajectory
# ═════════════════════════════════════════════════════════════════════════════

def generate_trajectory(R0_flat, masses_amu, V_vib, omegas_ASE,
                        T_kelvin=300.0, points_per_fast_cycle=10, seed=42):
    """
    Classical harmonic trajectory (ported from trajectory_pipeline_cluster.ipynb).

    Duration  = period of the slowest mode.
    Time step = period of the fastest mode / points_per_fast_cycle.
    Amplitude = sqrt(2 kB T) / ω_k  (classical equipartition).

    Returns time_fs (num_frames,), xyz_traj (num_frames, N, 3).
    """
    rng = np.random.default_rng(seed)
    N   = len(R0_flat) // 3

    periods_fs   = (2 * np.pi / omegas_ASE) / _ase_units.fs
    t_total_fs   = np.max(periods_fs)
    dt_fs        = np.min(periods_fs) / points_per_fast_cycle
    time_fs      = np.arange(0, t_total_fs + dt_fs, dt_fs)
    num_frames   = len(time_fs)

    time_ASE   = (time_fs * _ase_units.fs).reshape(-1, 1)
    A_k        = np.sqrt(2 * _ase_units.kB * T_kelvin) / omegas_ASE
    phi        = rng.uniform(0, 2 * np.pi, len(omegas_ASE))
    c_t        = A_k * np.cos(omegas_ASE * time_ASE + phi)   # (num_frames, n_modes)
    inv_sqrt_m = 1.0 / np.sqrt(masses_amu)
    delta_R    = (V_vib * inv_sqrt_m[:, None]) @ c_t.T        # (3N, num_frames)
    xyz_traj   = (R0_flat[:, None] + delta_R).T.reshape(num_frames, N, 3)

    print(f"  Slowest: {t_total_fs:.1f} fs  |  Fastest: {np.min(periods_fs):.2f} fs  "
          f"|  {num_frames} frames  (dt = {dt_fs:.2f} fs)")
    return time_fs, xyz_traj


# ═════════════════════════════════════════════════════════════════════════════
# Metadata helpers
# ═════════════════════════════════════════════════════════════════════════════

def parse_sample_idx(path):
    """Return numeric index from mol_XXXXXX_physics.npz, or None for named files."""
    m = re.search(r"mol_(\d+)_physics", os.path.basename(path))
    return int(m.group(1)) if m else None


def get_atom_names(sample_idx, npz_path=None):
    """
    Look up atom symbols from the ASE database.
    Falls back to reading 'symbols' from the npz if sample_idx is None
    (e.g. mol_water_physics.npz created outside the database pipeline).
    """
    if sample_idx is not None:
        db  = _ase_connect(DB_PATH)
        row = db.get(int(sample_idx) + 1)
        return list(row.toatoms().get_chemical_symbols())
    # Named file: try 'symbols' key, else infer from R_0 length
    if npz_path is not None:
        d = np.load(npz_path, allow_pickle=True)
        if "symbols" in d:
            return list(d["symbols"])
        if "atom_names" in d:
            return list(d["atom_names"])
    raise ValueError("Cannot determine atom names — no database index and no 'symbols' key in npz.")


def hill_formula(names):
    """Hill notation as a matplotlib math string, e.g. '$C_2H_3ClN$'."""
    counts = Counter(names)
    order  = [el for el in ("C", "H") if el in counts]
    order += [el for el in sorted(counts) if el not in ("C", "H")]
    parts  = [s if counts[s] == 1 else f"{s}_{{{counts[s]}}}" for s in order]
    return "$" + "".join(parts) + "$"


# ═════════════════════════════════════════════════════════════════════════════
# Discrete Bessel Transform  (DBT.m — Natan 2020)
# ═════════════════════════════════════════════════════════════════════════════

def _sph_bessel(n, x):
    """Spherical Bessel j_n(x) = sqrt(π/2x) J_{n+0.5}(x). Safe at x=0."""
    x   = np.asarray(x, dtype=float)
    lim = 1.0 if n == 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.sqrt(np.pi / (2.0 * x)) * jv(n + 0.5, x)
    return np.where(np.abs(x) < 1e-14, lim, val)


def dbt(data, q, M):
    """
    Discrete Bessel Transform of order 0 — direct port of DBT.m.

    For n=0 (isotropic scattering), j_0 zeros are exactly k·π.

    Parameters
    ----------
    data : (nq,) or (nq, nT)
    q    : (nq,) uniform q-grid [Å⁻¹]
    M    : int   padding / sampling points  (sets R-grid density)

    Returns
    -------
    PD    : (M-1,) or (M-1, nT)
    Rgrid : (M-1,)  [Å]
    """
    scalar = data.ndim == 1
    if scalar:
        data = data[:, None]

    dq  = q[1] - q[0]
    qp  = q[0] + dq * np.arange(M)            # padded q-grid
    BZ  = np.arange(1, M + 1) * np.pi         # zeros of j_0: k·π, k=1..M

    # Resampled grids (DBT.m: resample_q_grid / resample_R_grid)
    qgrid = qp[-1] * BZ[:-1] / BZ[-1]         # (M-1,)
    Rgrid = BZ[:-1] / qp[-1]                  # (M-1,)  [Å]

    # Resample data onto qgrid with cubic interpolation (≈ makima)
    f_int  = interp1d(q, data, axis=0, kind="cubic",
                      bounds_error=False, fill_value=0.0)
    data_s = f_int(qgrid)                      # (M-1, nT)

    # Scaling factor
    scalingFactor = qp[-1] ** 2 / BZ[-1]

    # Transform kernel G (M-1 × M-1)
    #   G_de[m]   = BZ[-1] · j_1(BZ[m])²           row denominator
    #   G_nu[m,n] = 2 · j_0(BZ[m]·BZ[n] / BZ[-1])  numerator matrix
    #   G[m,n]    = G_nu[m,n] / G_de[m]             row-wise division
    G_de  = BZ[-1] * _sph_bessel(1, BZ[:-1]) ** 2          # (M-1,)
    G_nu  = 2.0 * _sph_bessel(0, np.outer(BZ[:-1], BZ[:-1]) / BZ[-1])
    G     = G_nu / G_de[:, None]                            # (M-1, M-1)

    # Jacobian (multiply each row by qgrid) then transform
    data_s *= qgrid[:, None]
    PD = scalingFactor * (G @ data_s)          # (M-1, nT)

    return (PD[:, 0] if scalar else PD), Rgrid


# ═════════════════════════════════════════════════════════════════════════════
# Lorch window  (Untitled.m)
# ═════════════════════════════════════════════════════════════════════════════

def lorch_detector(q):
    """
    Lorch window normalized to unit sum — exact match of Untitled.m:
      Lorch    = sin(linspace(0, π, N))
      detector = Lorch / sum(Lorch)
    """
    lorch = np.sin(np.linspace(0.0, np.pi, len(q)))
    return lorch / lorch.sum()


# ═════════════════════════════════════════════════════════════════════════════
# Core DBT pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(xyz_traj, names, q, M, f_mat, self_term, fe, fp, ii, jj, detector):
    """
    Run the full DBT pipeline on one trajectory.

    Parameters
    ----------
    xyz_traj : (num_frames, N_atoms, 3)  [Å]
    names    : list of atom symbols
    q        : (nq,) q-grid [Å⁻¹]
    M        : DBT padding
    f_mat, self_term, fe, fp, ii, jj, detector : pre-computed form-factor quantities

    Returns
    -------
    S0_all  : (nq, nT)
    dS0_all : (nq, nT)
    PD_dbt  : (M-1, nT)
    dPD_dbt : (M-1, nT)
    traj_Rab: (nT, nPair)
    Rgrid   : (M-1,)
    """
    num_frames = len(xyz_traj)
    traj_Rab = np.sqrt(np.sum(
        (xyz_traj[:, ii, :] - xyz_traj[:, jj, :]) ** 2, axis=2
    ))  # (num_frames, nPair)

    batch  = max(1, int(200e6 / (len(q) * len(ii) * 8)))
    S0_all = np.empty((len(q), num_frames))
    for b0 in range(0, num_frames, batch):
        b1   = min(b0 + batch, num_frames)
        qRb  = q[:, None, None] * traj_Rab[b0:b1].T[None, :, :]
        j0_b = np.where(np.abs(qRb) > 1e-10, np.sin(qRb) / qRb, 1.0)
        S0_all[:, b0:b1] = (self_term[:, None]
                            + 2 * np.einsum("qp,qpt->qt", fp, j0_b))

    fe_safe = np.where(fe > 0, fe, 1.0)
    dS0_all = ((S0_all - self_term[:, None])
               * detector[:, None]
               / fe_safe[:, None])

    PD_dbt, Rgrid = dbt(dS0_all, q, M)
    dPD_dbt       = PD_dbt - PD_dbt[:, [0]]

    return S0_all, dS0_all, PD_dbt, dPD_dbt, traj_Rab, Rgrid


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _plot_mode_row(fig, axes_row, time_fs, q, Rgrid, S0_all, dS0_all,
                   PD_dbt, dPD_dbt, traj_Rab, pair_labels, qmax,
                   mode_title, first_row=False):
    """Fill one row of the per-mode figure (4 panels: S0 | PD | ΔPD | Rab)."""
    R_show = min(5.0, Rgrid[-1])
    Rm     = Rgrid <= R_show

    # Col 0: S0(q,t)
    ax = axes_row[0]
    im = ax.imshow(S0_all, aspect="auto", origin="lower", cmap="viridis",
                   extent=[time_fs[0], time_fs[-1], q[0], q[-1]])
    ax.set_ylim(0, qmax); ax.set_xlabel("τ (fs)"); ax.set_ylabel("q (Å⁻¹)")
    ax.set_title(f"S₀(q,τ) — {mode_title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Col 1: PD(R,t) from DBT
    ax = axes_row[1]
    im = ax.imshow(PD_dbt[Rm], aspect="auto", origin="lower", cmap="turbo",
                   extent=[time_fs[0], time_fs[-1], Rgrid[Rm][0], Rgrid[Rm][-1]])
    ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)")
    ax.set_title(f"PD₀(R,τ) — {mode_title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Col 2: ΔPD(R,t)
    ax = axes_row[2]
    vmax = np.percentile(np.abs(dPD_dbt[Rm]), 99) or 1e-10
    im = ax.imshow(dPD_dbt[Rm], aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[time_fs[0], time_fs[-1], Rgrid[Rm][0], Rgrid[Rm][-1]])
    ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)")
    ax.set_title(f"ΔPD₀(R,τ) — {mode_title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Col 3: ground-truth Rab(t)
    ax = axes_row[3]
    for k, lbl in enumerate(pair_labels):
        ax.plot(time_fs, traj_Rab[:, k], lw=1.0, label=lbl)
    ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)")
    ax.set_ylim(0, R_show); ax.set_title(f"Ground truth Rᵢⱼ — {mode_title}")
    ax.legend(fontsize=7)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path",                                      help="mol_XXXXXX_physics.npz from the ORCA/PySCF pipeline")
    parser.add_argument("--T",    type=float, default=300.0,              help="temperature [K] for thermal amplitudes (default 300)")
    parser.add_argument("--dq",   type=float, default=0.1,                help="q-grid spacing [Å⁻¹] (default 0.1)")
    parser.add_argument("--qmax", type=float, default=10.0,               help="maximum q [Å⁻¹] (default 10)")
    parser.add_argument("--dR",   type=float, default=0.025,              help="target R-grid spacing [Å], sets DBT padding M (default 0.025)")
    parser.add_argument("--points-per-fast-cycle", type=int, default=10,  help="frames per fastest-mode period (default 10)")
    parser.add_argument("--per-mode", action="store_true",
                        help="generate one trajectory per normal mode instead of "
                             "the combined thermal trajectory")
    parser.add_argument("--n-cycles", type=float, default=3.0,
                        help="per-mode only: slowest-mode periods to span (default 3)")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="fix max Cartesian displacement [Å] for all modes, "
                             "overriding thermal amplitudes (0.0794 = 0.15 Bohr)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip figure generation and saving (npz output is always written)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    # ── q grid and M ──────────────────────────────────────────────────────────
    q  = np.arange(1e-10, args.qmax + args.dq / 2, args.dq)
    dq = q[1] - q[0]
    M  = int(np.ceil(np.pi / (args.dR * dq))) + 1
    outdir = args.outdir or os.path.join(PROJECT_ROOT, "PD_data")
    os.makedirs(outdir, exist_ok=True)
    print(f"q: [{q[0]:.4f}, {q[-1]:.2f}] Å⁻¹  ({len(q)} pts)  M={M}  ΔR≈{np.pi/q[-1]:.3f} Å")

    # ── Load physics npz ──────────────────────────────────────────────────────
    data       = np.load(args.input_path)
    R0_flat    = data["R_0"]
    masses_amu = data["masses_amu"]
    V_vib      = data["V_vib"]
    omegas_ASE = data["omegas_ASE"]
    N_atoms    = len(R0_flat) // 3

    sample_idx  = parse_sample_idx(args.input_path)
    names       = get_atom_names(sample_idx, npz_path=args.input_path)
    formula     = hill_formula(names)
    stem        = (f"mol_{sample_idx:06d}" if sample_idx is not None
                   else os.path.splitext(os.path.basename(args.input_path))[0]
                        .replace("_physics", ""))
    print(f"{stem}  {formula}  ({N_atoms} atoms, {len(omegas_ASE)} modes)")

    f_mat_ff, self_term, fe, ii, jj = form_factors_and_fe(q, names)
    fp          = f_mat_ff[:, ii] * f_mat_ff[:, jj]
    pair_labels = [f"{names[i]}-{names[j]}" for i, j in zip(ii, jj)]
    detector    = lorch_detector(q)
    freq_cm1    = omegas_ASE * 521.47
    inv_sqrt_m  = 1.0 / np.sqrt(masses_amu)
    N           = N_atoms

    # ── Per-mode: one trajectory per normal mode ──────────────────────────────
    if args.per_mode:
        # All modes share one time grid: n_cycles × slowest period,
        # dt = fastest period / points_per_fast_cycle
        periods_fs  = (2 * np.pi / omegas_ASE) / _ase_units.fs
        t_total_fs  = np.max(periods_fs) * args.n_cycles
        dt_fs       = np.min(periods_fs) / args.points_per_fast_cycle
        shared_time = np.arange(0, t_total_fs + dt_fs, dt_fs)
        amp_label   = (f"fixed {args.amplitude:.4f} Å" if args.amplitude
                       else f"thermal T={args.T} K")
        print(f"Per-mode ({len(omegas_ASE)} modes)  "
              f"grid: 0–{shared_time[-1]:.1f} fs  ({len(shared_time)} frames, dt={dt_fs:.3f} fs)  "
              f"amplitude: {amp_label}")

        rng      = np.random.default_rng(42)
        time_ASE = (shared_time * _ase_units.fs).reshape(-1, 1)
        results  = []
        for k in range(len(omegas_ASE)):
            print(f"  mode {k+1}  ({freq_cm1[k]:.1f} cm⁻¹)  "
                  f"({args.n_cycles:.0f}× {periods_fs[k]:.2f} fs period)")
            cart_col = V_vib[:, k] * inv_sqrt_m
            A_k = (args.amplitude / np.max(np.abs(cart_col)) if args.amplitude is not None
                   else np.sqrt(2 * _ase_units.kB * args.T) / omegas_ASE[k])
            phi        = rng.uniform(0, 2 * np.pi)
            c_t        = A_k * np.cos(omegas_ASE[k] * time_ASE.flatten() + phi)
            dR         = np.outer(cart_col, c_t)                              # (3N, nT)
            xyz_traj_k = (R0_flat[:, None] + dR).T.reshape(len(shared_time), N, 3)
            S0, dS0, PD, dPD, Rab, Rgrid = run_pipeline(
                xyz_traj_k, names, q, M, f_mat_ff, self_term, fe, fp, ii, jj, detector
            )
            results.append((shared_time, S0, dS0, PD, dPD, Rab, Rgrid))

        save_dict = dict(q=q, Rgrid=results[0][6], freq_cm1=freq_cm1)
        for k, (tf, S0, dS0, PD, dPD, Rab, _) in enumerate(results):
            save_dict.update({f"time_fs_mode{k+1}": tf,
                              f"S0_mode{k+1}": S0, f"dS0_mode{k+1}": dS0,
                              f"PD_mode{k+1}": PD,  f"dPD_mode{k+1}": dPD,
                              f"Rab_mode{k+1}": Rab})
        np.savez(os.path.join(outdir, f"{stem}_dbt_permode.npz"), **save_dict)
        print(f"Saved: {os.path.join(outdir, f'{stem}_dbt_permode.npz')}")

        if not args.no_plot:
            n_modes = len(results)
            fig, axes = plt.subplots(n_modes, 4, figsize=(20, 5 * n_modes))
            if n_modes == 1:
                axes = axes[np.newaxis, :]
            for k, (time_fs_k, S0, dS0, PD, dPD, Rab, Rgrid) in enumerate(results):
                _plot_mode_row(fig, axes[k], time_fs_k, q, Rgrid,
                               S0, dS0, PD, dPD, Rab, pair_labels, args.qmax,
                               f"mode {k+1}  ({freq_cm1[k]:.1f} cm⁻¹)")
            fig.suptitle(f"{stem}  —  {formula}  (DBT per-mode, {amp_label})", fontsize=12)

    # ── Combined: all modes superimposed ─────────────────────────────────────
    else:
        if args.amplitude is not None:
            print(f"Generating combined trajectory (fixed amplitude {args.amplitude:.4f} Å per mode)...")
            rng      = np.random.default_rng(42)
            periods_fs = (2 * np.pi / omegas_ASE) / _ase_units.fs
            t_total_fs = np.max(periods_fs)
            dt_fs      = np.min(periods_fs) / args.points_per_fast_cycle
            time_fs    = np.arange(0, t_total_fs + dt_fs, dt_fs)
            time_ASE   = (time_fs * _ase_units.fs).reshape(-1, 1)
            delta_R    = np.zeros((len(R0_flat), len(time_fs)))
            for k in range(len(omegas_ASE)):
                cart_col = V_vib[:, k] * inv_sqrt_m
                A_k = args.amplitude / np.max(np.abs(cart_col))
                phi = rng.uniform(0, 2 * np.pi)
                c_t = A_k * np.cos(omegas_ASE[k] * time_ASE.flatten() + phi)
                delta_R += np.outer(cart_col, c_t)
            xyz_traj = (R0_flat[:, None] + delta_R).T.reshape(len(time_fs), N, 3)
            print(f"  {t_total_fs:.1f} fs  |  {len(time_fs)} frames  (dt={dt_fs:.2f} fs)")
        else:
            print("Generating combined thermal trajectory (all modes)...")
            time_fs, xyz_traj = generate_trajectory(
                R0_flat, masses_amu, V_vib, omegas_ASE,
                T_kelvin=args.T, points_per_fast_cycle=args.points_per_fast_cycle,
            )

        print("Computing S₀ and applying DBT...")
        S0_all, dS0_all, PD_dbt, dPD_dbt, traj_Rab, Rgrid = run_pipeline(
            xyz_traj, names, q, M, f_mat_ff, self_term, fe, fp, ii, jj, detector
        )

        np.savez(os.path.join(outdir, f"{stem}_dbt.npz"),
                 Rgrid=Rgrid, q=q, time_fs=time_fs, detector=detector,
                 S0_all=S0_all, dS0_all=dS0_all,
                 PD_dbt=PD_dbt, dPD_dbt=dPD_dbt, traj_Rab=traj_Rab)
        print(f"Saved: {os.path.join(outdir, f'{stem}_dbt.npz')}")

        amp_label = (f"fixed {args.amplitude:.4f} Å" if args.amplitude
                     else f"thermal T={args.T} K")
        if not args.no_plot:
            R_show = min(6.0, Rgrid[-1])
            Rm     = Rgrid <= R_show
            fig, axes = plt.subplots(3, 2, figsize=(14, 13))

            ax = axes[0, 0]
            im = ax.imshow(S0_all, aspect="auto", origin="lower", cmap="viridis",
                           extent=[time_fs[0], time_fs[-1], q[0], q[-1]])
            ax.set_ylim(0, args.qmax); ax.set_xlabel("τ (fs)"); ax.set_ylabel("q (Å⁻¹)")
            ax.set_title("S₀(q, τ)"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[0, 1]
            im = ax.imshow(PD_dbt[Rm], aspect="auto", origin="lower", cmap="turbo",
                           extent=[time_fs[0], time_fs[-1], Rgrid[Rm][0], Rgrid[Rm][-1]])
            ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)"); ax.set_title("PD(R, τ)  [DBT]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[1, 0]
            vmax = np.percentile(np.abs(dS0_all), 99)
            im = ax.imshow(dS0_all, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax,
                           extent=[time_fs[0], time_fs[-1], q[0], q[-1]])
            ax.set_ylim(0, args.qmax); ax.set_xlabel("τ (fs)"); ax.set_ylabel("q (Å⁻¹)")
            ax.set_title("dS₀(q, τ)  [(S₀−Σf²)·Lorch/fe]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[1, 1]
            vmax = np.percentile(np.abs(dPD_dbt[Rm]), 99)
            im = ax.imshow(dPD_dbt[Rm], aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax,
                           extent=[time_fs[0], time_fs[-1], Rgrid[Rm][0], Rgrid[Rm][-1]])
            ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)")
            ax.set_title("ΔPD(R, τ)  [PD(t)−PD(0), DBT]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[2, 0]
            ax.plot(q, dS0_all[:, 0],  color="navy",     lw=1.5, label="t = 0")
            ax.plot(q, dS0_all[:, -1], color="firebrick", lw=1.5, ls="--", label="t = T")
            ax.set_xlabel("q (Å⁻¹)"); ax.set_ylabel("dS₀ (a.u.)")
            ax.set_title("dS₀(q)  at t=0 and t=T"); ax.legend(fontsize=9)

            ax = axes[2, 1]
            for k, lbl in enumerate(pair_labels):
                ax.plot(time_fs, traj_Rab[:, k], lw=0.9, label=lbl)
            ax.set_xlabel("τ (fs)"); ax.set_ylabel("R (Å)")
            ax.set_ylim(0, R_show); ax.set_title("Ground truth  Rᵢⱼ(τ)")
            ax.legend(fontsize=6, ncol=2)

            fig.suptitle(f"{stem}  —  {formula}  (DBT combined, {amp_label})", fontsize=12)

    # ── Save figure ───────────────────────────────────────────────────────────
    if args.no_plot:
        print("Plot skipped (--no-plot).")
    else:
        out_fig = (os.path.join(outdir, f"{stem}_permode_dbt_analysis.png") if args.per_mode
                   else os.path.join(outdir, f"{stem}_dbt_analysis.png"))
        fig.tight_layout()
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
