"""
demo_methodology.py — End-to-end pipeline walkthrough using water (H₂O).

Shows the complete methodology of modes_lib.py and analyse_mol.py on a
simple 3-atom molecule so every step is easy to follow and verify.

Pipeline order (mirrors the production pipeline):
  1.  DFT calculation with ORCA  — geometry optimisation + Hessian
  2.  Normal-mode analysis        — mass-weighted Hessian → frequencies
  3.  Classical harmonic trajectory
  4.  Static pair density PD(R)
  5.  Differential pair density ΔPD(R)
  6.  Isotropic X-ray scattering S₀(q)  (Debye formula)
  7.  Differential scattering ΔPDF(R)   (sine Fourier transform)
  8.  Summary figure

Run from the project root:
    python processing/demo_methodology.py
"""

import os
import subprocess
import shutil
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import units as ase_units
from ase.data import atomic_masses, atomic_numbers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORCA_BIN     = os.path.join(PROJECT_ROOT,
               "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg", "orca")
MPI_DIR      = os.path.join(PROJECT_ROOT, "openmpi-4.1.8_install")
OUT_FIG      = os.path.join(PROJECT_ROOT, "processing", "demo", "demo_methodology.png")

# ── Production settings ───────────────────────────────────────────────────────
# These mirror the constants in modes_lib.py and analyse_mol.py exactly so the
# demo runs with the same configuration as the production pipeline.

# DFT settings (modes_lib.py: FUNCTIONAL, DISPERSION, ORCA_BASIS, ORCA_MAXCORE)
FUNCTIONAL   = "B3LYP"       # exchange-correlation functional
DISPERSION   = "D4"          # dispersion correction (Grimme D4)
BASIS        = "def2-TZVPD"  # triple-zeta polarised basis with diffuse functions
AUX_BASIS    = "def2/J"      # Weigend Coulomb-fitting auxiliary basis (RIJCOSX)
ORCA_MAXCORE = 800           # MB of RAM per MPI process
NPROCS       = 1             # MPI processes 

# Normal-mode filtering (modes_lib.py: FREQ_LB, FREQ_UB)
FREQ_LB = 5.0     # cm⁻¹  lower bound — discards near-zero numerical noise
FREQ_UB = 4000.0  # cm⁻¹  upper bound — covers all physical molecular vibrations

# Trajectory (analyse_mol.py: --T, --points-per-fast-cycle)
T_KELVIN            = 300.0  # temperature in K
DT_POINTS_PER_CYCLE = 10     # time steps per fastest-mode period

# Pair density (analyse_mol.py: --dR)
DR = 0.01   # histogram bin width in Å

# Scattering (analyse_mol.py: --q-min, --q-max, --window)
Q_MIN  = 0.01      # minimum q in Å⁻¹
Q_MAX  = 15.0      # maximum q in Å⁻¹
WINDOW = "hanning" # apodisation applied to ΔsM(q) before sine FT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — DFT calculation with ORCA
#
# ORCA performs B3LYP-D4/def2-TZVPD geometry optimisation followed by an
# analytical Hessian calculation (FREQ keyword).  RIJCOSX approximates the
# HF exchange integrals semi-numerically; error vs. conventional B3LYP is
# < 1 cm⁻¹ for harmonic frequencies.
#
# Output files we need:
#   water.hess  — 3N×3N Hessian in Hartree/Bohr²
#   water.xyz   — optimised geometry (written by TightOpt)
# ─────────────────────────────────────────────────────────────────────────────

ORCA_INPUT = f"""\
! {FUNCTIONAL} {DISPERSION} RIJCOSX {BASIS} {AUX_BASIS} TightSCF TightOpt FREQ
%pal nprocs {NPROCS} end
%maxcore {ORCA_MAXCORE}
%scf MaxIter 300 end
* xyz 0 1
O   0.000000   0.000000   0.117176
H   0.000000   0.757200  -0.467300
H   0.000000  -0.757200  -0.467300
*
"""

scratch   = tempfile.mkdtemp(prefix="demo_water_orca_")
inp_path  = os.path.join(scratch, "water.inp")
out_path  = os.path.join(scratch, "water.out")
hess_path = os.path.join(scratch, "water.hess")
xyz_path  = os.path.join(scratch, "water.xyz")

with open(inp_path, "w") as fh:
    fh.write(ORCA_INPUT)

# Configure the subprocess environment so ORCA can find its OpenMPI libraries
env = os.environ.copy()
env["PATH"]            = f"{os.path.dirname(ORCA_BIN)}:{MPI_DIR}/bin:{env.get('PATH', '')}"
env["LD_LIBRARY_PATH"] = f"{os.path.dirname(ORCA_BIN)}:{MPI_DIR}/lib:{env.get('LD_LIBRARY_PATH', '')}"

with open(out_path, "w") as fout:
    result = subprocess.run(
        [ORCA_BIN, inp_path],
        stdout=fout, stderr=subprocess.STDOUT,
        cwd=scratch, env=env,
    )

if result.returncode != 0 or not os.path.exists(hess_path):
    raise RuntimeError(f"ORCA failed — check {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Normal-mode analysis
#
# The Hessian H from ORCA is in Hartree/Bohr².  We:
#   (a) Parse the .hess file (block-column text format)
#   (b) Convert units: Hartree/Bohr² → eV/Å²  using ASE unit constants
#   (c) Read the DFT-optimised geometry from water.xyz (last XYZ frame)
#   (d) Build the mass vector: each atom's mass repeated for x, y, z
#   (e) Form the mass-weighted Hessian: H̃ = M^{-1/2} H M^{-1/2}
#   (f) Diagonalise: H̃ V = V Λ  →  eigenvalues λ_k = ω_k²
#   (g) Strip 6 rigid-body modes (3 translations + 3 rotations),
#       convert to cm⁻¹  (1 ASE unit = 521.47 cm⁻¹),
#       and filter to the physical range [5, 4000] cm⁻¹
# ─────────────────────────────────────────────────────────────────────────────

def parse_orca_hessian(filepath):
    """Parse ORCA .hess file → symmetric Hessian matrix in Hartree/Bohr²."""
    with open(filepath) as fh:
        lines = fh.readlines()
    start = next(i + 2 for i, l in enumerate(lines) if "$hessian" in l)
    dim = int(lines[start - 1].strip())
    H   = np.zeros((dim, dim))
    col_start = 0
    for line in lines[start:]:
        parts = line.split()
        if not parts or "$" in line:
            break
        if not any("." in p for p in parts):
            col_start = int(parts[0])
        else:
            row = int(parts[0])
            for j, val in enumerate(map(float, parts[1:])):
                H[row, col_start + j] = val
    return (H + H.T) / 2   # symmetrise to correct for floating-point asymmetry

def read_last_xyz_frame(xyz_file):
    """Return (symbols, coords [Å]) for the last frame of an ORCA .xyz file."""
    with open(xyz_file) as fh:
        lines = fh.readlines()
    last_sym, last_coords = [], []
    i = 0
    while i < len(lines):
        try:
            n   = int(lines[i].strip())
            sym = [lines[i+2+k].split()[0] for k in range(n)]
            xyz = [[float(v) for v in lines[i+2+k].split()[1:4]] for k in range(n)]
            last_sym, last_coords = sym, xyz
            i += n + 2
        except (ValueError, IndexError):
            i += 1
    return last_sym, np.array(last_coords)

# (a) Parse Hessian
H_hartree_bohr2 = parse_orca_hessian(hess_path)
N = H_hartree_bohr2.shape[0] // 3   # number of atoms

# (b) Convert to eV/Å²  (ASE standard units)
H_eV_A2 = H_hartree_bohr2 * (ase_units.Hartree / ase_units.Bohr**2)

# (c) Optimised geometry from the last frame of water.xyz
symbols, R0 = read_last_xyz_frame(xyz_path)   # R0: (N, 3) in Å
R0_flat     = R0.flatten()                     # (3N,)

# (d) Atomic masses repeated ×3 (one entry per Cartesian DOF)
masses_amu = np.repeat([atomic_masses[atomic_numbers[s]] for s in symbols], 3)

# (e) Mass-weighted Hessian  H̃ = M^{-1/2} H M^{-1/2}
inv_sqrt_m = 1.0 / np.sqrt(masses_amu)
H_mw = H_eV_A2 * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

# (f) Diagonalise  →  eigenvalues ω², orthonormal eigenvectors V
eigenvalues_all, V_all = np.linalg.eigh(H_mw)

# (g) Strip rigid-body modes, convert to cm⁻¹, filter
n_rigid    = 6
omegas_all = np.sqrt(np.abs(eigenvalues_all[n_rigid:]))
freqs_cm1  = omegas_all * 521.47   # 1 ASE frequency unit = 521.47 cm⁻¹
valid      = np.where((freqs_cm1 > FREQ_LB) & (freqs_cm1 < FREQ_UB))[0]
omegas_ASE = omegas_all[valid]                  # filtered angular frequencies
V_vib      = V_all[:, n_rigid:][:, valid]       # mass-weighted eigenvectors (3N × n_modes)

# H₂O has 3 modes: H-O-H bending (~1623 cm⁻¹), symmetric stretch (~3782 cm⁻¹),
# asymmetric stretch (~3882 cm⁻¹).  Harmonic DFT values exceed experiment
# (~1595, 3657, 3756 cm⁻¹) due to the harmonic approximation and functional error.

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Classical harmonic trajectory
#
# Each normal mode k is treated as an independent harmonic oscillator.
# Classical equipartition at temperature T gives amplitude A_k = sqrt(2kBT)/ω_k.
# The normal coordinate evolves as Q_k(t) = A_k cos(ω_k t + φ_k), where φ_k
# is a random initial phase.
#
# The Cartesian displacement from equilibrium:
#   Δr(t) = M^{-1/2} V Q(t) = (V / sqrt(masses)) @ Q(t)
#
# The trajectory spans one full period of the slowest mode; the time step
# is set to resolve the fastest mode with 10 samples per cycle.
# ─────────────────────────────────────────────────────────────────────────────

rng        = np.random.default_rng(seed=42)

periods_fs = (2 * np.pi / omegas_ASE) / ase_units.fs
t_total_fs = np.max(periods_fs)                              # duration = one full slow-mode period
dt_fs      = np.min(periods_fs) / DT_POINTS_PER_CYCLE       # time step
time_fs    = np.arange(0, t_total_fs + dt_fs, dt_fs)
n_frames   = len(time_fs)

# Classical amplitude A_k = sqrt(2 kB T) / ω_k  (from equipartition <E_k> = kBT)
A_k = np.sqrt(2 * ase_units.kB * T_KELVIN) / omegas_ASE
phi = rng.uniform(0, 2 * np.pi, len(omegas_ASE))

time_ASE = (time_fs * ase_units.fs).reshape(-1, 1)
Q_t      = A_k * np.cos(omegas_ASE * time_ASE + phi)    # (n_frames, n_modes)

# Convert normal coordinates to Cartesian displacements
delta_R  = (V_vib / np.sqrt(masses_amu)[:, None]) @ Q_t.T   # (3N, n_frames)
xyz_traj = (R0_flat[:, None] + delta_R).T.reshape(n_frames, N, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Static pair density PD(R)
#
# The IAM pair density histograms all unique pairwise distances at the
# DFT-optimised geometry, weighting each pair (i, j) by Z_i × Z_j.
# This weight is the q=0 limit of the X-ray form factor product f_i(0)f_j(0),
# giving the pair density in units of e².
#
#   PD(R) = Σ_{i<j} Z_i Z_j δ(R − R_ij)    [binned with width dR]
# ─────────────────────────────────────────────────────────────────────────────

Z_vals = np.array([atomic_numbers[s] for s in symbols], dtype=float)
ii, jj = np.tril_indices(N, k=-1)
w_ij   = Z_vals[ii] * Z_vals[jj]          # IAM weight for each unique pair

R_static = np.sqrt(np.sum((R0[ii] - R0[jj])**2, axis=1))  # equilibrium distances

R_edges = np.arange(0, R_static.max() + 2*DR, DR)
R       = 0.5 * (R_edges[:-1] + R_edges[1:])              # bin centres
PD_static, _ = np.histogram(R_static, bins=R_edges, weights=w_ij)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Differential pair density ΔPD(R)
#
# ΔPD(R) = PD(R, t_end) − PD(R, t=0)
#
# PD(R, t=0)   — static equilibrium pair density from Section 4 (R_0 from ORCA)
# PD(R, t_end) — pair density at the final trajectory frame (one full slow-mode
#                period after the start)
#
# Positive values at radius R mean pair distances shifted toward R;
# negative values mean pair distances moved away from R.
# Expressed as % of the peak static PD for scale-independent comparison.
# ─────────────────────────────────────────────────────────────────────────────

xyz_end  = xyz_traj[-1]                                             # (N, 3)
R_end    = np.sqrt(np.sum((xyz_end[ii] - xyz_end[jj])**2, axis=1)) # end-frame distances

PD_end, _    = np.histogram(R_end, bins=R_edges, weights=w_ij)
PD_max       = PD_static.max()
delta_PD_pct = (PD_end - PD_static) / PD_max * 100

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Isotropic X-ray scattering S₀(q)
#
# The Debye formula gives the orientationally-averaged scattering intensity:
#
#   S₀(q) = Σ_i f_i(q)²  +  2 Σ_{i<j} f_i(q) f_j(q) sin(qR_ij) / (qR_ij)
#            ───────────        ───────────────────────────────────────────
#            self-scattering         pair interference  (geometry-dependent)
#
# f_i(q) are Cromer-Mann (Waasmaier-Kirfel) form factors: f(q) → Z as q → 0,
# and f(q) → 0 as q → ∞ as the electron cloud scatters less coherently.
# Coefficients: [a1..a5, c0, b1..b5]  from International Tables Vol. C.
# ─────────────────────────────────────────────────────────────────────────────

CM = {
    "O": [2.960427,2.508818,0.637853,0.722838,1.142756,0.027014,
          14.18226,5.93686,0.11273,34.95848,0.39024],
    "H": [0.413048,0.294953,0.187491,0.080701,0.023736,4.9e-05,
          15.56995,32.39847,5.71140,61.88987,1.33412],
}

def form_factor(q, symbol):
    """Cromer-Mann X-ray form factor for a single element. Returns (nq,)."""
    c = CM[symbol]
    a, c0, b = np.array(c[:5]), c[5], np.array(c[6:])
    return np.exp(-((q / (4 * np.pi))**2)[:, None] * b) @ a + c0

q     = np.linspace(Q_MIN, Q_MAX, 600)
f_mat = np.column_stack([form_factor(q, s) for s in symbols])  # (nq, N)

# Self-scattering term — geometry-independent, cancels in ΔS
S0_self = np.sum(f_mat**2, axis=1)

# Pair interference at equilibrium geometry
qR_eq    = q[:, None] * R_static[None, :]
j0_eq    = np.where(qR_eq > 1e-10, np.sin(qR_eq) / qR_eq, 1.0)
S0_ground = S0_self + 2 * np.sum(f_mat[:, ii] * f_mat[:, jj] * j0_eq, axis=1)

# Pair interference at end-of-trajectory geometry
qR_end   = q[:, None] * R_end[None, :]
j0_end   = np.where(qR_end > 1e-10, np.sin(qR_end) / qR_end, 1.0)
S0_end   = S0_self + 2 * np.sum(f_mat[:, ii] * f_mat[:, jj] * j0_end, axis=1)

delta_S  = S0_end - S0_ground   # self-scattering cancels → pure structural signal

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Differential PDF via sine Fourier transform
#
# To recover real-space structural changes from ΔS(q):
#
# 1. Form the modified signal:  ΔsM(q) = q · ΔS(q) / Σ_i f_i(q)²
#    Multiplying by q flattens the amplitude decay; dividing by Σf_i²
#    removes the overall IAM form-factor envelope.
#
# 2. Apply a Hanning taper (1 at q_min → 0 at q_max) to suppress Gibbs
#    ripples that arise from the finite q range.
#
# 3. Sine Fourier transform:
#    ΔPDF(R) = (2/π) ∫₀^{q_max} ΔsM(q) sin(qR) dq   [trapezoid rule]
#
# Real-space resolution is limited to ~π/q_max ≈ 0.21 Å at q_max = 15 Å⁻¹.
# ─────────────────────────────────────────────────────────────────────────────

self_term = np.sum(f_mat**2, axis=1)
sM        = np.where(self_term > 0, q * delta_S / self_term, 0.0)

# Hanning taper: suppresses truncation artefacts at q_max
if WINDOW == "hanning":
    u  = (q - q.min()) / (q.max() - q.min())
    sM = sM * 0.5 * (1 + np.cos(np.pi * u))

# Trapezoid-rule sine FT: ΔPDF[r] = (2/π) Σ_q sM[q] sin(r·q) Δq
dq       = np.diff(q).mean()
trap_w   = np.ones(len(q)); trap_w[0] = trap_w[-1] = 0.5
sin_mat  = np.sin(np.outer(R, q))                          # (nR, nq)
delta_PDF = (2 / np.pi) * dq * (sin_mat @ (sM * trap_w))

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Summary figure
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("Pipeline demo  —  H₂O  (B3LYP-D4 / def2-TZVPD)", fontsize=13)

ax = axes[0, 0]
ax.bar(range(1, len(omegas_ASE)+1), freqs_cm1[valid], color="steelblue", width=0.6)
ax.set_xticks(range(1, len(omegas_ASE)+1))
ax.set_xticklabels(["H-O-H bend", "sym. stretch", "asym. stretch"], fontsize=9)
ax.set_ylabel("Frequency (cm⁻¹)")
ax.set_title("Vibrational frequencies  (Hessian diagonalisation)")

ax = axes[0, 1]
for k in range(len(omegas_ASE)):
    ax.plot(time_fs, Q_t[:, k], lw=1.2,
            label=f"Mode {k+1}  ({freqs_cm1[valid][k]:.0f} cm⁻¹)")
ax.set_xlabel("τ (fs)")
ax.set_ylabel("Q_k  (√amu · Å)")
ax.set_title("Normal coordinates  Q_k(t) = A_k cos(ω_k t + φ_k)")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.plot(R, PD_static, "k-", lw=1.5)
ax.set_xlabel("R (Å)")
ax.set_ylabel("Weighted pair density  (e²)")
ax.set_title("Static PD(R)  [Z_i Z_j weighted histogram]")
ax.set_xlim(0, R.max())

ax = axes[1, 0]
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.plot(R, delta_PD_pct, color="steelblue", lw=1.5)
ax.set_xlabel("R (Å)")
ax.set_ylabel("ΔI (%)")
ax.set_title(f"ΔPD(R)  [end of trajectory vs equilibrium,  T = {T_KELVIN} K]")
ax.set_xlim(0, R.max())

ax = axes[1, 1]
ax.plot(q, S0_ground, color="firebrick", lw=1.5, label="equilibrium  (t = 0)")
ax.plot(q, S0_end,    color="navy",      lw=1.0, ls="--", label="end of trajectory")
ax.set_xlabel("q (Å⁻¹)")
ax.set_ylabel("S₀(q) (e²)")
ax.set_title("Isotropic scattering  S₀(q)  (Debye formula)")
ax.set_xlim(0, q.max())
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.plot(R, delta_PDF, color="purple", lw=1.5)
ax.set_xlabel("R (Å)")
ax.set_ylabel("ΔPDF (a.u.)")
ax.set_title("ΔPDF(R)  [sine FT of ΔsM(q)]")
ax.set_xlim(0, R.max())

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)

shutil.rmtree(scratch, ignore_errors=True)
