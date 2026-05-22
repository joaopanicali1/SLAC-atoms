"""
Shared library: normal-mode computation functions (extracted from trajectory_pipeline_cluster.ipynb).

Both functions run:  B3LYP-D4 / def2-TZVPD, geometry optimization, analytical Hessian.
Output NPZ keys:
  R_0          – optimized Cartesian coordinates, flattened (Angstrom)
  masses_amu   – atomic masses repeated × 3 (amu)
  hessian_eV_Ang2 – 3N×3N Hessian in eV/Å²
  omegas_ASE   – vibrational angular frequencies, filtered (ASE units = eV^0.5 / (amu^0.5 Å))
  V_vib        – corresponding normal-mode eigenvectors (mass-weighted, 3N × n_modes)
  eigen_val    – all 3N eigenvalues of the mass-weighted Hessian (before filtering)
"""
import os
import numpy as np
from ase import units
from ase.calculators.orca import ORCA, OrcaProfile
from pyscf import gto, dft, lib
from pyscf.geomopt.geometric_solver import optimize

# Shared DFT settings — keep identical between PySCF and ORCA
FUNCTIONAL  = "b3lyp"
DISPERSION  = "d4"
BASIS       = "def2-tzvpd"          # PySCF spelling
ORCA_BASIS  = "def2-TZVPD"         # ORCA spelling (same basis set)
# RIJCOSX: RI approximation for Coulomb + chain-of-spheres for exact exchange.
# Benchmark error vs. conventional B3LYP: < 1 cm⁻¹ for harmonic frequencies
# (typically 0.1–0.3 cm⁻¹) — negligible relative to B3LYP's ~30–50 cm⁻¹ vs. experiment.
# def2/J is the Weigend Coulomb-fitting auxiliary basis, standard with RIJCOSX.
ORCA_SIMPLE      = f"B3LYP D4 RIJCOSX {ORCA_BASIS} def2/J TightSCF TightOpt FREQ"
# Looser optimization — used as a fallback for molecules that fail with TightOpt
ORCA_SIMPLE_OPT  = f"B3LYP D4 RIJCOSX {ORCA_BASIS} def2/J TightSCF Opt FREQ"

# %maxcore: memory per MPI process (MB).
# CPU nodes (icme-cpu-11-*) have 32 GB RAM (32000 MB per SLURM).
# 32 cores × 800 MB = 25.6 GB → ~6 GB headroom for OS and file buffers.
# For multi-node jobs each node still runs 32 processes, so per-node usage is the same.
ORCA_MAXCORE = 800
FREQ_LB     = 5.0    # cm⁻¹  — discard near-zero numerical noise
FREQ_UB     = 4000.0 # cm⁻¹  — covers all physical molecular vibrations


# ---------------------------------------------------------------------------
# ORCA helpers
# ---------------------------------------------------------------------------

def parse_orca_hessian(filepath: str) -> np.ndarray:
    """Parse an ORCA .hess file and return the symmetric Hessian matrix."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_line = next((i + 2 for i, line in enumerate(lines) if "$hessian" in line), None)
    if start_line is None:
        raise ValueError(f"Hessian section not found in: {filepath}")

    dim = int(lines[start_line - 1].strip())
    hessian = np.zeros((dim, dim))
    col_start = 0
    i = start_line

    while i < len(lines):
        parts = lines[i].split()
        if not parts or "$" in lines[i]:
            break
        if not any("." in p for p in parts):
            col_start = int(parts[0])
        else:
            row_idx = int(parts[0])
            for j, val in enumerate(map(float, parts[1:])):
                hessian[row_idx, col_start + j] = val
        i += 1

    return (hessian + hessian.T) / 2


def _parse_xyz_last_frame(xyz_path: str) -> np.ndarray:
    """Return the coordinates of the last frame in an XYZ file (Angstrom, flattened)."""
    with open(xyz_path) as f:
        lines = f.readlines()
    last_coords = None
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
            coord_lines = lines[i + 2 : i + 2 + n]
            coords = [[float(v) for v in l.split()[1:4]] for l in coord_lines]
            last_coords = np.array(coords, dtype=float).flatten()
            i += n + 2
        except (ValueError, IndexError):
            i += 1
    return last_coords


def _get_orca_R0(scratch_dir: str, atoms_fallback) -> np.ndarray:
    """
    Return the optimized geometry from orca.xyz (written by TightOpt).
    Falls back to the ASE atoms positions if the file is absent (no optimization run).
    """
    xyz_path = os.path.join(scratch_dir, "orca.xyz")
    if os.path.exists(xyz_path):
        coords = _parse_xyz_last_frame(xyz_path)
        if coords is not None:
            return coords
    return atoms_fallback.get_positions().flatten()


# ---------------------------------------------------------------------------
# Internal utilities shared by both functions
# ---------------------------------------------------------------------------

def _n_rigid(n_atoms: int, positions: np.ndarray) -> int:
    """
    Number of zero-frequency rigid-body modes to strip before normal-mode analysis.

    For linear molecules (all atoms collinear) only 2 rotational axes exist, giving
    5 rigid modes (3 translational + 2 rotational) and 3N-5 vibrational modes.
    For non-linear polyatomics the standard 6 rigid modes (3+3) apply, giving 3N-6.

    Linearity is detected via SVD: if the second singular value of the centred
    position matrix is negligible compared to the first, the molecule is linear.
    """
    if n_atoms == 1:
        return 3
    if n_atoms == 2:
        return 5
    pos = np.asarray(positions).reshape(n_atoms, 3)
    centered = pos - pos.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    # s[0] is the "spread" along the principal axis; s[1] measures off-axis spread.
    # Ratio < 1e-4 reliably identifies collinear molecules.
    is_linear = s[1] < 1e-4 * s[0]
    return 5 if is_linear else 6


def _filter_modes(eigenvalues, V, n_rigid, freq_lb, freq_ub):
    """
    Strip rigid-body modes, convert eigenvalues to angular frequencies, and apply
    frequency window [freq_lb, freq_ub] in cm⁻¹.

    np.abs before sqrt converts small negative eigenvalues (numerical noise from
    the near-zero rigid-body modes) to near-zero frequencies that are then removed
    by freq_lb. Genuine imaginary frequencies from an unconverged/non-minimum
    geometry are also folded to positive — they appear as anomalously low values
    that the freq_lb filter usually catches.  The full eigen_val array (saved
    separately) lets you diagnose such cases post-hoc.

    Conversion: 1 ASE-frequency unit = 521.47 cm⁻¹
      Derived from: ν [cm⁻¹] = √(eV/amu/Å²) / (2π c)
                              = √(1.60218e-19/(1.66054e-27 × 1e-20)) / (2π × 2.99792e10)
                              = 521.47 cm⁻¹
    """
    omegas = np.sqrt(np.abs(eigenvalues[n_rigid:]))
    freqs_cm1 = omegas * 521.47
    valid = np.where((freqs_cm1 > freq_lb) & (freqs_cm1 < freq_ub))[0]
    return omegas[valid], V[:, n_rigid:][:, valid]


def _save(output_npz_path, R_0, masses_amu, hess_ASE, eigenvalues, omegas_filt, V_filt):
    """
    Write output atomically: save to a .tmp file first, then rename.
    On POSIX filesystems os.replace is atomic within the same filesystem, so a
    job killed mid-write leaves behind only a .tmp file — the next run sees no
    completed output and re-processes the molecule correctly.
    """
    out_dir = os.path.dirname(os.path.abspath(output_npz_path))
    os.makedirs(out_dir, exist_ok=True)
    # np.savez_compressed appends .npz if the filename doesn't already end in it,
    # so the tmp file must also end in .npz for os.replace to find the right file.
    tmp_path = output_npz_path[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        R_0=R_0,
        masses_amu=masses_amu,
        hessian_eV_Ang2=hess_ASE,
        omegas_ASE=omegas_filt,
        V_vib=V_filt,
        eigen_val=eigenvalues,
    )
    os.replace(tmp_path, output_npz_path)
    print(f"Saved: {output_npz_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_and_save_modes_pyscf(
    atoms,
    output_npz_path: str,
    n_threads: int = 16,
    opt: bool = True,
    functional: str = FUNCTIONAL,
    dispersion: str = DISPERSION,
    basis: str = BASIS,
    freq_lb: float = FREQ_LB,
    freq_ub: float = FREQ_UB,
):
    atom_str = "; ".join(f"{a.symbol} {a.x} {a.y} {a.z}" for a in atoms)
    lib.num_threads(n_threads)

    pyscf_spin = atoms.info.get("spin", 1) - 1
    charge     = atoms.info.get("charge", 0)

    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=pyscf_spin, verbose=0)

    def _build(mol_obj):
        mf = dft.UKS(mol_obj) if pyscf_spin > 0 else dft.RKS(mol_obj)
        mf.xc = functional
        if dispersion:
            mf.disp = dispersion
        mf.max_cycle = 300
        mf.conv_tol  = 1e-8
        return mf

    mol_final = optimize(_build(mol)) if opt else mol

    mf = _build(mol_final)
    mf.kernel()

    N = mol_final.natm
    raw_hess_4d = mf.Hessian().kernel()
    raw_hess_2d = raw_hess_4d.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    conv     = units.Hartree / (units.Bohr ** 2)
    hess_ASE = raw_hess_2d * conv

    R_0        = mol_final.atom_coords() * units.Bohr   # optimized coords in Å
    masses_amu = np.repeat([mol_final.atom_mass_list()], 3)

    # M⁻½ H M⁻½ via broadcasting — O(N²) instead of two O(N³) matmuls.
    # Equivalent to np.diag(w) @ hess_ASE @ np.diag(w) but avoids the N×N diagonal matrix.
    w              = 1.0 / np.sqrt(masses_amu)
    tilde_H        = hess_ASE * w[:, None] * w[None, :]
    eigenvalues, V = np.linalg.eigh(tilde_H)

    omegas_filt, V_filt = _filter_modes(eigenvalues, V, _n_rigid(N, R_0), freq_lb, freq_ub)
    _save(output_npz_path, R_0.flatten(), masses_amu, hess_ASE, eigenvalues, omegas_filt, V_filt)


def compute_and_save_modes_orca(
    atoms,
    output_npz_path: str,
    orca_path: str,
    scratch_dir: str,
    nprocs: int = 32,
    freq_lb: float = FREQ_LB,
    freq_ub: float = FREQ_UB,
    orca_keywords: str = ORCA_SIMPLE,
):
    os.makedirs(scratch_dir, exist_ok=True)

    atoms.calc = ORCA(
        profile=OrcaProfile(command=orca_path),
        directory=scratch_dir,
        label="orca",
        orcasimpleinput=orca_keywords,
        orcablocks=f"%pal nprocs {nprocs} end\n%maxcore {ORCA_MAXCORE}\n%scf MaxIter 300 end",
        charge=atoms.info["charge"],
        mult=atoms.info["spin"],
    )

    atoms.get_potential_energy()

    # R_0: use ORCA's optimized geometry from orca.xyz (written by TightOpt),
    # not atoms.get_positions() which still holds the pre-optimization coordinates.
    R_0        = _get_orca_R0(scratch_dir, atoms)
    masses_amu = np.repeat(atoms.get_masses(), 3)

    hess_path = os.path.join(scratch_dir, "orca.hess")
    raw_hess  = parse_orca_hessian(hess_path)

    conv     = units.Hartree / (units.Bohr ** 2)
    hess_ASE = raw_hess * conv

    # M⁻½ H M⁻½ via broadcasting — O(N²) instead of two O(N³) matmuls.
    # Equivalent to np.diag(w) @ hess_ASE @ np.diag(w) but avoids the N×N diagonal matrix.
    w              = 1.0 / np.sqrt(masses_amu)
    tilde_H        = hess_ASE * w[:, None] * w[None, :]
    eigenvalues, V = np.linalg.eigh(tilde_H)

    n = len(atoms)
    omegas_filt, V_filt = _filter_modes(eigenvalues, V, _n_rigid(n, R_0), freq_lb, freq_ub)
    _save(output_npz_path, R_0, masses_amu, hess_ASE, eigenvalues, omegas_filt, V_filt)