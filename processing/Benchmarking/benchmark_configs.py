"""
benchmark_configs.py — Run all ORCA configurations for ONE molecule.

Called once per molecule by submit_benchmark_configs.slurm.
The molecule is selected via --mol (or MOL env var set by sbatch --export).
Results are saved as a .npz file in processing/Benchmarking/configs_results/.

After all molecule jobs finish, run benchmark_configs_analyze.py
to aggregate results, compute frequency deviations, and generate plots.

Configurations compared
-----------------------
  B3LYP_RIJCOSX : B3LYP D4 RIJCOSX def2-TZVPD def2/J  (production approx.)
  wB97M-V       : wB97M-V def2-TZVPD RIJCOSX            (higher-quality functional)

Usage
-----
  python processing/Benchmarking/benchmark_configs.py --mol H2O_3atoms
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import tempfile
import numpy as np
from ase import units as ase_units
from ase.db import connect as ase_connect
from ase.data import atomic_masses, atomic_numbers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORCA_BIN     = os.path.join(PROJECT_ROOT,
               "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg", "orca")
MPI_DIR      = os.path.join(PROJECT_ROOT, "openmpi-4.1.8_install")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "processing", "Benchmarking", "configs_results")
TRAIN_4M_DIR = os.path.join(PROJECT_ROOT, "train_4M")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_from_db(db_file, row_id):
    db    = ase_connect(os.path.join(TRAIN_4M_DIR, db_file))
    row   = db.get(id=row_id)
    atoms = row.toatoms()
    syms  = atoms.get_chemical_symbols()
    pos   = atoms.positions
    charge    = int(getattr(row, "charge", 0) or 0)
    n_elec    = int(sum(atoms.numbers)) - charge
    magmoms   = atoms.get_initial_magnetic_moments()
    tot_mag   = abs(round(float(magmoms.sum())))
    mult      = int(tot_mag) + 1
    # Ensure (n_elec + mult) is odd — the parity rule for valid spin states.
    # Databases often store zero magnetic moments; this corrects to the lowest
    # valid multiplicity when the stored value is inconsistent.
    if (n_elec + mult) % 2 == 0:
        mult += 1
    xyz = "\n".join(f"{s}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}"
                    for s, p in zip(syms, pos))
    return {"charge": charge, "mult": mult, "xyz": xyz}

# ── Production settings ───────────────────────────────────────────────────────
ORCA_MAXCORE = 4000
NPROCS       = 1
FREQ_LB      = 5.0
FREQ_UB      = 4000.0

# ── ORCA configurations ───────────────────────────────────────────────────────
CONFIGS = {
    "B3LYP_RIJCOSX": "B3LYP D4 RIJCOSX def2-TZVPD def2/J TightSCF TightOpt FREQ",
    "wB97M-V":       "wB97M-V def2-TZVPD RIJCOSX TightSCF Opt NUMFREQ",
    # NUMFREQ required: wB97M-V uses non-local VV10 dispersion (NL), for which
    # ORCA does not yet implement an analytical Hessian. NUMFREQ computes
    # frequencies via finite differences of analytical gradients.
}

# ── Benchmark molecules (loaded from train_4M) ────────────────────────────────
MOLECULES = {
    "H2O_3atoms":      _load_from_db("data0001.aselmdb", 27726),
    "CH3NO2_7atoms":   _load_from_db("data0000.aselmdb", 7),
    "C3HAuO2_7atoms":  _load_from_db("data0000.aselmdb", 25703),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_orca(tag, keywords, mol_info, scratch_root):
    scratch   = os.path.join(scratch_root, tag)
    os.makedirs(scratch, exist_ok=True)

    inp = (
        f"! {keywords}\n"
        f"%pal nprocs {NPROCS} end\n"
        f"%maxcore {ORCA_MAXCORE}\n"
        f"%scf MaxIter 300 end\n"
        f"%geom MaxIter 500 end\n"
        f"* xyz {mol_info['charge']} {mol_info['mult']}\n"
        f"{mol_info['xyz']}\n"
        f"*\n"
    )

    inp_path  = os.path.join(scratch, "mol.inp")
    out_path  = os.path.join(scratch, "mol.out")
    hess_path = os.path.join(scratch, "mol.hess")
    xyz_path  = os.path.join(scratch, "mol.xyz")

    with open(inp_path, "w") as fh:
        fh.write(inp)

    env = os.environ.copy()
    env["PATH"]            = f"{os.path.dirname(ORCA_BIN)}:{MPI_DIR}/bin:{env.get('PATH','')}"
    env["LD_LIBRARY_PATH"] = f"{os.path.dirname(ORCA_BIN)}:{MPI_DIR}/lib:{env.get('LD_LIBRARY_PATH','')}"

    with open(out_path, "w") as fout:
        result = subprocess.run(
            [ORCA_BIN, inp_path],
            stdout=fout, stderr=subprocess.STDOUT,
            cwd=scratch, env=env,
        )

    if result.returncode != 0 or not os.path.exists(hess_path):
        # Copy output to logs for post-mortem inspection
        log_path = os.path.join(PROJECT_ROOT, "processing", "logs",
                                f"bench_configs_{tag}.out")
        shutil.copy(out_path, log_path)
        raise RuntimeError(f"ORCA failed for {tag} — see {log_path}")

    return hess_path, xyz_path


def parse_hessian(filepath):
    with open(filepath) as fh:
        lines = fh.readlines()
    start = next(i + 2 for i, l in enumerate(lines) if "$hessian" in l)
    dim = int(lines[start - 1].strip())
    H, col_start = np.zeros((dim, dim)), 0
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
    return (H + H.T) / 2


def read_last_xyz(xyz_file):
    with open(xyz_file) as fh:
        lines = fh.readlines()
    last_sym, last_coords = [], []
    i = 0
    while i < len(lines):
        try:
            n      = int(lines[i].strip())
            sym    = [lines[i+2+k].split()[0] for k in range(n)]
            coords = [[float(v) for v in lines[i+2+k].split()[1:4]] for k in range(n)]
            last_sym, last_coords = sym, coords
            i += n + 2
        except (ValueError, IndexError):
            i += 1
    return last_sym, np.array(last_coords)


def compute_modes(hess_path, xyz_path):
    H_au      = parse_hessian(hess_path)
    H_eV      = H_au * (ase_units.Hartree / ase_units.Bohr**2)
    symbols, _ = read_last_xyz(xyz_path)
    masses_amu = np.repeat([atomic_masses[atomic_numbers[s]] for s in symbols], 3)
    inv_sqrt_m = 1.0 / np.sqrt(masses_amu)
    H_mw       = H_eV * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    eigenvalues, V_all = np.linalg.eigh(H_mw)
    n_rigid   = 6
    omegas    = np.sqrt(np.abs(eigenvalues[n_rigid:]))
    freqs     = omegas * 521.47
    valid     = np.where((freqs > FREQ_LB) & (freqs < FREQ_UB))[0]
    return freqs[valid], V_all[:, n_rigid:][:, valid]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--mol", default=os.environ.get("MOL"),
                    choices=list(MOLECULES), help="Molecule to benchmark")
args = parser.parse_args()

if args.mol is None:
    parser.error("Specify a molecule with --mol or MOL env var")

mol_name = args.mol
mol_info = MOLECULES[mol_name]

out_path = os.path.join(RESULTS_DIR, f"{mol_name}.npz")
if os.path.exists(out_path):
    print(f"Results already exist: {out_path} — skipping.")
    sys.exit(0)

print(f"Molecule: {mol_name}  |  NPROCS={NPROCS}")

scratch_root = tempfile.mkdtemp(prefix=f"bench_{mol_name}_")

freqs_all = {}
V_all_    = {}

for config_name, keywords in CONFIGS.items():
    print(f"  {config_name} ...")
    t0 = time.time()
    tag = f"{mol_name}__{config_name}"
    hess_path, xyz_path = run_orca(tag, keywords, mol_info, scratch_root)
    freqs, V = compute_modes(hess_path, xyz_path)
    freqs_all[config_name] = freqs
    V_all_[config_name]    = V
    elapsed = time.time() - t0
    print(f"    {len(freqs)} modes,  {freqs.min():.1f}–{freqs.max():.1f} cm⁻¹  ({elapsed/60:.1f} min)")

shutil.rmtree(scratch_root, ignore_errors=True)

# Save results as .npz for the analysis script to load
np.savez(out_path,
         mol_name=mol_name,
         config_names=list(CONFIGS.keys()),
         **{f"freqs_{k}": v for k, v in freqs_all.items()},
         **{f"V_{k}":     v for k, v in V_all_.items()})

print(f"Saved: {out_path}")
