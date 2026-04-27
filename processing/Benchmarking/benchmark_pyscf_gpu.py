"""
PySCF GPU benchmark — same molecules as the ORCA CPU benchmark.

Runs B3LYP-D4/def2-TZVPD geometry optimization + Hessian on one molecule
per SLURM array task, using gpu4pyscf for GPU acceleration.

Usage:
  python processing/Benchmarking/benchmark_pyscf_gpu.py --mol-idx 0
  (mol-idx is auto-read from SLURM_ARRAY_TASK_ID)

Output CSV: processing/Benchmarking/benchmark_results_pyscf_gpu.csv
"""
import os
import sys
import csv
import time
import traceback
import argparse

import numpy as np
from ase.db import connect
from pyscf import gto, lib
from pyscf.geomopt.geometric_solver import optimize

try:
    from gpu4pyscf import dft
    GPU_AVAILABLE = True
except ImportError:
    from pyscf import dft
    GPU_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modes_lib import FUNCTIONAL, DISPERSION, BASIS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH      = os.path.join(PROJECT_ROOT, "train_4M", "data0000.aselmdb")
OUT_CSV      = os.path.join(PROJECT_ROOT, "processing", "Benchmarking",
                            "benchmark_results_pyscf_gpu.csv")

# Same molecules used in the ORCA CPU benchmarks (see benchmark.py).
# (label, target_atoms, sample_idx, db_id)
ALL_MOLECULES = [
    ("C2H8N4P",        15, 22891, 22892),   # large tier
    ("C6H17NO",        25, 20164, 20165),   # large tier
    ("PtC11H18N3O6P",  40,  2408,  2409),   # large tier — hard for 1-node ORCA
    ("Br2C15H32IN2O3", 55, 15628, 15629),   # xlarge tier
    ("C18H42N6O6",     72,  9972,  9973),   # xlarge tier
    ("C25ClH49N7O17S", 100,  4209,  4210),  # xlarge tier — timed out ORCA
]


def _build_mf(mol_obj, pyscf_spin: int):
    """Build RKS or UKS mean-field object. dft is gpu4pyscf.dft or pyscf.dft depending on import."""
    mf = dft.UKS(mol_obj) if pyscf_spin > 0 else dft.RKS(mol_obj)
    mf.xc       = FUNCTIONAL
    mf.disp     = DISPERSION
    mf.max_cycle = 300
    mf.conv_tol = 1e-8
    return mf


def run_benchmark(atoms, n_cpus: int):
    """
    Returns dict with timing (seconds) for each phase and the final hessian shape.
    Raises on any failure so the caller can record status.
    """
    atom_str   = "; ".join(f"{a.symbol} {a.x} {a.y} {a.z}" for a in atoms)
    pyscf_spin = atoms.info.get("spin", 1) - 1
    charge     = atoms.info.get("charge", 0)

    lib.num_threads(n_cpus)

    mol = gto.M(atom=atom_str, basis=BASIS, charge=charge, spin=pyscf_spin, verbose=3)

    # --- geometry optimization ---
    t0 = time.perf_counter()
    mol_opt = optimize(_build_mf(mol, pyscf_spin))
    t_opt = time.perf_counter() - t0

    # --- final SCF at optimized geometry ---
    mf = _build_mf(mol_opt, pyscf_spin)
    t1 = time.perf_counter()
    mf.kernel()
    t_scf = time.perf_counter() - t1

    # --- analytical Hessian ---
    t2 = time.perf_counter()
    h = mf.Hessian().kernel()
    t_hess = time.perf_counter() - t2

    return dict(t_opt=t_opt, t_scf=t_scf, t_hess=t_hess,
                t_total=t_opt + t_scf + t_hess)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol-idx", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--cpus", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = parser.parse_args()

    if args.mol_idx >= len(ALL_MOLECULES):
        print(f"mol-idx {args.mol_idx} out of range (max {len(ALL_MOLECULES)-1})")
        sys.exit(1)

    label, target_atoms, sample_idx, db_id = ALL_MOLECULES[args.mol_idx]

    print(f"PySCF GPU benchmark  mol-idx={args.mol_idx}  target_atoms={target_atoms}  "
          f"label={label}")
    print(f"gpu4pyscf available: {GPU_AVAILABLE}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"CPU threads: {args.cpus}")

    db     = connect(DB_PATH)
    row    = db.get(db_id)
    atoms  = row.toatoms()
    atoms.info["charge"] = row.data.get("charge", 0)
    atoms.info["spin"]   = row.data.get("spin",   1)

    print(f"\nMolecule: {row.formula}  ({len(atoms)} atoms)  "
          f"charge={atoms.info['charge']}  spin={atoms.info['spin']}\n")

    status = "ok"
    timings = dict(t_opt=float("nan"), t_scf=float("nan"),
                   t_hess=float("nan"), t_total=float("nan"))
    try:
        timings = run_benchmark(atoms, n_cpus=args.cpus)
        print(f"\nopt={timings['t_opt']/60:.2f} min  "
              f"scf={timings['t_scf']/60:.2f} min  "
              f"hessian={timings['t_hess']/60:.2f} min  "
              f"total={timings['t_total']/60:.2f} min")
    except Exception:
        status = "FAILED"
        print("\nFAILED:")
        traceback.print_exc()

    write_header = not os.path.exists(OUT_CSV)
    fields = ["mol_idx", "label", "target_atoms", "actual_atoms", "formula",
              "sample_idx", "gpu_available", "n_cpus",
              "opt_min", "scf_min", "hessian_min", "total_min", "status"]
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(
            mol_idx=args.mol_idx,
            label=label,
            target_atoms=target_atoms,
            actual_atoms=len(atoms),
            formula=row.formula,
            sample_idx=sample_idx,
            gpu_available=GPU_AVAILABLE,
            n_cpus=args.cpus,
            opt_min=round(timings["t_opt"] / 60, 3),
            scf_min=round(timings["t_scf"] / 60, 3),
            hessian_min=round(timings["t_hess"] / 60, 3),
            total_min=round(timings["t_total"] / 60, 3),
            status=status,
        ))
    print(f"Result appended to: {OUT_CSV}")


if __name__ == "__main__":
    main()
