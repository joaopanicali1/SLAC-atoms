"""
ORCA timing benchmark for large and xlarge molecule tiers.

Each SLURM array task runs one molecule so it gets the full wall-time window.

Usage:
  python processing/benchmark.py --tier large  --mol-idx 0 [--mpi]
  python processing/benchmark.py --tier xlarge --mol-idx 1 [--mpi]

  --tier    {large, xlarge}
  --mol-idx Index into the tier's molecule list (= SLURM_ARRAY_TASK_ID)
  --mpi     Use orca_mpi_wrapper.sh (multi-node ORCA only)

Output CSV: processing/benchmark_results_<tier>_<nprocs>cores.csv
"""
import os
import sys
import csv
import time
import shutil
import argparse
from ase.db import connect
from ase.calculators.orca import ORCA, OrcaProfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modes_lib import ORCA_SIMPLE, ORCA_MAXCORE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCA_BIN     = os.path.join(PROJECT_ROOT,
               "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg", "orca")
MPI_WRAPPER  = os.path.join(PROJECT_ROOT, "processing", "orca_mpi_wrapper.sh")
DB_PATH      = os.path.join(PROJECT_ROOT, "data/train_4m", "data0000.aselmdb")

# (target_atoms, sample_idx, db_id)
BENCHMARK_MOLECULES = {
    "large": [
        (15, 22891, 22892),
        (25, 20164, 20165),
        (40,  2408,  2409),
    ],
    "xlarge": [
        ( 55, 15628, 15629),
        ( 72,  9972,  9973),   # C18H42N6O6 — replaces RuAs (too hard); neutral, singlet, organic
        (100,  4209,  4210),
    ],
}


def parse_orca_engine_time(out_path: str) -> float:
    if not os.path.exists(out_path):
        return float("nan")
    try:
        with open(out_path) as f:
            for line in reversed(f.readlines()):
                if "TOTAL RUN TIME:" in line:
                    p = line.split("TOTAL RUN TIME:")[1].split()
                    d, h, m, s = int(p[0]), int(p[2]), int(p[4]), int(p[6])
                    return d * 86400 + h * 3600 + m * 60 + s
    except Exception:
        pass
    return float("nan")


def run_orca(atoms, nprocs: int, orca_command: str, scratch_dir: str):
    os.makedirs(scratch_dir, exist_ok=True)
    atoms.calc = ORCA(
        profile=OrcaProfile(command=orca_command),
        directory=scratch_dir,
        label="orca",
        orcasimpleinput=ORCA_SIMPLE,
        orcablocks=f"%pal nprocs {nprocs} end\n%maxcore {ORCA_MAXCORE}\n%scf MaxIter 300 end",
        charge=atoms.info["charge"],
        mult=atoms.info["spin"],
    )
    t0 = time.perf_counter()
    try:
        atoms.get_potential_energy()
    except Exception as exc:
        out_path = os.path.join(scratch_dir, "orca.out")
        if os.path.exists(out_path) and "****ORCA TERMINATED NORMALLY****" in open(out_path).read():
            pass
        else:
            raise
    wall_s = time.perf_counter() - t0
    orca_s = parse_orca_engine_time(os.path.join(scratch_dir, "orca.out"))
    return wall_s, orca_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",    required=True, choices=["large", "xlarge"])
    parser.add_argument("--mol-idx", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--mpi",     action="store_true",
                        help="Use orca_mpi_wrapper.sh (multi-node only)")
    args = parser.parse_args()

    nprocs       = int(os.environ.get("SLURM_NTASKS", 32))
    orca_command = MPI_WRAPPER if args.mpi else ORCA_BIN
    mol_list     = BENCHMARK_MOLECULES[args.tier]

    if args.mol_idx >= len(mol_list):
        print(f"mol-idx {args.mol_idx} out of range (max {len(mol_list)-1})")
        sys.exit(1)

    target_atoms, sample_idx, db_id = mol_list[args.mol_idx]

    print(f"Benchmark  tier={args.tier}  mol-idx={args.mol_idx}  "
          f"nprocs={nprocs}  command={'mpi_wrapper' if args.mpi else 'direct'}")
    print(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'not set')}")

    db     = connect(DB_PATH)
    row_db = db.get(db_id)
    atoms  = row_db.toatoms()
    atoms.info["charge"] = row_db.data.get("charge", 0)
    atoms.info["spin"]   = row_db.data.get("spin",   1)

    print(f"\nMolecule: {row_db.formula}  ({len(atoms)} atoms)  "
          f"charge={atoms.info['charge']}  spin={atoms.info['spin']}")

    scratch_dir = os.path.join(
        os.environ.get("RAM_SCRATCH",
                       os.path.join(PROJECT_ROOT, "orca_scratch_tmp")),
        f"bench_{args.tier}_{sample_idx}_{nprocs}cores"
    )

    status = "ok"
    wall_s = orca_s = float("nan")
    try:
        wall_s, orca_s = run_orca(atoms, nprocs, orca_command, scratch_dir)
        print(f"\nwall={wall_s/60:.2f} min  orca_engine={orca_s/60:.2f} min")
    except Exception as exc:
        status = f"FAILED: {exc}"
        print(f"\nFAILED: {exc}")
        # Print the last 30 lines of orca.out for diagnosis before cleaning up
        out_path = os.path.join(scratch_dir, "orca.out")
        if os.path.exists(out_path):
            print("\n--- last 30 lines of orca.out ---")
            with open(out_path) as f:
                lines = f.readlines()
            print("".join(lines[-30:]))
            print("--- end orca.out ---")
        else:
            print("(orca.out not found — ORCA may not have started)")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    out_csv = os.path.join(PROJECT_ROOT, "processing",
                           f"benchmark_results_{args.tier}_{nprocs}cores.csv")
    write_header = not os.path.exists(out_csv)
    fields = ["tier", "mol_idx", "target_atoms", "actual_atoms", "formula",
              "sample_idx", "nprocs", "wall_min", "orca_engine_min", "status"]
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(
            tier=args.tier,
            mol_idx=args.mol_idx,
            target_atoms=target_atoms,
            actual_atoms=len(atoms),
            formula=row_db.formula,
            sample_idx=sample_idx,
            nprocs=nprocs,
            wall_min=round(wall_s / 60, 2),
            orca_engine_min=round(orca_s / 60, 2),
            status=status,
        ))
    print(f"Result appended to: {out_csv}")


if __name__ == "__main__":
    main()