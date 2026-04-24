"""
PySCF worker — processes one molecule per SLURM array task.
Called automatically by submit_small.slurm; can also be run manually:
  python processing/process_pyscf.py <task_id> <index_file>
"""
import os
import sys
import logging
import numpy as np
from ase.db import connect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modes_lib import compute_and_save_modes_pyscf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(PROJECT_ROOT, "train_4M", "data0000.aselmdb")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "mol_data")
N_THREADS    = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))


def setup_logging(task_id: int):
    log_dir = os.path.join(PROJECT_ROOT, "processing", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"pyscf_task{task_id}.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    task_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else 0))
    index_file = os.environ.get("INDEX_FILE", sys.argv[2] if len(sys.argv) > 2 else
                                os.path.join(PROJECT_ROOT, "processing", "indices", "small_indices.npy"))

    setup_logging(task_id)
    log = logging.getLogger()

    indices = np.load(index_file)
    if task_id >= len(indices):
        log.error(f"Task {task_id} out of range (array has {len(indices)} entries).")
        sys.exit(1)

    sample_idx = int(indices[task_id])
    db_id      = sample_idx + 1  # ASE databases are 1-indexed
    output_path = os.path.join(OUTPUT_DIR, f"mol_{sample_idx:06d}_physics.npz")

    if os.path.exists(output_path):
        log.info(f"Already done: {output_path} — skipping.")
        return

    log.info(f"Task {task_id}: sample_idx={sample_idx}, db_id={db_id}, threads={N_THREADS}")

    db  = connect(DB_PATH)
    row = db.get(db_id)
    atoms = row.toatoms()
    atoms.info["charge"] = row.data.get("charge", 0)
    atoms.info["spin"]   = row.data.get("spin", 1)
    n_atoms = len(atoms)
    log.info(f"Molecule: {row.formula}  ({n_atoms} atoms)  charge={atoms.info['charge']}  spin={atoms.info['spin']}")

    try:
        compute_and_save_modes_pyscf(
            atoms=atoms,
            output_npz_path=output_path,
            n_threads=N_THREADS,
            opt=True,
        )
        log.info(f"Success: {output_path}")
    except Exception as e:
        log.exception(f"FAILED sample_idx={sample_idx}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()