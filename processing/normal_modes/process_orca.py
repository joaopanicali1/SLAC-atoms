"""ORCA worker for normal-mode extraction.

Processes one molecule per Slurm array task and writes
``data/mol_data_new/mol_XXXXXX_physics.npz``.  The molecule id is read from one of
the index arrays in ``processing/normal_modes/indices``.

Manual usage from the repository root:
    python processing/normal_modes/process_orca.py 0 processing/normal_modes/indices/medium_indices.npy
"""
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.db import connect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modes_lib import compute_and_save_modes_orca, ORCA_SIMPLE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get(
    "SLAC_ATOMS_ASE_DB",
    PROJECT_ROOT / "data" / "train_4m" / "data0000.aselmdb",
))
OUTPUT_DIR = Path(os.environ.get(
    "SLAC_ATOMS_MOL_DATA_DIR",
    PROJECT_ROOT / "data" / "mol_data_new",
))
ORCA_PATH = Path(os.environ.get(
    "ORCA_EXE",
    PROJECT_ROOT / "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg" / "orca",
))
# Use SLURM_NTASKS if available; fall back to 32
NPROCS = int(os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_NTASKS_PER_NODE", 32)))
# Allow overriding ORCA keywords via environment variable (e.g. for retry with Opt)
ORCA_KEYWORDS = os.environ.get("ORCA_KEYWORDS", ORCA_SIMPLE)


def make_scratch_dir(task_id: int) -> str:
    """Return a unique, per-task scratch directory, preferring the RAM disk."""
    job_id  = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    name    = f"orca_{job_id}_{task_id}"
    ram_dir = os.environ.get("RAM_SCRATCH")
    if ram_dir and Path(ram_dir).is_dir():
        return str(Path(ram_dir) / name)
    return str(PROJECT_ROOT / "orca_scratch_tmp" / name)


def setup_logging(task_id: int):
    log_dir = PROJECT_ROOT / "processing" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"orca_task{task_id}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    task_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[1] if len(sys.argv) > 1 else 0))
    index_file = os.environ.get(
        "INDEX_FILE",
        sys.argv[2] if len(sys.argv) > 2 else
        PROJECT_ROOT / "processing" / "normal_modes" / "indices" / "medium_indices.npy",
    )

    setup_logging(task_id)
    log = logging.getLogger()

    indices = np.load(index_file)
    if task_id >= len(indices):
        log.error(f"Task {task_id} out of range (array has {len(indices)} entries).")
        sys.exit(1)

    sample_idx  = int(indices[task_id])
    db_id       = sample_idx + 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"mol_{sample_idx:06d}_physics.npz"

    if output_path.exists():
        log.info(f"Already done: {output_path} — skipping.")
        return

    log.info(f"Task {task_id}: sample_idx={sample_idx}, db_id={db_id}, nprocs={NPROCS}")

    db    = connect(str(DB_PATH))
    row   = db.get(db_id)
    atoms = row.toatoms()
    atoms.info["charge"] = row.data.get("charge", 0)
    atoms.info["spin"]   = row.data.get("spin", 1)
    n_atoms = len(atoms)
    log.info(f"Molecule: {row.formula}  ({n_atoms} atoms)  charge={atoms.info['charge']}  spin={atoms.info['spin']}")

    scratch_dir = make_scratch_dir(task_id)
    log.info(f"Scratch dir: {scratch_dir}")

    try:
        compute_and_save_modes_orca(
            atoms=atoms,
            output_npz_path=str(output_path),
            orca_path=str(ORCA_PATH),
            scratch_dir=scratch_dir,
            nprocs=NPROCS,
            orca_keywords=ORCA_KEYWORDS,
        )
        log.info(f"Success: {output_path}")
    except Exception as e:
        log.exception(f"FAILED sample_idx={sample_idx}: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
