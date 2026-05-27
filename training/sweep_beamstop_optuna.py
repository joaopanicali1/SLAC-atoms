#!/usr/bin/env python3
"""
sweep_beamstop_optuna.py — Bayesian hyperparameter search for the beamstop
model using Optuna (TPE sampler) over a continuous parameter space.

Multiple SLURM jobs coordinate through a shared SQLite study file — each job
claims one trial, trains, and reports val_loss back.  TPE uses completed
trials to suggest better parameters for subsequent jobs.

Usage
-----
    # 1. Create the study (run once)
    python training/sweep_beamstop_optuna.py --create

    # 2. Submit N jobs (each runs 1 trial)
    python training/sweep_beamstop_optuna.py --submit --n-jobs 20

    # 3. Check progress / add more trials anytime
    python training/sweep_beamstop_optuna.py --compare

    # Run one trial locally (testing)
    python training/sweep_beamstop_optuna.py --run-trial
"""

import argparse
import os
import subprocess
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────

STUDY_NAME   = "beamstop_v2"
PROJECT_DIR  = Path(os.environ.get(
    "SLAC_ATOMS_PROJECT_DIR",
    Path(__file__).resolve().parents[1],
)).resolve()
SWEEP_DIR    = PROJECT_DIR / "checkpoints/optuna"
DB_PATH      = SWEEP_DIR / "beamstop.db"
LOG_DIR      = PROJECT_DIR / "training/logs"
PARTITION    = "gpu-volta"
DATA_DIR     = PROJECT_DIR / "data/trajectory_data_cap200"

# ─── Search space ───────────────────────────────────────────────────────────
#
# n_channels      : categorical — powers of 2 in a sensible range
# lambda_boundary : log-uniform [0.05, 20.0]  — slope continuity penalty
# lambda_pd       : log-uniform [1e-5, 5e-2]  — Tikhonov on R-space intermediate
# R_cap           : uniform     [3.0, 12.0]   — architectural band-limit (Å)
# lr              : log-uniform [3e-4, 5e-3]  — peak LR after warmup
#
# lr is included because it interacts with n_channels: larger models have
# higher gradient variance and often benefit from a smaller peak LR.
# Log-uniform for all continuous penalties/rates so TPE explores weak and
# strong values at equal resolution on a log scale.

SUGGEST = dict(
    n_channels      = dict(type="categorical", choices=[64, 128]),
    lambda_boundary = dict(type="float", low=0.05, high=20.0,  log=True),
    lambda_pd       = dict(type="float", low=1e-5,  high=5e-2, log=True),
    R_cap           = dict(type="float", low=3.0,   high=12.0, log=False),
    lr              = dict(type="float", low=3e-4,  high=5e-3, log=True),
)

# Fixed across all trials
FIXED = dict(
    batch_size    = 128,
    lambda_smooth = 0.0,
    lambda_masked = 10.0,
    num_workers   = 0,
)


# ─── Optuna objective (runs inside each SLURM job) ──────────────────────────

def objective(trial):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from data.Dataset.dataset_beamstop import BeamstopDataset
    from models.beamstop import build_sinc_operator
    from training.train_beamstop import (
        train, compute_dataset_R_max, build_truncated_R_grid, FixedMaskSubset,
    )

    # Suggest parameters
    n_channels      = trial.suggest_categorical("n_channels",      SUGGEST["n_channels"]["choices"])
    lambda_boundary = trial.suggest_float("lambda_boundary",
                                          SUGGEST["lambda_boundary"]["low"],
                                          SUGGEST["lambda_boundary"]["high"],
                                          log=True)
    lambda_pd       = trial.suggest_float("lambda_pd",
                                          SUGGEST["lambda_pd"]["low"],
                                          SUGGEST["lambda_pd"]["high"],
                                          log=True)
    R_cap           = trial.suggest_float("R_cap",
                                          SUGGEST["R_cap"]["low"],
                                          SUGGEST["R_cap"]["high"],
                                          log=False)
    lr              = trial.suggest_float("lr",
                                          SUGGEST["lr"]["low"],
                                          SUGGEST["lr"]["high"],
                                          log=True)

    print(f"\nTrial {trial.number}: n_channels={n_channels}, "
          f"lambda_boundary={lambda_boundary:.3g}, "
          f"lambda_pd={lambda_pd:.2e}, R_cap={R_cap:.2f}, lr={lr:.2e}")

    # Dataset — same split as training (seed 42)
    dataset = BeamstopDataset(DATA_DIR, q_min_lo=0.5, q_min_hi=1.0,
                              max_atoms=75, max_modes=219,
                              use_conditioning=False)

    n_files = len(dataset.loaded_files)
    n_val   = max(1, int(0.1 * n_files))
    rng     = np.random.default_rng(42)
    file_indices = np.arange(n_files)
    rng.shuffle(file_indices)
    val_files   = set(file_indices[:n_val])
    train_files = set(file_indices[n_val:])

    train_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in train_files]
    val_idx   = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]

    # B_sinc with trial's R_cap
    q_np         = dataset.q.numpy()
    R_max_data   = compute_dataset_R_max(dataset.loaded_files, margin=1.2)
    R_max        = min(R_max_data, R_cap)
    dR           = float(dataset.R_grid[1] - dataset.R_grid[0])
    R_grid_trunc = build_truncated_R_grid(R_max, dR=dR)
    B_sinc       = build_sinc_operator(q_np, R_grid_trunc, verbose=False)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=FIXED["batch_size"], shuffle=True,
        num_workers=FIXED["num_workers"], pin_memory=False,
    )
    val_loader = DataLoader(
        FixedMaskSubset(dataset, val_idx, seed=12345),
        batch_size=FIXED["batch_size"], shuffle=False,
        num_workers=FIXED["num_workers"], pin_memory=False,
    )

    ckpt_dir = SWEEP_DIR / f"trial_{trial.number:04d}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    _, best_val = train(
        train_loader    = train_loader,
        val_loader      = val_loader,
        B_sinc          = B_sinc,
        n_channels      = n_channels,
        checkpoint_dir  = str(ckpt_dir),
        use_conditioning = False,
        lambda_masked   = FIXED["lambda_masked"],
        lambda_smooth   = FIXED["lambda_smooth"],
        lambda_boundary = lambda_boundary,
        lambda_pd       = lambda_pd,
        lr              = lr,
    )
    return best_val


# ─── CLI commands ────────────────────────────────────────────────────────────

def cmd_create(args):
    """Create (or re-open) the Optuna study."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Path(SWEEP_DIR).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name   = STUDY_NAME,
        storage      = f"sqlite:///{DB_PATH}",
        direction    = "minimize",
        load_if_exists = True,
        sampler      = optuna.samplers.TPESampler(seed=0),
    )
    print(f"Study '{STUDY_NAME}' ready.")
    print(f"  DB       : {DB_PATH}")
    print(f"  Trials   : {len(study.trials)} total")
    print(f"  Completed: {sum(1 for t in study.trials if t.state.name == 'COMPLETE')}")


def cmd_submit(args):
    """Create the study and submit N SLURM jobs, each running one trial."""
    cmd_create(args)

    partition = args.partition or PARTITION
    script = f"""\
#!/bin/bash
#SBATCH --job-name=bs_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --output={LOG_DIR}/optuna_%j.out
#SBATCH --error={LOG_DIR}/optuna_%j.err

cd {PROJECT_DIR}
source ~/.bashrc
conda activate SLAC

mkdir -p {LOG_DIR}

PYTHONPATH={PROJECT_DIR} python -u training/sweep_beamstop_optuna.py --run-trial
"""

    tmp = Path("/tmp/sweep_optuna.slurm")
    tmp.write_text(script)

    print(f"\nSubmitting {args.n_jobs} jobs to '{partition}' …")
    submitted = 0
    for _ in range(args.n_jobs):
        res = subprocess.run(["sbatch", str(tmp)], capture_output=True, text=True)
        if res.returncode == 0:
            submitted += 1
            print(f"  job {res.stdout.strip().split()[-1]}")
        else:
            print(f"  FAILED: {res.stderr.strip()}")

    print(f"\n{submitted}/{args.n_jobs} jobs submitted.")
    print(f"Monitor : squeue -u $USER")
    print(f"Results : python training/sweep_beamstop_optuna.py --compare")


def cmd_run_trial(args):
    """Run exactly one Optuna trial (called inside each SLURM job)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name = STUDY_NAME,
        storage    = f"sqlite:///{DB_PATH}",
    )
    study.optimize(objective, n_trials=1)


def cmd_compare(args):
    """Print a sorted results table of all completed trials."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.load_study(
            study_name = STUDY_NAME,
            storage    = f"sqlite:///{DB_PATH}",
        )
    except Exception as e:
        print(f"Could not load study: {e}")
        return

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    running  = [t for t in study.trials if t.state.name == "RUNNING"]
    failed   = [t for t in study.trials if t.state.name == "FAIL"]

    if not complete:
        print(f"No completed trials yet. ({len(running)} running, {len(failed)} failed)")
        return

    complete.sort(key=lambda t: t.value)

    print(f"\n{'Rank':<5}  {'val_loss':>12}  {'nc':>4}  {'lb':>8}  {'pd':>8}  {'rc':>6}  {'lr':>8}  trial")
    print("─" * 82)
    for i, t in enumerate(complete, 1):
        p = t.params
        marker = "  ← best" if i == 1 else ""
        print(
            f"{i:<5}  {t.value:>12.6e}"
            f"  {p['n_channels']:>4}"
            f"  {p['lambda_boundary']:>8.3f}"
            f"  {p['lambda_pd']:>8.1e}"
            f"  {p['R_cap']:>6.2f}"
            f"  {p['lr']:>8.2e}"
            f"  #{t.number}{marker}"
        )

    best = complete[0]
    print(f"\nBest so far — trial #{best.number}  val_loss={best.value:.6e}")
    print("  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v}")

    print(f"\n{len(complete)} completed | {len(running)} running | {len(failed)} failed")
    if running:
        print(f"More results coming — re-run --compare when jobs finish.")
    print(f"\nTo submit more trials:  python training/sweep_beamstop_optuna.py --submit --n-jobs N")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optuna sweep for the beamstop model.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--create",    action="store_true", help="Create/open the Optuna study")
    g.add_argument("--submit",    action="store_true", help="Submit N SLURM jobs")
    g.add_argument("--run-trial", action="store_true", help="Run one trial (called by SLURM)")
    g.add_argument("--compare",   action="store_true", help="Show results table")
    p.add_argument("--n-jobs",    type=int, default=25,
                   help="Number of SLURM jobs to submit (default 25, ~5× n_params)")
    p.add_argument("--partition", type=str, default=None,
                   help=f"SLURM partition (default: {PARTITION})")
    p.add_argument("--study-name", type=str, default=STUDY_NAME,
                   help=f"Optuna study name (default: {STUDY_NAME})")
    args = p.parse_args()
    STUDY_NAME = args.study_name

    if args.create:
        cmd_create(args)
    elif args.submit:
        cmd_submit(args)
    elif args.run_trial:
        cmd_run_trial(args)
    elif args.compare:
        cmd_compare(args)
