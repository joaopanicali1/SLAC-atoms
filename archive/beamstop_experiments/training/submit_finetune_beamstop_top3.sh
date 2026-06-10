#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/training/logs"

PARTITION="${SLAC_ATOMS_PARTITION:-}"
CONDA_ENV="${SLAC_ATOMS_CONDA_ENV:-SLAC}"
TIME_LIMIT="04:00:00"
CPUS_PER_TASK="16"
MEM="64G"
BATCH_SIZE="128"
NUM_WORKERS="8"
N_EPOCHS="30"
N_WARMUP_EPOCHS="2"
MAX_RUNTIME_MIN="215"
STOP_BUFFER_MIN="12"
DRY_RUN=0

usage() {
  cat <<USAGE
Usage:
  $0 --partition PARTITION [options]

Options:
  --partition NAME       Slurm GPU partition, or set SLAC_ATOMS_PARTITION
  --conda-env NAME       Conda env to activate (default: $CONDA_ENV)
  --time-limit HH:MM:SS  Slurm wall time (default: $TIME_LIMIT)
  --cpus-per-task N      Slurm CPUs per job (default: $CPUS_PER_TASK)
  --mem SIZE             Slurm memory per job (default: $MEM)
  --dry-run              Print generated sbatch scripts without submitting
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --time-limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PARTITION" ]]; then
  echo "ERROR: pass --partition or set SLAC_ATOMS_PARTITION." >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints/optuna_finetune"

# Format:
# trial|n_channels|lambda_boundary|lambda_pd|R_cap|fine_tune_lr
# Fine-tune LRs are intentionally lower than the original Optuna LRs because
# these runs initialize from converging checkpoints and start fresh schedulers.
RUNS=(
  "31|128|0.1165924730574005|0.0025963885194745205|8.68191406196774|0.0002"
  "34|128|0.45026251076764284|1.8997338367460246e-05|9.451833630105082|0.0002"
  "24|64|0.08437558982201855|3.748427299328596e-05|8.607105433017162|0.00015"
)

for run in "${RUNS[@]}"; do
  IFS="|" read -r trial n_channels lambda_boundary lambda_pd r_cap lr <<< "$run"
  src_ckpt="$PROJECT_DIR/checkpoints/optuna/trial_$(printf "%04d" "$trial")/best_model.pt"
  dst_dir="$PROJECT_DIR/checkpoints/optuna_finetune/trial_$(printf "%04d" "$trial")"

  if [[ ! -f "$src_ckpt" ]]; then
    echo "ERROR: missing checkpoint: $src_ckpt" >&2
    exit 1
  fi

  job_script="$(mktemp "/tmp/beamstop_finetune_${trial}.XXXXXX.slurm")"
  cat > "$job_script" <<SLURM
#!/bin/bash
#SBATCH --job-name=bs_ft_${trial}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_DIR}/finetune_${trial}_%j.out
#SBATCH --error=${LOG_DIR}/finetune_${trial}_%j.err

cd ${PROJECT_DIR}
source ~/.bashrc
conda activate ${CONDA_ENV}

PYTHONPATH=${PROJECT_DIR} python -u models/beamstop/training/train_beamstop.py \\
  --n-channels ${n_channels} \\
  --lambda-boundary ${lambda_boundary} \\
  --lambda-pd ${lambda_pd} \\
  --R-cap ${r_cap} \\
  --lr ${lr} \\
  --batch-size ${BATCH_SIZE} \\
  --num-workers ${NUM_WORKERS} \\
  --no-pin-memory \\
  --n-epochs ${N_EPOCHS} \\
  --n-warmup-epochs ${N_WARMUP_EPOCHS} \\
  --max-runtime-min ${MAX_RUNTIME_MIN} \\
  --stop-buffer-min ${STOP_BUFFER_MIN} \\
  --init-checkpoint ${src_ckpt} \\
  --checkpoint-dir ${dst_dir}
SLURM

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "===== $job_script ====="
    cat "$job_script"
  else
    sbatch "$job_script"
  fi
done
