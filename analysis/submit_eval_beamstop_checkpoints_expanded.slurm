#!/bin/bash
#SBATCH --job-name=eval_bs_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=analysis/logs/eval_beamstop_checkpoints_%j.out
#SBATCH --error=analysis/logs/eval_beamstop_checkpoints_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/cme290/joaop/SLAC-atoms}"
cd "$PROJECT_DIR"
if [[ ! -f analysis/evaluate_beamstop_checkpoints.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p analysis/logs

PYTHONPATH="$PROJECT_DIR" python -u analysis/evaluate_beamstop_checkpoints.py \
  --data-dir data/trajectory_data_cap200_new \
  --R-cap 9.451833630105082 \
  --max-samples 50000 \
  --batch-size 1024 \
  --num-workers 0 \
  --checkpoint original=checkpoints/optuna/trial_0034/best_model.pt \
  --checkpoint old_finetune=checkpoints/optuna_finetune/trial_0034/best_model.pt \
  --checkpoint expanded_finetune=checkpoints/optuna_finetune_expanded/trial_0034/best_model.pt
