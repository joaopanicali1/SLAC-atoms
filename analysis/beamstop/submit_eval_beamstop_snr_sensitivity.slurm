#!/bin/bash
#SBATCH --job-name=eval_bs_snr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-pascal
#SBATCH --gres=gpu:1
#SBATCH --output=analysis/logs/eval_beamstop_snr_%j.out
#SBATCH --error=analysis/logs/eval_beamstop_snr_%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f analysis/beamstop/evaluate_beamstop_snr_sensitivity.py || ! -d data/trajectory_data_cap200_new ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi

source ~/.bashrc
conda activate SLAC-p100

mkdir -p analysis/logs
RESULT_DIR="analysis/beamstop/final_results/snr_sensitivity_${SLURM_JOB_ID}"

PYTHONPATH="$PROJECT_DIR" python -u analysis/beamstop/evaluate_beamstop_snr_sensitivity.py \
  --data-dir data/trajectory_data_cap200_new \
  --R-cap 9.451833630105082 \
  --snrs 1,2,5,10,20,50,100 \
  --qmin-lo 0.5 \
  --qmin-hi 1.0 \
  --max-samples 0 \
  --batch-size 256 \
  --num-workers 0 \
  --direct-checkpoint checkpoints/optuna_finetune_expanded/trial_0034/best_model.pt \
  --feature-checkpoint models/beamstop/code/beamstop_final.pt \
  --output-dir "${RESULT_DIR}"
