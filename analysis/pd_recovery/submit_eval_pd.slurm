#!/bin/bash
#SBATCH --job-name=eval_pd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=analysis/logs/%x_%j.out
#SBATCH --error=analysis/logs/%x_%j.err

# Benchmark PD-recovery methods (DBT, Tikhonov, the paper's NSK-ℓ1 with L-curve ε and with tuned ε,
# LPD checkpoints) on all validation molecules of data/splits/split_lpd.npy, 20 random frames each
# (≈ 10–15 min per condition with the exact NSK path). One job per measurement condition; the
# condition index is the first script argument (job arrays are capped at 2 tasks on this cluster):
#   for i in 0 1 2; do sbatch --job-name=eval_pd_$i analysis/pd_recovery/submit_eval_pd.slurm $i; done
# (from the repo root). Results: analysis/pd_recovery/results/lpd_v2/pd_eval_{clean,bs_q08,delta}.json.
#   0  clean     : full q-range 0–10 Å⁻¹, noise-free (the condition the LPD was trained on)
#   1  bs_q08    : beamstop at q_min = 0.8 Å⁻¹, noise-free
#   2  delta     : ΔPD = PD(t) − PD(0) task (the metric used in Table 2 of the report), noise-free
# Noise conditions are left out until a physical noise model is chosen (--snr-db's convention
# follows the report and is dominated by the low-q variance; see the eval script docstring).
# CPU-only: the work is numpy/scipy (DBT, Tikhonov, exact NSK lasso paths); the LPD forward pass is tiny.
# See the docstring of analysis/pd_recovery/eval_pd_recovery.py for the metrics.
# analysis/logs/ must exist before submission (Slurm opens the log files before this script runs):
#   mkdir -p analysis/logs

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f analysis/pd_recovery/eval_pd_recovery.py || ! -d data/trajectory_data_cap200 ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root (needs data/trajectory_data_cap200)." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi
source ~/.bashrc
conda activate SLAC

mkdir -p analysis/logs analysis/pd_recovery/results/lpd_v2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PROJECT_DIR"

NAMES=(clean bs_q08 delta)
ARGS=("" "--qmin 0.8" "--delta")
i=${1:?usage: sbatch analysis/pd_recovery/submit_eval_pd.slurm <0|1|2> [extra eval_pd_recovery.py args]}
shift
# Any further arguments are passed through and override the defaults above (argparse keeps the
# last occurrence), e.g. --methods ...,lpd_v2 --checkpoint-v2 <path> --out <other json>.

python -u analysis/pd_recovery/eval_pd_recovery.py --n-mol 0 ${ARGS[$i]} \
    --out analysis/pd_recovery/results/lpd_v2/pd_eval_${NAMES[$i]}.json "$@"
