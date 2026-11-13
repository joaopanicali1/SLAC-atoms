#!/bin/bash
#SBATCH --job-name=nsk_fig3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=analysis/logs/%x_%j.out
#SBATCH --error=analysis/logs/%x_%j.err

# The NSK paper's Fig. 3 noise / super-resolution test (O–C–O, two approaching pair distances) for
# NSK-ℓ1 and LPD v2: analysis/pd_recovery/replicate_nsk_paper.py --case fig3 (see its docstring for the setup,
# the noise model and its assumptions). Default: q_max = 4 (paper) and 10 (LPD's training range),
# 41 positions × 16 SNRs × 20 noise realisations + noise-free ≈ 26k NSK paths at 0.3–0.5 s each,
# spread over the CPUs (one single-threaded NSK worker per CPU) → ≈ 15–20 min on 16 CPUs.
#   mkdir -p analysis/logs   # once, before the first submission (Slurm opens the logs first)
#   sbatch analysis/pd_recovery/submit_nsk_fig3.slurm [extra replicate_nsk_paper.py args]
# Outputs: analysis/pd_recovery/results/lpd_v2/figures/nsk_fig3_qmax{4,10}.png, analysis/pd_recovery/results/lpd_v2/nsk_fig3.npz, summary in the log.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f analysis/pd_recovery/replicate_nsk_paper.py || ! -d data/trajectory_data_cap200 ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root (needs data/trajectory_data_cap200)." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi
source ~/.bashrc
conda activate SLAC

mkdir -p analysis/logs analysis/pd_recovery/results/lpd_v2/figures
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # parallelism is across NSK workers
export TORCH_CPP_LOG_LEVEL=ERROR PYTHONPATH="$PROJECT_DIR"

python -u analysis/pd_recovery/replicate_nsk_paper.py --case fig3 "$@"
