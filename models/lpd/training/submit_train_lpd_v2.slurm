#!/bin/bash
#SBATCH --job-name=train_lpd_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-dgx
#SBATCH --gres=gpu:8
#SBATCH --output=models/lpd/training/logs/train_lpd_v2_%j.out
#SBATCH --error=models/lpd/training/logs/train_lpd_v2_%j.err

# Train LearnedPrimalDualV2 (models/lpd/code/learned_primal_dual_v2.py), noise-free with beamstop
# augmentation, on the cap200 trajectories; molecule split from data/splits/split_lpd.npy (read only).
# Checkpoints: models/lpd/code/checkpoints/lpd_v2/{best_model.pt, latest_checkpoint.pt}.
#
# Resources: gpu-dgx is allocated as a whole node (OverSubscribe=EXCLUSIVE: 8 V100-SXM2-32GB,
# 80 CPUs, 500 GB) whatever is requested, so all 8 GPUs are used: one process per GPU (torchrun +
# DistributedDataParallel). --batch-size is the GLOBAL batch (default 256 → 32 frames per GPU), so
# the optimisation is the same as on 1 GPU. CPUs: 8 × (main process + 4 DataLoader workers).
# Memory: every process preloads the training set (≈ 10 GB: p as float16 + S0) → ≈ 100 GB in total;
# 240 GB leaves room for the page cache while the files are read.
# Speed reference: 1 GPU, batch 256 = 0.25 s/step ≈ 56 min/epoch (job 26720).
#
# --resume continues from models/lpd/code/checkpoints/lpd_v2/latest_checkpoint.pt if it exists
# (also one written by a run on a different number of GPUs), so the same script can be resubmitted after a time-out.
# Delete or rename that directory to start a fresh run. Extra arguments are passed through to
# train_lpd_v2.py, e.g.
#   mkdir -p models/lpd/training/logs   # once, before the first submission
#   sbatch models/lpd/training/submit_train_lpd_v2.slurm --epochs 25
# The released checkpoint models/lpd/code/lpd_v2_final.pt is epoch 8 of this run (best validation W1).

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [[ ! -f models/lpd/training/train_lpd_v2.py || ! -d data/trajectory_data_cap200 ]]; then
    echo "ERROR: submit this job from the SLAC-atoms repo root (needs data/trajectory_data_cap200)." >&2
    echo "PROJECT_DIR resolved to: $PROJECT_DIR" >&2
    exit 1
fi
source ~/.bashrc
conda activate SLAC

mkdir -p models/lpd/training/logs models/lpd/code/checkpoints/lpd_v2
export OMP_NUM_THREADS=2
export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv
# GPUs actually allocated to this job (nvidia-smi lists every GPU of the node).
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES → $NGPU processes"

torchrun --standalone --nproc_per_node="$NGPU" models/lpd/training/train_lpd_v2.py --num-workers 4 --resume "$@"
