#!/bin/bash
# Master launch script — run from project root: bash processing/submit_all.sh
#
# Step 1: classify molecules (fast, runs on login node)
# Step 2: submit SLURM array jobs for each tier

set -e
cd "$(dirname "$0")/.."   # always run from project root

echo "=== Step 1: Classifying molecules ==="
/home/cme290/joaop/.conda/envs/SLAC/bin/python processing/classify.py

# Read actual array sizes from the saved index files
N_SMALL=$(  /home/cme290/joaop/.conda/envs/SLAC/bin/python -c \
    "import numpy as np; a=np.load('processing/indices/small_indices.npy'); print(len(a))")
N_MEDIUM=$( /home/cme290/joaop/.conda/envs/SLAC/bin/python -c \
    "import numpy as np; a=np.load('processing/indices/medium_indices.npy'); print(len(a))")
N_LARGE=$(  /home/cme290/joaop/.conda/envs/SLAC/bin/python -c \
    "import numpy as np; a=np.load('processing/indices/large_indices.npy'); print(len(a))")
N_XLARGE=$( /home/cme290/joaop/.conda/envs/SLAC/bin/python -c \
    "import numpy as np; a=np.load('processing/indices/xlarge_indices.npy'); print(len(a))")

echo ""
echo "=== Step 2: Submitting SLURM array jobs ==="

# --- Tier 1: small (PySCF) ---
if [ "$N_SMALL" -gt 0 ]; then
    MAX_SMALL=$(( N_SMALL - 1 ))
    sbatch --array=0-${MAX_SMALL}%34 processing/submit_small.slurm
    echo "  Submitted: small  (${N_SMALL} tasks)"
else
    echo "  Skipped:   small  (0 molecules)"
fi

# --- Tier 2: medium (ORCA) ---
if [ "$N_MEDIUM" -gt 0 ]; then
    MAX_MEDIUM=$(( N_MEDIUM - 1 ))
    sbatch --array=0-${MAX_MEDIUM}%17 processing/submit_medium.slurm
    echo "  Submitted: medium (${N_MEDIUM} tasks)"
else
    echo "  Skipped:   medium (0 molecules)"
fi

# --- Tier 3: large (ORCA) ---
if [ "$N_LARGE" -gt 0 ]; then
    MAX_LARGE=$(( N_LARGE - 1 ))
    sbatch --array=0-${MAX_LARGE}%17 processing/submit_large.slurm
    echo "  Submitted: large  (${N_LARGE} tasks)"
else
    echo "  Skipped:   large  (0 molecules)"
fi

# --- Tier 4: xlarge (ORCA) ---
if [ "$N_XLARGE" -gt 0 ]; then
    MAX_XLARGE=$(( N_XLARGE - 1 ))
    sbatch --array=0-${MAX_XLARGE}%17 processing/submit_xlarge.slurm
    echo "  Submitted: xlarge (${N_XLARGE} tasks)"
else
    echo "  Skipped:   xlarge (0 molecules)"
fi

echo ""
echo "All jobs submitted. Track with: squeue -u \$USER"
echo "Output files will appear in: mol_data/"
echo "Per-task logs in:            processing/logs/"