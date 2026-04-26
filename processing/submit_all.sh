#!/bin/bash
# Master launch script — run from project root: bash processing/submit_all.sh
#
# Resource allocation (from benchmarking):
#   Tier 1 small   (≤6  atoms): PySCF, 1 node, 16 threads  → 2 jobs/node → %30 concurrent
#   Tier 2 medium  (7-12 atoms): ORCA, 1 node, 32 cores     → 1 job/node  → %15 concurrent
#   Tier 3 large  (13-50 atoms): ORCA, 2 nodes, 64 cores    → %7 concurrent (7×2=14 nodes)
#   Tier 4 xlarge  (>50 atoms):  ORCA, 2 nodes, 64 cores    → %7 concurrent (7×2=14 nodes)
# Nodes 16 and 18 excluded (only 16 GB /dev/shm, too small for ORCA scratch).

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

# --- Tier 1: small (PySCF, 1 node, 16 threads) ---
if [ "$N_SMALL" -gt 0 ]; then
    MAX_SMALL=$(( N_SMALL - 1 ))
    sbatch --array=0-${MAX_SMALL}%30 processing/submit_small.slurm
    echo "  Submitted: small  (${N_SMALL} tasks, 1 node each)"
else
    echo "  Skipped:   small  (0 molecules)"
fi

# --- Tier 2: medium (ORCA, 1 node, 32 cores) ---
if [ "$N_MEDIUM" -gt 0 ]; then
    MAX_MEDIUM=$(( N_MEDIUM - 1 ))
    sbatch --array=0-${MAX_MEDIUM}%15 processing/submit_medium.slurm
    echo "  Submitted: medium (${N_MEDIUM} tasks, 1 node each)"
else
    echo "  Skipped:   medium (0 molecules)"
fi

# --- Tier 3: large (ORCA, 2 nodes, 64 cores) ---
if [ "$N_LARGE" -gt 0 ]; then
    MAX_LARGE=$(( N_LARGE - 1 ))
    sbatch --array=0-${MAX_LARGE}%7 processing/submit_large.slurm
    echo "  Submitted: large  (${N_LARGE} tasks, 2 nodes each)"
else
    echo "  Skipped:   large  (0 molecules)"
fi

# --- Tier 4: xlarge (ORCA, 2 nodes, 64 cores) ---
if [ "$N_XLARGE" -gt 0 ]; then
    MAX_XLARGE=$(( N_XLARGE - 1 ))
    sbatch --array=0-${MAX_XLARGE}%7 processing/submit_xlarge.slurm
    echo "  Submitted: xlarge (${N_XLARGE} tasks, 2 nodes each)"
else
    echo "  Skipped:   xlarge (0 molecules)"
fi

echo ""
echo "All jobs submitted. Track with: squeue -u \$USER"
echo "Output files will appear in: mol_data/"
echo "Per-task logs in:            processing/logs/"
echo ""
echo "Note: molecules near 100+ atoms in the xlarge tier may time out."
echo "      Identify them with: grep FAILED processing/logs/xlarge_*.err"
echo "      and resubmit individually with a longer --time if needed."