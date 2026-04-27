"""
Classify molecules from the sample by atom count, save index arrays per tier.
Run from project root: python processing/classify.py
"""
import os
import numpy as np

SAMPLE_PATH   = "sampling/samples/sample_1k_restricted_uniform.npy"
FEATURES_PATH = "sampling/phase_space_shard0.npy"
OUT_DIR       = "processing/indices"

# ── Molecule size filter ──────────────────────────────────────────────────────
# Molecules with more than MAX_ATOMS are excluded from processing.
# Benchmark results (B3LYP-D4/def2-TZVPD, 2-node/64-core, 4-hr wall time):
#   ≤75 atoms: all completed within 4 hours (~4 days total for 725 molecules)
#   76-150 atoms: most hit the wall time; 4+ nodes causes ORCA OOM errors
# Set to None to process the full sample.
MAX_ATOMS = 12
# ─────────────────────────────────────────────────────────────────────────────

sample_idx = np.load(SAMPLE_PATH)
features   = np.load(FEATURES_PATH)
n_atoms    = features[sample_idx, 0].astype(int)

if MAX_ATOMS is not None:
    excluded   = (n_atoms > MAX_ATOMS).sum()
    sample_idx = sample_idx[n_atoms <= MAX_ATOMS]
    n_atoms    = n_atoms[n_atoms <= MAX_ATOMS]
    print(f"Excluding {excluded} molecules with >{MAX_ATOMS} atoms  "
          f"({excluded} / {excluded + len(sample_idx)} total)")
else:
    print("No atom-count filter applied (MAX_ATOMS = None)")

small_mask  = n_atoms <= 6
medium_mask = (n_atoms >= 7)  & (n_atoms <= 12)
large_mask  = (n_atoms >= 13) & (n_atoms <= 50)
xlarge_mask = n_atoms > 50

small_idx  = sample_idx[small_mask]
medium_idx = sample_idx[medium_mask]
large_idx  = sample_idx[large_mask]
xlarge_idx = sample_idx[xlarge_mask]

print(f"Total molecules to process: {len(sample_idx)}")
print(f"  Tier 1 — small   (≤6   atoms, PySCF 16-core):  {len(small_idx):4d}")
print(f"  Tier 2 — medium  (7-12  atoms,  ORCA 32-core):  {len(medium_idx):4d}")
print(f"  Tier 3 — large   (13-50 atoms,  ORCA 64-core):  {len(large_idx):4d}")
print(f"  Tier 4 — xlarge  (51-{MAX_ATOMS if MAX_ATOMS else 150} atoms,  ORCA 64-core):  {len(xlarge_idx):4d}")

os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/small_indices.npy",  small_idx)
np.save(f"{OUT_DIR}/medium_indices.npy", medium_idx)
np.save(f"{OUT_DIR}/large_indices.npy",  large_idx)
np.save(f"{OUT_DIR}/xlarge_indices.npy", xlarge_idx)

print(f"\nSaved index arrays to {OUT_DIR}/")
print("Array sizes for SLURM --array flags:")
print(f"  small:  0-{max(len(small_idx)-1, 0)}")
print(f"  medium: 0-{len(medium_idx)-1}")
print(f"  large:  0-{len(large_idx)-1}")
print(f"  xlarge: 0-{len(xlarge_idx)-1}")