import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from tqdm import tqdm

features = np.load("sampling/phase_space_shard0.npy")

N_atoms = features[:, 0]
pair_weight = features[:, 1]

log_pair_weight = np.log10(np.maximum(pair_weight, 1))

def bucket_sample(x, y, n_buckets_x, n_buckets_y, total_n, rng=None):
    """
    Uniformly sample total_n indices from a 2-D grid of buckets.
    Each non-empty bucket gets floor(total_n / n_occupied_buckets) samples
    and remainder distributed randomly across buckets.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x_edges = np.linspace(x.min(), x.max() + 1e-9, n_buckets_x + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-9, n_buckets_y + 1)

    xi = np.searchsorted(x_edges, x, side='right') - 1 
    yi = np.searchsorted(y_edges, y, side='right') - 1
    xi = np.clip(xi, 0, n_buckets_x - 1)
    yi = np.clip(yi, 0, n_buckets_y - 1)
    bucket_id = xi * n_buckets_y + yi

    # map molecules into each bucket
    unique_buckets, inv = np.unique(bucket_id, return_inverse=True)
    n_occupied = len(unique_buckets)

    base_per_bucket, remainder = divmod(total_n, n_occupied)
    # buckets that get one extra draw
    extra_buckets = set(rng.choice(unique_buckets, size=remainder, replace=False).tolist())

    selected = []
    for b in unique_buckets:
        mask = np.where(bucket_id == b)[0]
        quota = base_per_bucket + (1 if b in extra_buckets else 0)
        quota = min(quota, len(mask))
        chosen = rng.choice(mask, size=quota, replace=False)
        selected.append(chosen)

    return np.concatenate(selected)


def region_uniform_sample(x, y, x_min, x_max, y_min, y_max, total_n, rng=None):
    """
    Keep only points inside [x_min, x_max] × [y_min, y_max],
    then draw `total_n` uniformly at random.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    pool = np.where(mask)[0]
    print(f"  Region filter kept {len(pool):,} / {len(x):,} molecules")
    if len(pool) < total_n:
        raise ValueError(f"Only {len(pool)} molecules in region; requested {total_n}")
    return rng.choice(pool, size=total_n, replace=False)

rng = np.random.default_rng(42)

# Sample 1: 1k, full phase space, uniform buckets (20×20 = 400 buckets)
print("Sample 1 – 1k full-space bucket sample")
n_buckets = 20
idx1 = bucket_sample(log_pair_weight, N_atoms, n_buckets, n_buckets, 1_000, rng)
print(f"  Selected {len(idx1)} molecules\n")

# Sample 2: 1k, restricted region, pure uniform draw
print("Sample 2 – 1k restricted-region uniform sample")
idx2 = region_uniform_sample(
    log_pair_weight, N_atoms,
    x_min=1.5, x_max=log_pair_weight.max(),
    y_min=0,   y_max=150,
    total_n=1_000, rng=rng
)
print(f" Selected {len(idx2)} molecules\n")

# Sample 3: 10k, full phase space, uniform buckets (40×40 = 1600 buckets)
print("Sample 3 – 10k full-space bucket sample")
n_buckets_large = 40
idx3 = bucket_sample(log_pair_weight, N_atoms, n_buckets_large, n_buckets_large, 10_000, rng)
print(f"  Selected {len(idx3)} molecules\n")

# Sample 4: 10k, restricted region, pure uniform draw
print("Sample 4 – 10k restricted-region uniform sample")
idx4 = region_uniform_sample(
    log_pair_weight, N_atoms,
    x_min=1.5, x_max=log_pair_weight.max(),
    y_min=0,   y_max=150,
    total_n=10_000, rng=rng
)
print(f"  Selected {len(idx4)} molecules\n")

# Save index arrays
np.save("sample_1k_fullspace_buckets.npy",    idx1)
np.save("sample_1k_restricted_uniform.npy",   idx2)
np.save("sample_10k_fullspace_buckets.npy",   idx3)
np.save("sample_10k_restricted_uniform.npy",  idx4)
print("Saved all four index arrays.")

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = [
    "Sample 1 - 1k full-space buckets",
    "Sample 2 - 1k restricted uniform",
    "Sample 3 - 10k full-space buckets",
    "Sample 4 - 10k restricted uniform",
]
indices = [idx1, idx2, idx3, idx4]

for ax, idx, title in zip(axes.flat, indices, titles):
    ax.hexbin(
        log_pair_weight[idx], N_atoms[idx],
        gridsize=30, bins='log', cmap='viridis'
    )
    ax.set_xlabel("log10(Max Pair Weight)")
    ax.set_ylabel("Number of Atoms")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("sample_diagnostics.png", dpi=150)
plt.show()
print("Saved diagnostic plot.")