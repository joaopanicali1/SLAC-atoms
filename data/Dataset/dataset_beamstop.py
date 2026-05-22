"""
dataset_beamstop.py — Dataset for the beamstop prediction model.

Loads mol_*_pd.npz files and returns normalized differential scattering
signals with a randomly masked low-q region simulating the experimental
beamstop.

Signal definition:
    ΔS0_norm(q, t) = (S0(q, t) − S0_ref(q)) / fe(q)

where S0_ref = S0[ti, 0] is the first frame of the same trajectory, analogous
to the pre-pump ground-state measurement in a TRXSS experiment. Each trajectory
has its own reference. Frame 0 is excluded from the dataset (ΔS0=0 trivially).

This signal is zero at q=0 by physics (S0(q=0) = (ΣZi)² is geometry-
independent), and is not Lorch-windowed.

Beamstop augmentation:
    At each __getitem__ call, q_min is drawn uniformly from [q_min_lo, q_min_hi]
    and all q < q_min are set to zero in the input. The target is always the
    full ΔS0_norm. The mask (1=observed, 0=missing) is returned alongside the
    input so the model knows which values are real.

NPZ keys consumed:
    S0   : (n_traj, n_frames, nq)  — raw Debye structure factor
    fe   : (nq,)                   — effective form factor Σ_{i<j} f_if_j
    q    : (nq,)                   — q grid in Å⁻¹
    R_grid: (nR,)                  — R grid in Å

Usage
-----
    from data.Dataset.dataset_beamstop import BeamstopDataset
    ds = BeamstopDataset('data/trajectory_data_cap200')
    sample = ds[0]
    # sample['delta_S0_masked'] : (nq,)  — input with low-q zeroed
    # sample['delta_S0_norm']   : (nq,)  — full target
    # sample['mask']            : (nq,)  — 1=observed, 0=masked
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class BeamstopDataset(Dataset):
    """
    Dataset for the beamstop prediction model.

    Parameters
    ----------
    data_dir   : directory containing mol_*_pd.npz files
    q_min_lo   : lower bound of q_min augmentation range [Å⁻¹]  (default 0.5)
    q_min_hi   : upper bound of q_min augmentation range [Å⁻¹]  (default 1.0)
    seed       : random seed for reproducible masks (None = non-deterministic)

    Attributes
    ----------
    q      : (nq,)  q grid in Å⁻¹
    R_grid : (nR,)  R grid in Å
    """

    def __init__(
        self,
        data_dir: str | Path,
        q_min_lo: float = 0.5,
        q_min_hi: float = 1.0,
        seed: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.q_min_lo = q_min_lo
        self.q_min_hi = q_min_hi
        self.rng      = np.random.default_rng(seed)

        self.files = sorted(self.data_dir.glob("mol_*_pd.npz"))
        if not self.files:
            raise FileNotFoundError(f"No mol_*_pd.npz files found in {data_dir}")

        # Load q and R_grid once — identical across all molecules.
        ref = np.load(self.files[0])
        self.q      = torch.from_numpy(ref["q"].astype(np.float32))        # (nq,)
        self.R_grid = torch.from_numpy(ref["R_grid"].astype(np.float32))   # (nR,)
        self._dq    = float(self.q[1] - self.q[0])

        # Build flat (file, traj, frame) index and precompute ΔS0_norm per file.
        # Reference for trajectory ti is its own first frame S0[ti, 0].
        # Frame 0 is excluded — ΔS0[ti, 0] = 0 trivially, nothing to learn.
        self._index: list[tuple[int, int, int]] = []
        self._delta_S0: list[np.ndarray] = []   # one (n_traj, n_frames, nq) per file

        for fi, fpath in enumerate(self.files):
            data = np.load(fpath)
            S0   = data["S0"].astype(np.float32)   # (n_traj, n_frames, nq)
            fe   = data["fe"].astype(np.float32)   # (nq,)

            # Per-trajectory reference: first frame of each trajectory.
            S0_ref = S0[:, 0:1, :]                 # (n_traj, 1, nq)

            delta_S0_norm = (S0 - S0_ref) / (fe[None, None, :] + 1e-12)
            self._delta_S0.append(delta_S0_norm)   # (n_traj, n_frames, nq)

            n_traj, n_frames, _ = S0.shape
            for ti in range(n_traj):
                for fr in range(1, n_frames):      # skip fr=0 (trivially zero)
                    self._index.append((fi, ti, fr))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fi, ti, fr = self._index[idx]

        delta_S0_norm = self._delta_S0[fi][ti, fr]                      # (nq,) numpy
        delta_S0_norm = torch.from_numpy(delta_S0_norm)                  # (nq,)

        # Sample q_min uniformly from [q_min_lo, q_min_hi] and build mask.
        q_min     = self.rng.uniform(self.q_min_lo, self.q_min_hi)
        q_min_idx = int(q_min / self._dq)
        q_min_idx = max(0, min(q_min_idx, len(self.q) - 1))

        mask = torch.ones(len(self.q), dtype=torch.float32)
        mask[:q_min_idx] = 0.0

        delta_S0_masked = delta_S0_norm * mask

        return {
            "delta_S0_masked": delta_S0_masked,   # (nq,) — model input
            "delta_S0_norm":   delta_S0_norm,      # (nq,) — target
            "mask":            mask,               # (nq,) — 1=observed, 0=missing
            "mol_idx":         fi,
            "traj_idx":        ti,
            "frame_idx":       fr,
        }
