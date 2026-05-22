"""
dataset_lpd.py — Lean dataset for the Learned Primal-Dual model.

Loads mol_*_pd.npz files produced by processing/analyse_mol_v2.py and
returns one (trajectory, frame) pair per sample.

Input/output convention (matches LearnedPrimalDual):
    dS0    : (nq,)  — normalized q-space scattering signal  [model input]
    pd_true: (nR,)  — ground-truth pair density             [model target]

The q and R grids are identical across all molecules (fixed pipeline
parameters). They are exposed as dataset-level attributes rather than
per-sample tensors, since the model uses them only at build time to
construct the DBT operator A_dbt.

NPZ keys consumed (all others ignored):
    dS0      : (n_traj, n_frames, nq)
    pd_true  : (n_traj, n_frames, nR)
    q        : (nq,)
    R_grid   : (nR,)
    mol_idx  : scalar

Usage
-----
    from data.Dataset.dataset_lpd import LPDDataset
    ds = LPDDataset('data/trajectory_data_cap200')
    print(ds.q.shape, ds.R_grid.shape)   # grids for building A_dbt
    sample = ds[0]
    # sample['dS0']     : (nq,)
    # sample['pd_true'] : (nR,)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class LPDDataset(Dataset):
    """
    Lean dataset for the Learned Primal-Dual model.

    Each item is one (trajectory, frame) pair drawn from the set of
    mol_*_pd.npz files in data_dir.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing mol_*_pd.npz files.

    Attributes
    ----------
    q      : torch.Tensor (nq,)  — q grid in Å⁻¹, shared across all molecules
    R_grid : torch.Tensor (nR,)  — R grid in Å,   shared across all molecules
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("mol_*_pd.npz"))

        if not self.files:
            raise FileNotFoundError(f"No mol_*_pd.npz files found in {data_dir}")

        # Load q and R_grid once from the first file — same for all molecules.
        ref = np.load(self.files[0])
        self.q      = torch.from_numpy(ref["q"].astype(np.float32))       # (nq,)
        self.R_grid = torch.from_numpy(ref["R_grid"].astype(np.float32))  # (nR,)

        # Build flat index: list of (file_idx, traj_idx, frame_idx)
        self._index: list[tuple[int, int, int]] = []
        for fi, fpath in enumerate(self.files):
            data = np.load(fpath)
            n_traj, n_frames, _ = data["dS0"].shape
            for ti in range(n_traj):
                for fr in range(n_frames):
                    self._index.append((fi, ti, fr))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fi, ti, fr = self._index[idx]
        data = np.load(self.files[fi])

        dS0     = torch.from_numpy(data["dS0"][ti, fr].astype(np.float32))     # (nq,)
        pd_true = torch.from_numpy(data["pd_true"][ti, fr].astype(np.float32)) # (nR,)

        return {
            "dS0":      dS0,
            "pd_true":  pd_true,
            "mol_idx":  int(data["mol_idx"]),
            "traj_idx": ti,
            "frame_idx": fr,
        }
