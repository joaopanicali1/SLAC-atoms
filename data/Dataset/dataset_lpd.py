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

import time
import zipfile
import zlib
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


def _load_npz_with_retry(fpath, keys=("dS0", "pd_true"), max_retries=4, backoff=1.0):
    """
    np.load an npz file and force-read the listed arrays, with retry on
    transient I/O errors (BadZipFile, OSError). HPC parallel filesystems
    under concurrent load from multiple jobs can return CRC errors on
    individual reads; retrying typically succeeds.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            d = np.load(fpath)
            arrays = {k: np.asarray(d[k]) for k in keys}   # force full read of each array
            return arrays
        except (zipfile.BadZipFile, OSError, zlib.error) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to load {fpath} after {max_retries} retries: {last_err}")


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
        self.loaded_files: list[Path] = []
        self.skipped_files: list[Path] = []

        if not self.files:
            raise FileNotFoundError(f"No mol_*_pd.npz files found in {data_dir}")

        # Load q and R_grid once from the first file — same for all molecules.
        ref = np.load(self.files[0])
        self.q      = torch.from_numpy(ref["q"].astype(np.float32))       # (nq,)
        self.R_grid = torch.from_numpy(ref["R_grid"].astype(np.float32))  # (nR,)

        # Preload all dS0 and pd_true arrays into memory — avoids per-sample
        # disk reads in __getitem__ which would make each epoch extremely slow.
        self._index:   list[tuple[int, int, int]] = []
        self._dS0:     list[np.ndarray] = []   # (n_traj, n_frames, nq) per file
        self._pd_true: list[np.ndarray] = []   # (n_traj, n_frames, nR) per file
        self._mol_idx: list[int]        = []

        n_skipped  = 0
        loaded_idx = 0   # index into _dS0/_pd_true (differs from fi when files are skipped)
        for fpath in self.files:
            try:
                arrays = _load_npz_with_retry(fpath, keys=("dS0", "pd_true", "mol_idx"))
            except RuntimeError as e:
                print(f"  WARNING: skipping {fpath.name} — {e}")
                n_skipped += 1
                self.skipped_files.append(fpath)
                continue

            dS0_arr = arrays["dS0"].astype(np.float32)
            pd_arr  = arrays["pd_true"].astype(np.float32)
            self._dS0.append(dS0_arr)
            self._pd_true.append(pd_arr)
            self._mol_idx.append(int(arrays["mol_idx"]))

            n_traj, n_frames, _ = dS0_arr.shape
            for ti in range(n_traj):
                for fr in range(n_frames):
                    self._index.append((loaded_idx, ti, fr))
            self.loaded_files.append(fpath)
            loaded_idx += 1

        if n_skipped:
            print(f"  {n_skipped} file(s) skipped due to read errors.")
        if not self.loaded_files:
            raise RuntimeError(f"No usable mol_*_pd.npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fi, ti, fr = self._index[idx]

        dS0     = torch.from_numpy(self._dS0[fi][ti, fr])
        pd_true = torch.from_numpy(self._pd_true[fi][ti, fr])

        return {
            "dS0":      dS0,
            "pd_true":  pd_true,
            "mol_idx":  self._mol_idx[fi],
            "traj_idx": ti,
            "frame_idx": fr,
        }
