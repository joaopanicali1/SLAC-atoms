"""
dataset_lpd_v2.py — Dataset for LearnedPrimalDualV2 (models/lpd/code/learned_primal_dual_v2.py).

One item = one (trajectory, frame) of one molecule from the mol_*_pd.npz files produced by
processing/analyse_mol_v2.py. Unlike LPDDataset (v1), items carry the RAW scattering S0 plus
the molecule's Σf² and fe, so the training loop can re-process the signal per sample — e.g. with
a random beamstop q_min — exactly as an experiment would (see process_scattering in the model).

Returned per item
    S0           : (nq,)  Debye scattering of the frame (true per-atom form factors)
    self_scatter : (nq,)  Σ_a f_a(q)²
    fe           : (nq,)  effective form factor used for normalisation
    p            : (nR,)  normalised ground-truth PD, pd_true / Σ_{a<b} Z_a Z_b  (Σ p = 1)
    mol_idx, traj_idx, frame_idx : metadata

Memory: everything is preloaded. p is stored as float16 (relative precision 1e-3, far below the
accuracy at stake) to halve the footprint: ≈ 10 GB for p + 1.6 GB for S0 for the 662-molecule
cap200 set.

Usage
-----
    from models.lpd.dataset.dataset_lpd_v2 import LPDv2Dataset, split_files
    val_files, train_files = split_files('data/trajectory_data_cap200', 'data/splits/split_lpd.npy')
    ds = LPDv2Dataset(train_files)
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def split_files(data_dir, split_path, val_frac: float = 0.1):
    """
    Molecule-level split identical to models/lpd/training/train_lpd.py and analysis/pd_recovery/eval_pd_recovery.py:
    split_path holds a permutation of the sorted file list; the first val_frac are validation.
    The split file is only read, never rewritten.
    """
    files = sorted(Path(data_dir).glob("mol_*_pd.npz"))
    perm = np.load(split_path)
    if len(perm) != len(files):
        raise RuntimeError(f"{split_path} has {len(perm)} entries but {len(files)} files exist")
    n_val = max(1, int(val_frac * len(files)))
    return [files[i] for i in perm[:n_val]], [files[i] for i in perm[n_val:]]


def _load(fpath, keys, max_retries=4, backoff=1.0):
    last = None
    for attempt in range(max_retries):
        try:
            d = np.load(fpath)
            return {k: np.asarray(d[k]) for k in keys}
        except (zipfile.BadZipFile, OSError) as e:      # transient parallel-FS read errors
            last = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"failed to load {fpath}: {last}")


class LPDv2Dataset(Dataset):
    def __init__(self, files, verbose: bool = True):
        self.files = [Path(f) for f in files]
        ref = np.load(self.files[0])
        self.q = ref["q"].astype(np.float32)
        self.R_grid = ref["R_grid"].astype(np.float32)

        self._S0, self._p, self._ss, self._fe, self._mol = [], [], [], [], []
        index = []
        t0 = time.time()
        for fi, f in enumerate(self.files):
            a = _load(f, ("S0", "pd_true", "self_scatter", "fe", "mol_idx"))
            pd = a["pd_true"].astype(np.float32)
            self._p.append((pd / pd.sum(-1, keepdims=True)).astype(np.float16))
            self._S0.append(a["S0"].astype(np.float32))
            self._ss.append(a["self_scatter"].astype(np.float32))
            self._fe.append(a["fe"].astype(np.float32))
            self._mol.append(int(a["mol_idx"]))
            n_traj, n_frames, _ = pd.shape
            t, fr = np.meshgrid(np.arange(n_traj), np.arange(n_frames), indexing="ij")
            index.append(np.stack([np.full(t.size, fi), t.ravel(), fr.ravel()], 1))
            if verbose and (fi + 1) % 100 == 0:
                print(f"  loaded {fi + 1}/{len(self.files)} files ({time.time() - t0:.0f} s)", flush=True)
        self._index = np.concatenate(index).astype(np.int32)       # (n_samples, 3)
        if verbose:
            print(f"  {len(self._index):,} samples from {len(self.files)} molecules "
                  f"({time.time() - t0:.0f} s)", flush=True)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, ti, fr = (int(v) for v in self._index[idx])
        return {
            "S0": torch.from_numpy(self._S0[fi][ti, fr]),
            "self_scatter": torch.from_numpy(self._ss[fi]),
            "fe": torch.from_numpy(self._fe[fi]),
            "p": torch.from_numpy(self._p[fi][ti, fr].astype(np.float32)),
            "mol_idx": self._mol[fi],
            "traj_idx": ti,
            "frame_idx": fr,
        }
