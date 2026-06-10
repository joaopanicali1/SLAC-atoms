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
    from models.beamstop.dataset.dataset_beamstop import BeamstopDataset
    ds = BeamstopDataset('data/trajectory_data_cap200_new')
    sample = ds[0]
    # sample['delta_S0_masked'] : (nq,)  — input with low-q zeroed
    # sample['delta_S0_norm']   : (nq,)  — full target
    # sample['mask']            : (nq,)  — 1=observed, 0=masked
"""

import time
import zipfile
import zlib
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


def _load_npz_with_retry(fpath, keys, max_retries=4, backoff=1.0):
    """Force-read selected arrays from an npz with retry on transient I/O errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            d = np.load(fpath)
            return {k: np.asarray(d[k]) for k in keys}
        except (zipfile.BadZipFile, OSError, zlib.error) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to load {fpath} after {max_retries} retries: {last_err}")


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
        max_atoms: int = 75,
        max_modes: int = 219,
        use_conditioning: bool = True,
        use_mol_features: bool = False,
        seed: int | None = None,
    ):
        self.data_dir        = Path(data_dir)
        self.q_min_lo        = q_min_lo
        self.q_min_hi        = q_min_hi
        self.max_atoms       = max_atoms
        self.max_modes       = max_modes
        self.use_conditioning = use_conditioning
        self.use_mol_features = use_mol_features
        self.rng             = np.random.default_rng(seed)

        self.files = sorted(self.data_dir.glob("mol_*_pd.npz"))
        self.loaded_files: list[Path] = []
        self.skipped_files: list[Path] = []
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
        self._index:     list[tuple[int, int, int]] = []
        self._delta_S0:  list[np.ndarray] = []  # (n_traj, n_frames, nq) per file
        # Mode data only loaded when use_conditioning=True to keep per-batch
        # payload small and avoid crashing the pin_memory IPC thread.
        self._V_vib:     list[np.ndarray] = []
        self._omegas:    list[np.ndarray] = []
        self._atom_mask: list[np.ndarray] = []
        self._mode_mask: list[np.ndarray] = []
        self._mol_features: list[np.ndarray] = []
        self.mol_feature_mean: np.ndarray | None = None
        self.mol_feature_std: np.ndarray | None = None

        n_skipped  = 0
        loaded_idx = 0
        for fpath in self.files:
            keys = ["S0", "fe"]
            if use_conditioning or use_mol_features:
                keys += ["V_vib", "omegas_ASE"]
            try:
                arrays = _load_npz_with_retry(fpath, keys=keys)
            except RuntimeError as e:
                print(f"  WARNING: skipping {fpath.name} — {e}")
                n_skipped += 1
                self.skipped_files.append(fpath)
                continue

            S0 = arrays["S0"].astype(np.float32)   # (n_traj, n_frames, nq)
            fe = arrays["fe"].astype(np.float32)   # (nq,)

            S0_ref = S0[:, 0:1, :]
            self._delta_S0.append((S0 - S0_ref) / (fe[None, None, :] + 1e-12))

            V_raw = None
            om_raw = None
            n_atoms = 0
            n_modes = 0
            if use_conditioning or use_mol_features:
                V_raw  = arrays["V_vib"].astype(np.float32)
                om_raw = arrays["omegas_ASE"].astype(np.float32)
                three_N, n_modes = V_raw.shape
                n_atoms = three_N // 3

            if use_conditioning:
                V_pad = np.zeros((3 * max_atoms, max_modes), dtype=np.float32)
                V_pad[:three_N, :n_modes] = V_raw
                om_pad = np.zeros(max_modes, dtype=np.float32)
                om_pad[:n_modes] = om_raw
                atom_mask = np.zeros(max_atoms, dtype=bool); atom_mask[:n_atoms] = True
                mode_mask = np.zeros(max_modes, dtype=bool); mode_mask[:n_modes] = True

                self._V_vib.append(V_pad)
                self._omegas.append(om_pad)
                self._atom_mask.append(atom_mask)
                self._mode_mask.append(mode_mask)

            if use_mol_features:
                self._mol_features.append(self._build_mol_features(fpath, om_raw, n_atoms, n_modes))

            n_traj, n_frames, _ = S0.shape
            for ti in range(n_traj):
                for fr in range(1, n_frames):
                    self._index.append((loaded_idx, ti, fr))
            self.loaded_files.append(fpath)
            loaded_idx += 1

        if n_skipped:
            print(f"  {n_skipped} file(s) skipped due to read errors.")
        if not self.loaded_files:
            raise RuntimeError(f"No usable mol_*_pd.npz files found in {data_dir}")
        if self.use_mol_features:
            feats = np.stack(self._mol_features).astype(np.float32)
            self.mol_feature_mean = feats.mean(axis=0).astype(np.float32)
            self.mol_feature_std = feats.std(axis=0).astype(np.float32)
            self.mol_feature_std[self.mol_feature_std < 1e-6] = 1.0
            self._mol_features = [
                ((f - self.mol_feature_mean) / self.mol_feature_std).astype(np.float32)
                for f in feats
            ]

    def _build_mol_features(self, fpath: Path, omegas: np.ndarray,
                            n_atoms: int, n_modes: int) -> np.ndarray:
        omega = np.asarray(omegas, dtype=np.float32)
        if omega.size:
            omega_stats = [
                float(np.mean(omega)),
                float(np.std(omega)),
                float(np.min(omega)),
                float(np.percentile(omega, 25)),
                float(np.median(omega)),
                float(np.percentile(omega, 75)),
                float(np.max(omega)),
                float(np.mean(omega <= np.percentile(omega, 25))),
            ]
        else:
            omega_stats = [0.0] * 8

        r_stats = [0.0, 0.0, 0.0]
        try:
            with np.load(fpath) as d:
                R_grid = np.asarray(d["R_grid"], dtype=np.float32)
                pd_true = np.asarray(d["pd_true"], dtype=np.float32)
            weights = pd_true.sum(axis=(0, 1))
            total = float(weights.sum())
            if total > 0:
                mean_R = float((weights * R_grid).sum() / total)
                var_R = float((weights * (R_grid - mean_R) ** 2).sum() / total)
                max_R = float(R_grid[np.where(weights > 0)[0].max()])
                r_stats = [mean_R, float(np.sqrt(max(var_R, 0.0))), max_R]
        except (zipfile.BadZipFile, OSError, zlib.error, KeyError, ValueError) as e:
            print(f"  WARNING: mol-feature R stats unavailable for {fpath.name} — {e}")

        return np.asarray([
            float(n_atoms),
            float(n_modes),
            *omega_stats,
            *r_stats,
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fi, ti, fr = self._index[idx]

        delta_S0_norm = self._delta_S0[fi][ti, fr]                      # (nq,) numpy
        delta_S0_norm = torch.from_numpy(delta_S0_norm)                  # (nq,)

        # Sample q_min uniformly from [q_min_lo, q_min_hi] and build mask.
        q_min     = self.rng.uniform(self.q_min_lo, self.q_min_hi)
        q_min_idx = int(torch.searchsorted(self.q, torch.tensor(q_min), right=False))
        q_min_idx = max(0, min(q_min_idx, len(self.q) - 1))

        mask = torch.ones(len(self.q), dtype=torch.float32)
        mask[:q_min_idx] = 0.0

        delta_S0_masked = delta_S0_norm * mask

        sample = {
            "delta_S0_masked": delta_S0_masked,
            "delta_S0_norm":   delta_S0_norm,
            "mask":            mask,
            "mol_idx":         fi,
            "traj_idx":        ti,
            "frame_idx":       fr,
        }
        if self.use_conditioning:
            sample["V_vib"]     = torch.from_numpy(self._V_vib[fi])
            sample["omegas"]    = torch.from_numpy(self._omegas[fi])
            sample["atom_mask"] = torch.from_numpy(self._atom_mask[fi])
            sample["mode_mask"] = torch.from_numpy(self._mode_mask[fi])
        if self.use_mol_features:
            sample["mol_features"] = torch.from_numpy(self._mol_features[fi])
        return sample
