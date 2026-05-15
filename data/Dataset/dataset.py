import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from ase.db import connect

class MolScatteringDataset(Dataset):
    """
    Each item is one (trajectory, frame) pair from one molecule.
    
    Returns:
        pd_limited  : (n_R,)        - bandlimited scattering signal (input)
        pd_true     : (n_R,)        - ground truth pair distribution (target)
        R_grid      : (n_R,)        - real-space grid in Angstrom
        time_fs     : scalar        - time of this frame
        mol_features: (n_mol_feat,) - per-molecule features (atomic numbers, charge, etc.)
        omegas_ASE  : (n_modes,)    - vibrational frequencies
        V_vib       : (3N, n_modes) - normal mode eigenvectors
    """
    def __init__(
        self,
        data_dir: str | Path,
        db_path: str | Path,
        flatten_traj: bool = True, # treat each (traj, frame) as independent sample
        max_atoms: int = 350, # for padding atomic numbers
    ):
        self.data_dir = Path(data_dir)
        self.flatten_traj = flatten_traj
        self.max_atoms = max_atoms
        self.db = connect(str(db_path))
        
        self.files = sorted(self.data_dir.glob("mol_*_pd.npz"))
        
        # build flat index: list of (file_idx, traj_idx, frame_idx)
        self._index = []
        self._cache = {}
        
        for fi, fpath in enumerate(self.files):
            data = np.load(fpath)
            n_traj, n_frames, n_R = data["pd_limited"].shape
            for ti in range(n_traj):
                for fr in range(n_frames):
                    self._index.append((fi, ti, fr))
        
    def __len__(self):
        return len(self._index)
    
    def _load_file(self, fi):
        if fi not in self._cache:
            self._cache[fi] = np.load(self.files[fi])
        return self._cache[fi]
    
    def _get_mol_features(self, mol_idx: int, data) -> torch.Tensor:
        """
        Pull per-molecule features from ASE db.
        Returns a fixed-size feature vector.
        """
        row = self.db.get(int(mol_idx) + 1)
        atoms = row.toatoms()
        
        atomic_numbers = atoms.get_atomic_numbers()  # (N,)
        
        # pad to max_atoms
        Z_padded = np.zeros(self.max_atoms, dtype=np.float32)
        n = min(len(atomic_numbers), self.max_atoms)
        Z_padded[:n] = atomic_numbers[:n]
        
        n_atoms = np.array([len(atomic_numbers)], dtype=np.float32)
        charge = np.array([atoms.get_initial_charges().sum()], dtype=np.float32)
        
        mol_feat = np.concatenate([Z_padded, n_atoms, charge])
        return torch.from_numpy(mol_feat)
    
    def __getitem__(self, idx):
        fi, ti, fr = self._index[idx]
        data = self._load_file(fi)
        
        mol_idx = int(data["mol_idx"])
        
        pd_limited = torch.from_numpy(data["pd_limited"][ti, fr])   # (n_R,)
        pd_true = torch.from_numpy(data["pd_true"][ti, fr])      # (n_R,)
        R_grid = torch.from_numpy(data["R_grid"])               # (n_R,)
        time_fs = torch.tensor(data["time_fs"][fr], dtype=torch.float32)
        omegas = torch.from_numpy(data["omegas_ASE"])           # (n_modes,)
        V_vib = torch.from_numpy(data["V_vib"])                # (3N, n_modes)
        
        mol_features = self._get_mol_features(mol_idx, data)
        
        return {
            "pd_limited": pd_limited,
            "pd_true": pd_true,
            "R_grid": R_grid,
            "time_fs": time_fs,
            "mol_features": mol_features,
            "omegas_ASE": omegas,
            "V_vib": V_vib,
            "mol_idx": mol_idx,
            "traj_idx": ti,
            "frame_idx": fr,
        }