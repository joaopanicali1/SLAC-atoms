import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
# from ase.db import connect

class MolScatteringDataset(Dataset):
    """
    Each item is one (trajectory, frame) pair from one molecule.
    
    Returns:
        pd_limited: (n_R,) - bandlimited scattering signal (input)
        pd_true: (n_R,) - ground truth pair distribution (target)
        R_grid: (n_R,) - real-space grid in Angstrom
        time_fs: scalar - time of this frame
        mol_features: (n_mol_feat,) - per-molecule features (atomic numbers, charge, etc.)
        omegas_ASE: (n_modes,) - vibrational frequencies
        V_vib: (3N, n_modes) - normal mode eigenvectors
        mode_mask: (max_modes,) - True for real modes, False for padding
        atom_mask: (max_atoms,) - True for real atoms, False for padding
        mol_idx: int - metadata for debugging
        traj_idx: int - metadata for debugging
        frame_idx: int - metadata for debugging
    """
    def __init__(
        self,
        data_dir: str or Path,
        # db_path: str or Path,
        flatten_traj: bool = True, # treat each (traj, frame) as independent sample
        max_atoms: int = 50, # for padding atomic numbers
        max_modes: int = 144,
        index_cache_path: str or Path = None,
        preload: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.flatten_traj = flatten_traj
        self.max_atoms = max_atoms
        self.max_modes = max_modes
        # print("Connecting to ASE db...")
        # self.db = connect(str(db_path))
        # print("Connected to ASE db.")
        
        print("Scanning for files...")
        self.files = sorted(self.data_dir.glob("mol_*_pd.npz"))
        print(f"Found {len(self.files)} files.")
        
        # build flat index: list of (file_idx, traj_idx, frame_idx)
        self._index = []
        # self._mol_features_cache = {}
        
        if index_cache_path and Path(index_cache_path).exists():
            print("Loading cached index...")
            self._index = [tuple(x) for x in np.load(index_cache_path, allow_pickle=True).tolist()]
            print(f"Loaded {len(self._index):,} samples from cache.")
        else:
            for fi, fpath in enumerate(self.files):
                if fi % 100 == 0:
                    print(f"Indexing file {fi}/{len(self.files)}...")
                data = np.load(fpath)
                n_traj, n_frames, n_R = data["pd_limited"].shape
                for ti in range(n_traj):
                    for fr in range(n_frames):
                        self._index.append((fi, ti, fr))
            
            if index_cache_path:
                print("Saving index cache...")
                np.save(index_cache_path, np.array(self._index))
                print("Index cached.")

        self._file_cache = {}
        if preload:
            print("Preloading all files into RAM...")
            for fi, fpath in enumerate(self.files):
                if fi % 100 == 0:
                    print(f"  {fi}/{len(self.files)}", flush=True)
                d = np.load(fpath)
                # extract arrays immediately — don't store the NpzFile handle
                self._file_cache[fi] = {
                    "pd_limited": d["pd_limited"],
                    "pd_true": d["pd_true"],
                    "R_grid": d["R_grid"],
                    "time_fs": d["time_fs"],
                    "omegas_ASE": d["omegas_ASE"],
                    "V_vib": d["V_vib"],
                    "mol_idx": d["mol_idx"],
                }
            print("Preload complete.")
        
    def __len__(self):
        return len(self._index)
    
    def _load_file(self, fi):
        if fi not in self._file_cache:
            d = np.load(self.files[fi])
            self._file_cache[fi] = {
                "pd_limited": d["pd_limited"],
                "pd_true": d["pd_true"],
                "R_grid": d["R_grid"],
                "time_fs": d["time_fs"],
                "omegas_ASE": d["omegas_ASE"],
                "V_vib": d["V_vib"],
                "mol_idx": d["mol_idx"],
            }
        return self._file_cache[fi]
    
    def _compute_mol_features(self, mol_idx: int) -> torch.Tensor:
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
        
        pd_limited = torch.from_numpy(data["pd_limited"][ti, fr]) # (n_R,)
        pd_true = torch.log1p(torch.from_numpy(data["pd_true"][ti, fr]))
        R_grid = torch.from_numpy(data["R_grid"]) # (n_R,)
        time_fs = torch.tensor(data["time_fs"][fr], dtype=torch.float32)

        omegas_raw = data["omegas_ASE"] # (n_modes,)
        V_vib_raw = data["V_vib"] # (3N, n_modes)

        three_N, n_modes = V_vib_raw.shape
        n_atoms = three_N // 3
        atom_mask = torch.zeros(self.max_atoms, dtype = torch.bool)
        atom_mask[:n_atoms] = True


        V_vib_padded = np.zeros((3 * self.max_atoms, self.max_modes), dtype=np.float32)
        V_vib_padded[:three_N, :n_modes] = V_vib_raw

        omegas_padded = np.zeros(self.max_modes, dtype=np.float32)
        omegas_padded[:n_modes] = omegas_raw

        mode_mask = torch.zeros(self.max_modes, dtype=torch.bool)
        mode_mask[:n_modes] = True
        
        return {
            "pd_limited": pd_limited,
            "pd_true": pd_true,
            "R_grid": R_grid,
            "time_fs": time_fs,
            # "mol_features": self._mol_features_cache[mol_idx],
            "omegas_ASE": torch.from_numpy(omegas_padded),
            "V_vib": torch.from_numpy(V_vib_padded),
            "mode_mask": mode_mask,
            "atom_mask": atom_mask,
            "mol_idx": mol_idx,
            "traj_idx": ti,
            "frame_idx": fr,
        }