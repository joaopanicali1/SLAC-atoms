import modal
from pathlib import Path

# ── volumes (same as training script) ─────────────────────────────────────
data_volume = modal.Volume.from_name("mol-scattering-data")
ckpt_volume = modal.Volume.from_name("mol-scattering-checkpoints")

DATA_MOUNT = "/data"
CKPT_MOUNT = "/checkpoints"
TRAJ_DIR   = "/data/trajectory_data"

# ── image (mirrors training image + scipy) ─────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "ase",
        "ase_db_backends",
    )
    .add_local_dir(".", remote_path="/app",
                   ignore=["data/", "data/train_4m/", "*.slurm", "__pycache__"])
)

app = modal.App("mol-scattering-eval", image=image)

# --- single run ---------

@app.function(
    cpu=2,
    memory=4096,
    timeout=600,  # 10 min per job is very generous for one L1 solve
    volumes={DATA_MOUNT: data_volume, CKPT_MOUNT: ckpt_volume},
)
def run_l1_single(pd_limited: list, D_flat: list, D_shape: list) -> list:
    import sys
    sys.path.insert(0, "/app")
    import numpy as np
    from processing.analyse_mol_v2 import decon_TVL1

    pd_limited = np.array(pd_limited, dtype=np.float64)
    D = np.array(D_flat, dtype=np.float64).reshape(D_shape)
    opts = dict(normalizeColumns=True, lambdaFrac=1e-2,
            rho=1.0, maxIter=200,   # was 500
            tolRel=1e-2,            # was 1e-3
            tolAbs=1e-6,
            adaptiveRho=True, debias=False)  # skip debias, saves ~30%
    
    recovered, info = decon_TVL1(pd_limited, D, lam=None, opts=opts)
    result = recovered.squeeze()
    return result.tolist()


# ── eval function ──────────────────────────────────────────────────────────

@app.function(
    cpu=16,
    memory=32768,               # 32 GB — L1 pool is memory-hungry
    timeout=60 * 60 * 10,        # 3 hours should be plenty
    volumes={
        DATA_MOUNT: data_volume,
        CKPT_MOUNT: ckpt_volume,
    },
)
def run_eval(
    frames_per_traj: int = 1,
    n_workers: int = 14,        # leave 2 cores for torch / main process
    seed: int = 42,
    unet_ckpt_name: str  = "best_model_unet.pt",
    lpd_ckpt_name: str   = "best_model_lpd.pt",
    lpd_split_name: str  = "split_lpd.npy",
    unet_split_name: str = "split.npy",
    out_subdir: str      = "eval_results",
):
    import os, sys, time
    import numpy as np
    import torch
    from scipy.signal import find_peaks

    sys.path.insert(0, "/app")

    from models.unet.code.unet import PairDensityUNet
    from models.lpd.code.learned_primal_dual import LearnedPrimalDual, build_dbt_operator
    from analysis.pd_recovery.l2_recovery import recover_l2, build_D_dict, auto_lam_l2
    from processing.analyse_mol_v2 import decon_TVL1, DBT, lorch

    PD_SCALE = 1.0
    device   = torch.device("cpu")

    out_dir = os.path.join(CKPT_MOUNT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ── copy trajectory data to local SSD (fast random access) ────────────
    import shutil
    local_traj_dir = "/tmp/trajectory_data"
    if not os.path.exists(local_traj_dir):
        print("Copying trajectory data to /tmp SSD...")
        shutil.copytree(TRAJ_DIR, local_traj_dir)
        print("Copy done.")
    else:
        print("Using existing /tmp copy.")

    # ── metrics ────────────────────────────────────────────────────────────

    def mse(pred, true):
        return float(np.mean((pred - true) ** 2))

    def mae(pred, true):
        return float(np.mean(np.abs(pred - true)))

    def r2(pred, true):
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    def peak_position_error(pred, true, R_grid, prominence=0.01):
        pred_norm = pred / (np.max(np.abs(pred)) + 1e-12)
        true_norm = true / (np.max(np.abs(true)) + 1e-12)
        pred_peaks, _ = find_peaks(pred_norm, prominence=prominence)
        true_peaks, _ = find_peaks(true_norm, prominence=prominence)
        if len(pred_peaks) == 0 or len(true_peaks) == 0:
            return float("nan")
        pred_R = R_grid[pred_peaks]
        true_R = R_grid[true_peaks]
        return float(np.mean([np.min(np.abs(rp - true_R)) for rp in pred_R]))

    def compute_all_metrics(pred, true, R_grid):
        return {
            "mse":        mse(pred, true),
            "mae":        mae(pred, true),
            "r2":         r2(pred, true),
            "peak_err_A": peak_position_error(pred, true, R_grid),
        }

    # ── classical recovery helpers ─────────────────────────────────────────

    def build_D_for_npz(npz_path):
        d      = np.load(npz_path)
        R_grid = d["R_grid"].astype(np.float64)

        if "q" in d:
            q = d["q"].astype(np.float64)
        else:
            dq = 0.1
            q  = np.arange(np.finfo(float).eps, 10 + dq, dq)
        dq     = q[1] - q[0]
        dR     = R_grid[1] - R_grid[0]
        M   = len(R_grid) + 1
        det    = lorch(q)
        D      = build_D_dict(q, R_grid, det, M, DBT)
        assert D.shape == (len(R_grid), len(R_grid)), f"D shape mismatch: {D.shape}, expected ({len(R_grid)}, {len(R_grid)})"
        return D, R_grid, q, det, M

    def _run_l2_one(pd_limited, D):
        lam = auto_lam_l2(pd_limited, D)
        return recover_l2(pd_limited, D, lam=lam)

    # ── load splits ────────────────────────────────────────────────────────
    unet_split = np.load(os.path.join(CKPT_MOUNT, unet_split_name))
    lpd_split  = np.load(os.path.join(CKPT_MOUNT, lpd_split_name))

    files   = sorted(Path(local_traj_dir).glob("mol_*_pd.npz"))
    n_files = len(files)
    n_val   = max(1, int(0.1 * n_files))

    unet_val     = set(unet_split[:n_val].tolist())
    lpd_val      = set(lpd_split[:n_val].tolist())
    intersection = sorted(unet_val & lpd_val)

    valid_intersection = []
    for fi in intersection:
        fpath = files[fi]
        keys = np.load(fpath, mmap_mode='r').keys()
        if "dS0" in keys and "q" in keys:
            valid_intersection.append(fi)
        else:
            print(f"  Skipping stale: {fpath.name}")

    print(f"Intersection val molecules: {len(intersection)}")
    print(f"Valid after stale filter:   {len(valid_intersection)}")

    # ── build eval index ───────────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    index = []

    n_traj = None
    for fi in valid_intersection:
        fpath = files[fi]
        d     = np.load(fpath)
        n_traj, n_frames, _ = d["pd_limited"].shape
        chosen_frames = rng.choice(
            np.arange(n_frames),
            size=min(frames_per_traj, n_frames),
            replace=False,
        )
        for ti in range(n_traj):
            for fr in chosen_frames:
                index.append((fpath, int(ti), int(fr)))

    print(f"Total eval samples: {len(index):,}  "
          f"({len(valid_intersection)} mol × {n_traj} traj × {frames_per_traj} frames)")

    # ── load models ────────────────────────────────────────────────────────
    unet = PairDensityUNet(
        n_R=1257, features=[32, 64, 128, 256],
        d_model=256, n_heads=8, max_atoms=50, max_modes=144,
    ).to(device)
    unet_ckpt = torch.load(os.path.join(CKPT_MOUNT, unet_ckpt_name), map_location=device)
    unet.load_state_dict(unet_ckpt["model_state_dict"])
    unet.eval()
    print(f"Loaded UNet from epoch {unet_ckpt.get('epoch', '?')}")

    ref_npz = np.load(index[0][0])
    if "q" in ref_npz:
        q_np = ref_npz["q"].astype(np.float64)
    else:
        dq   = 0.1
        q_np = np.arange(np.finfo(float).eps, 10 + dq, dq)
    R_np    = ref_npz["R_grid"].astype(np.float64)
    M_lpd   = len(R_np) + 1
    print(f"Building A_dbt (M={M_lpd})...")
    A_dbt, _ = build_dbt_operator(q_np, M_lpd, verbose=True)

    lpd = LearnedPrimalDual(
        A_dbt=A_dbt, n_primal=5, n_dual=5,
        n_iter=10, n_channels=32, kernel_size=5,
    ).to(device)
    lpd_ckpt = torch.load(os.path.join(CKPT_MOUNT, lpd_ckpt_name), map_location=device)
    lpd.load_state_dict(lpd_ckpt["model_state_dict"])
    lpd.eval()
    print(f"Loaded LPD from epoch {lpd_ckpt.get('epoch', '?')}")

    # ── build D cache ──────────────────────────────────────────────────────
    print("Building D_dict cache...")
    D_cache = {}
    for fpath, _, _ in index:
        if fpath not in D_cache:
            D_cache[fpath], _, _, _, _ = build_D_for_npz(fpath)
    print(f"Built D for {len(D_cache)} molecules.")

    # ── run L1 in parallel ─────────────────────────────────────────────────
    print(f"Spawning {len(index)} L1 containers...")
    t0 = time.perf_counter()

    # spawn, reusing futures for same D
    futures = []
    d_flat_cache = {}  # fpath -> (D_flat, D_shape) so we only tolist() once per molecule

    for fpath, ti, fr in index:
        if fpath not in d_flat_cache:
            D = D_cache[fpath]
            d_flat_cache[fpath] = (D.ravel().tolist(), list(D.shape))
        
        d = np.load(fpath)
        pd_limited = d["pd_limited"][ti, fr].astype(np.float64)
        D_flat, D_shape = d_flat_cache[fpath]
        
        futures.append(run_l1_single.spawn(
            pd_limited.tolist(),
            D_flat,
            D_shape,
        ))

    # collect with progress
    l1_preds = []
    for i, f in enumerate(futures):
        l1_preds.append(np.array(f.get()))
        if (i + 1) % 10 == 0:
            print(f"  L1 progress: {i+1}/{len(futures)} done")

    print(f"L1 done in {(time.perf_counter()-t0)/60:.1f} min")

    # ── main eval loop ─────────────────────────────────────────────────────
    methods  = ["raw_dbt", "l2", "l1", "lpd", "unet"]
    results  = {m: [] for m in methods}
    timing   = {m: [] for m in methods}
    all_preds    = {m: [] for m in methods}
    all_pd_true  = []
    all_R_grids  = []
    all_metadata = []

    for sample_i, (fpath, ti, fr) in enumerate(index):
        d          = np.load(fpath)
        pd_limited = d["pd_limited"][ti, fr].astype(np.float32)
        pd_true    = d["pd_true"][ti, fr].astype(np.float32)
        R_grid     = d["R_grid"].astype(np.float32)
        dS0        = d["dS0"][ti, fr].astype(np.float32)
        mol_idx    = int(d["mol_idx"])

        V_vib_raw  = d["V_vib"].astype(np.float32)
        omegas_raw = d["omegas_ASE"].astype(np.float32)

        max_atoms, max_modes = 50, 144
        three_N, n_modes = V_vib_raw.shape
        n_atoms = three_N // 3

        V_vib_padded  = np.zeros((3 * max_atoms, max_modes), dtype=np.float32)
        omegas_padded = np.zeros(max_modes, dtype=np.float32)
        V_vib_padded[:three_N, :n_modes] = V_vib_raw
        omegas_padded[:n_modes]          = omegas_raw

        atom_mask = torch.zeros(max_atoms, dtype=torch.bool)
        mode_mask = torch.zeros(max_modes, dtype=torch.bool)
        atom_mask[:n_atoms] = True
        mode_mask[:n_modes] = True

        all_pd_true.append(pd_true)
        all_R_grids.append(R_grid)
        all_metadata.append({"mol_idx": mol_idx, "fi": str(fpath), "ti": ti, "fr": fr})

        D = D_cache[fpath]

        # raw DBT
        t0 = time.perf_counter()
        raw_pred = pd_limited.copy()
        timing["raw_dbt"].append(time.perf_counter() - t0)
        results["raw_dbt"].append({**compute_all_metrics(raw_pred, pd_true, R_grid), "mol_idx": mol_idx})
        all_preds["raw_dbt"].append(raw_pred)

        # L2
        t0 = time.perf_counter()
        l2_pred = _run_l2_one(pd_limited.astype(np.float64), D).astype(np.float32)
        timing["l2"].append(time.perf_counter() - t0)
        results["l2"].append({**compute_all_metrics(l2_pred, pd_true, R_grid), "mol_idx": mol_idx})
        all_preds["l2"].append(l2_pred)

        # L1 (pre-computed)
        t0 = time.perf_counter()
        l1_pred = l1_preds[sample_i].astype(np.float32)
        timing["l1"].append(time.perf_counter() - t0)
        results["l1"].append({**compute_all_metrics(l1_pred, pd_true, R_grid), "mol_idx": mol_idx})
        all_preds["l1"].append(l1_pred)

        # LPD
        t0 = time.perf_counter()
        with torch.no_grad():
            lpd_out = lpd(
                torch.from_numpy(dS0).unsqueeze(0)
            ).squeeze(0).numpy()
        timing["lpd"].append(time.perf_counter() - t0)
        results["lpd"].append({**compute_all_metrics(lpd_out, pd_true, R_grid), "mol_idx": mol_idx})
        all_preds["lpd"].append(lpd_out)

        # UNet
        t0 = time.perf_counter()
        with torch.no_grad():
            unet_log = unet(
                torch.from_numpy(pd_limited).unsqueeze(0),
                torch.from_numpy(V_vib_padded).unsqueeze(0),
                torch.from_numpy(omegas_padded).unsqueeze(0),
                atom_mask.unsqueeze(0),
                mode_mask.unsqueeze(0),
            ).squeeze(0).numpy()
        unet_pred = np.expm1(unet_log).astype(np.float32)
        timing["unet"].append(time.perf_counter() - t0)
        results["unet"].append({**compute_all_metrics(unet_pred, pd_true, R_grid), "mol_idx": mol_idx})
        all_preds["unet"].append(unet_pred)

        if sample_i % 50 == 0:
            print(f"  [{sample_i}/{len(index)}] mol={mol_idx} ti={ti} fr={fr}")

    # ── aggregate ──────────────────────────────────────────────────────────
    print("\n── Aggregate metrics (mean ± std) ──")
    agg = {}
    for m in methods:
        agg[m] = {}
        for key in ["mse", "mae", "r2", "peak_err_A"]:
            vals = [r[key] for r in results[m] if not np.isnan(r[key])]
            agg[m][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg[m]["time_ms"] = {
            "mean": float(np.mean(timing[m]) * 1e3),
            "std":  float(np.std(timing[m])  * 1e3),
        }
        print(f"\n{m}:")
        for k, v in agg[m].items():
            print(f"  {k:15s}  {v['mean']:.4f} ± {v['std']:.4f}")

    # ── save to checkpoints volume ─────────────────────────────────────────
    np.save(os.path.join(out_dir, "per_sample_results.npy"),  results)
    np.save(os.path.join(out_dir, "aggregate_metrics.npy"),   agg)
    np.save(os.path.join(out_dir, "all_preds.npy"),           all_preds)
    np.save(os.path.join(out_dir, "all_pd_true.npy"),         all_pd_true)
    np.save(os.path.join(out_dir, "all_R_grids.npy"),         all_R_grids)
    np.save(os.path.join(out_dir, "all_metadata.npy"),        all_metadata)

    ckpt_volume.commit()
    print(f"\nSaved all results to {out_dir}/ and committed to volume.")
    return agg


# ── local entrypoint ──────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    frames_per_traj: int = 1,
    n_workers: int = 14,
    seed: int = 42,
):
    """
    BEFORE RUNNING — upload your local files to the checkpoints volume:

        modal volume put mol-scattering-checkpoints best_model_lpd.pt best_model_lpd.pt
        modal volume put mol-scattering-checkpoints split_lpd.npy split_lpd.npy

    Then run:

        modal run modal_eval.py

    Download results afterwards:

        modal volume get mol-scattering-checkpoints eval_results ./eval_results
    """
    handle = run_eval.spawn(
        frames_per_traj=frames_per_traj,
        n_workers=n_workers,
        seed=seed,
    )
    print(f"Spawned eval job: {handle.object_id}")
    print("Check logs with:    modal app logs mol-scattering-eval")
    print("Download results:   modal volume get mol-scattering-checkpoints eval_results ./eval_results")