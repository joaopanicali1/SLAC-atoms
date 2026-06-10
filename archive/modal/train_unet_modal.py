import modal

# ── volumes ────────────────────────────────────────────────────────────────
# Create these once:  modal volume create mol-scattering-data
#                     modal volume create mol-scattering-checkpoints
data_volume  = modal.Volume.from_name("mol-scattering-data",        create_if_missing=True)
ckpt_volume  = modal.Volume.from_name("mol-scattering-checkpoints", create_if_missing=True)

DATA_MOUNT  = "/data"
CKPT_MOUNT  = "/checkpoints"
DB_PATH = "/data/train4M/data0000.aselmdb"
TRAJ_DIR = "/data/trajectory_data"

# ── image ──────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "numpy",
        "scipy",
        "ase",
        # "ase_db_backends",
        "tqdm",
    )
    # copy the project source into the container image
    .add_local_dir(".", remote_path="/app", ignore=["data/trajectory_data", "data/mol_data_new", "data/mol_data_new", "data/train_4m/", "*.slurm", "__pycache__"])
)

app = modal.App("mol-scattering-train", image=image)


@app.function(
    gpu="A10G",                 # swap for "H100" or "A100" as needed
    timeout=60 * 60 * 12,      # 12-hour max (adjust for your dataset size)
    volumes={
        "/data": data_volume,
        "/checkpoints": ckpt_volume,
    },
    # secrets=[modal.Secret.from_name("wandb-secret")],   # uncomment for W&B
)
def train_remote(
    n_epochs: int      = 200,
    lr: float          = 4e-3,
    batch_size: int    = 256,
    n_warmup_epochs: int = 5,
    n_workers: int     = 4,
    n_r: int           = 1257,
    max_atoms: int     = 50,
    max_modes: int     = 144,
    seed: int          = 42,
    resume: bool       = True,   # pick up from latest_checkpoint.pt if it exists
    debug: bool        = False,
    max_steps_per_epoch: int = 100,
):
    import os, math, sys
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torch.optim import AdamW

    sys.path.insert(0, "/app")

    from models.unet.dataset.dataset import MolScatteringDataset
    from models.unet.code.unet import PairDensityUNet
    from models.unet.training.train_unet import warmup_cosine_scheduler, run_epoch

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # copying data locally
    import shutil
    local_traj_dir = "/tmp/trajectory_data"
    if not os.path.exists(local_traj_dir):
        print("Copying trajectory data to local SSD...")
        shutil.copytree(TRAJ_DIR, local_traj_dir)
        print("Copy complete.")
        total, used, free = shutil.disk_usage("/tmp")
        print(f"Disk usage: {used/1e9:.1f}GB used, {free/1e9:.1f}GB free")
    else:
        print("Local SSD copy already exists, skipping.")

    # -- If in debug mode -----------------------------------------------------
    if debug:
        n_epochs = 2
        batch_size = 4
        n_workers = 0

    # ── dataset ─────────────────────────────────────────────────────────────
    dataset = MolScatteringDataset(
        data_dir=local_traj_dir,
        #db_path=DB_PATH,
        max_atoms=max_atoms,
        max_modes=max_modes,
        index_cache_path=os.path.join(CKPT_MOUNT, "dataset_index.npy"),
        preload=True,
    )
    ckpt_volume.commit()
    print(f"Dataset: {len(dataset):,} samples from {len(dataset.files)} molecules")

    # Sanity checks after loading the dataset
    print("DB_PATH:", DB_PATH)
    print("TRAJ_DIR:", TRAJ_DIR)
    print("num files:", len(dataset.files))
    print("num samples:", len(dataset))
    print("first index sample:", dataset._index[0] if len(dataset._index) > 0 else None)

    # deterministic train/val split on molecules (not frames)
    n_files = len(dataset.files)
    n_val   = max(1, int(0.1 * n_files))
    rng     = np.random.default_rng(seed)
    file_indices = np.arange(n_files)
    rng.shuffle(file_indices)

    split_path = os.path.join(CKPT_MOUNT, "split.npy")
    if resume and os.path.exists(split_path):
        file_indices = np.load(split_path)
        print("Loaded existing train/val split")
    else:
        np.save(split_path, file_indices)
        ckpt_volume.commit()
        print("Saved new train/val split")

    val_files   = set(file_indices[:n_val].tolist())
    train_files = set(file_indices[n_val:].tolist())

    train_idx = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in train_files]
    val_idx   = [i for i, (fi, _, _) in enumerate(dataset._index) if fi in val_files]
    print(f"Train samples: {len(train_idx):,}  |  Val samples: {len(val_idx):,}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers>0,
        prefetch_factor=2 if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers>0,
        prefetch_factor=2 if n_workers > 0 else None,
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = PairDensityUNet(
        n_R=n_r,
        features=[32, 64, 128, 256],
        d_model=256,
        n_heads=8,
        max_atoms=max_atoms,
        max_modes=max_modes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = max_steps_per_epoch
    n_total_steps   = n_epochs * steps_per_epoch
    n_warmup_steps  = n_warmup_epochs * steps_per_epoch
    scheduler = warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps)

    # ── resume from checkpoint ───────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    ckpt_latest   = os.path.join(CKPT_MOUNT, "latest_checkpoint.pt")

    if resume and not debug and os.path.exists(ckpt_latest):
        ckpt = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  (best val = {best_val_loss:.6f})")

    if debug:
        # sanity checks
        print("Saving checkpoint test...")
        torch.save(
            {"test": True, "epoch": 0},
            os.path.join(CKPT_MOUNT, "debug_checkpoint_test.pt"),
        )
        ckpt_volume.commit()
        print("checkpoint dir contents:", os.listdir(CKPT_MOUNT))

        # ── Dataset sanity check ─────────────────────────────────────────
        sample = dataset[0]
        print("pd_limited: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
            sample["pd_limited"].min(), sample["pd_limited"].max(),
            sample["pd_limited"].mean(), sample["pd_limited"].std()
        ))
        print("pd_true:    min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
            sample["pd_true"].min(), sample["pd_true"].max(),
            sample["pd_true"].mean(), sample["pd_true"].std()
        ))
        print("\n--- Sample keys:", list(sample.keys()))
        print("pd_limited shape:", sample["pd_limited"].shape)
        print("pd_true shape:   ", sample["pd_true"].shape)
        print("R_grid shape:    ", sample["R_grid"].shape)
        print("V_vib shape:     ", sample["V_vib"].shape)
        print("omegas shape:    ", sample["omegas_ASE"].shape)
        # print("mol_features:    ", sample["mol_features"].shape)
        print("atom_mask sum:   ", sample["atom_mask"].sum().item(), "/ 50")
        print("mode_mask sum:   ", sample["mode_mask"].sum().item(), "/ 144")
        print("time_fs:         ", sample["time_fs"].item())

        # ── DataLoader / collation check ─────────────────────────────────
        batch = next(iter(train_loader))
        print("\n--- First train batch loaded OK")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:20s} {tuple(v.shape)}  dtype={v.dtype}")

        # ── Forward pass check ───────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pd_limited = batch["pd_limited"].to(device)
            V_vib      = batch["V_vib"].to(device)
            omegas     = batch["omegas_ASE"].to(device)
            atom_mask  = batch["atom_mask"].to(device)
            mode_mask  = batch["mode_mask"].to(device)
            out = model(pd_limited, V_vib, omegas, atom_mask, mode_mask)
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        print("\n--- Forward pass OK, output shape:", out.shape)
        print("output range: [{:.4f}, {:.4f}]".format(out.min().item(), out.max().item()))
        print("any NaN in output:", out.isnan().any().item())

        # ── Single training step check ───────────────────────────────────
        model.train()
        pd_limited = batch["pd_limited"].to(device)
        pd_true    = batch["pd_true"].to(device)
        V_vib      = batch["V_vib"].to(device)
        omegas     = batch["omegas_ASE"].to(device)
        atom_mask  = batch["atom_mask"].to(device)
        mode_mask  = batch["mode_mask"].to(device)

        optimizer.zero_grad()
        pred = model(pd_limited, V_vib, omegas, atom_mask, mode_mask)
        loss = nn.functional.mse_loss(pred, pd_true)
        loss.backward()
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        print(f"\n--- Backward pass OK, loss={loss.item():.6f}, grad_norm={total_norm:.4f}")
        print("any NaN in grads:", any(
            p.grad.isnan().any().item()
            for p in model.parameters() if p.grad is not None
        ))
    
        from tqdm import tqdm 
        # sample 10 frames from each of 100 random molecules
        rng_stats = np.random.default_rng(0)
        file_sample = rng_stats.choice(n_files, size=min(100, n_files), replace=False)
        vals = []
        for fi in tqdm(file_sample):
            d = dataset._load_file(fi)
            vals.append(d["pd_true"].reshape(-1, d["pd_true"].shape[-1])[:10])
        vals = np.concatenate(vals).ravel()

        print(f"pd_true global: mean={vals.mean():.4f}, std={vals.std():.4f}, max={vals.max():.2f}, min={vals.min():.2f}")
        vals_log = np.log1p(vals)
        print(f"log1p pd_true: mean={vals_log.mean():.4f}, std={vals_log.std():.4f}, max={vals_log.max():.2f}") 

        print("\nDebug checks passed. Exiting.")
        return float("inf")

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True, max_steps=max_steps_per_epoch)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, device, train=False, max_steps=max_steps_per_epoch)
        curr_lr    = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:03d}/{n_epochs} | "
            f"train {train_loss:.6f} | "
            f"val {val_loss:.6f} | "
            f"lr {curr_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                os.path.join(CKPT_MOUNT, "best_model.pt"),
            )
            # commit so the file is immediately visible outside the container
            ckpt_volume.commit()
            print(f"  ✓ best model saved (val = {val_loss:.6f})")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
            ckpt_latest,
        )
        ckpt_volume.commit()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return best_val_loss


@app.local_entrypoint()
def main():
    """Called by `modal run modal_train.py`"""
    result = train_remote.remote()
    if result < float("inf"):
        print(f"Finished. Best val loss: {result:.6f}")
    else:
        print("Debug run complete.")
