"""
train_lpd_v2.py — Train LearnedPrimalDualV2 to recover the normalised pair density p(R) from
noise-free scattering, with beamstop augmentation. Runs on one device or on several GPUs
(DistributedDataParallel, launched with torchrun).

Task
----
Input : S0(q) of one frame, processed per sample as dS0 = (S0 − Σf²)/fe · W_qmin with a random
        beamstop q_min (see models/learned_primal_dual_v2.process_scattering).
Target: p(R) = pd_true(R) / Σ_{a<b} Z_a Z_b  (unit-mass histogram, 0.025 Å bins).

Loss = mean W1(p̂, p) + λ_dc · mean ‖T f_raw − g‖² / ‖g‖²
  W1      1-D Wasserstein distance [Å]: a peak displaced by δ costs δ × its weight, so the
          loss rewards placing pair weight at the right distance instead of averaging
          (MSE on spike-like targets drives predictions to a blurred mean).
  data    misfit of the pre-activation primal iterate through the Debye operator, over the
          measured q only. Kept soft (λ_dc small): the point-atom fe approximation leaves a
          few-% misfit even for the true PD.

Split: molecule-level, read from data/splits/split_lpd.npy (first 10 % = validation) — the same molecules
analysis/pd_recovery/eval_pd_recovery.py evaluates on. The split file is never rewritten.

Validation (each epoch, fixed random subset of validation frames) in two conditions,
q_min = 0 (clean) and q_min = 0.8 Å⁻¹ (beamstop), with the harness's metrics:
  W1 up to 1 Å past the longest true pair distance, junk (|weight| beyond that), and R² after
  blurring prediction and truth by σ = 0, 0.025, 0.05 Å. Baselines (all 66 validation molecules,
  medians; analysis/pd_recovery/results/lpd_v2/pd_eval_{clean,bs_q08}.json):
    clean     NSK-ℓ1 (paper, L-curve ε) W1 0.099 Å, junk 0.000, R²@0.025/0.05 = 0.21 / 0.69;
              Tikhonov W1 0.074 Å, junk 0.027, R²@0.025/0.05 = 0.52 / 0.73.
    beamstop  NSK-ℓ1 (paper) W1 0.270 Å, R²@0.05 = 0.61;  Tikhonov W1 2.16 Å (fails).
  The released checkpoint (epoch 8) reaches W1 0.037 / 0.043 Å and R²@0.05 = 0.90 / 0.90.
The best checkpoint is chosen on clean validation median W1.
The same metrics are also computed on a fixed random subset of TRAINING frames ("train-subset"
in the log, "train_eval" in the checkpoint). The training loss is not comparable to the validation
metrics: it mixes beamstop conditions and integrates W1 over the whole R grid. The train-subset
numbers are directly comparable, so their gap to validation measures how well the model generalises
to unseen molecules.

Multi-GPU (torchrun)
--------------------
One process per GPU. Every process preloads the full training set (≈ 10 GB of host RAM each) and
trains on a disjoint 1/N share of every epoch (DistributedSampler, reshuffled per epoch). Gradients
are averaged over processes, so with the same global --batch-size a run on N GPUs follows the same
optimisation as a single-GPU run (only the sample order and the random beamstop draws differ).
Process 0 alone validates, logs and writes checkpoints; checkpoints have the same format whatever
the number of GPUs, and a run can be resumed on a different number of GPUs.
A step whose gradient norm is non-finite is skipped. The norm is computed after averaging, so all
processes see the same value and skip together.

Flags
-----
  --data-dir         directory of mol_*_pd.npz files
  --split            molecule permutation (first 10 % validation)
  --checkpoint-dir   where best_model.pt / latest_checkpoint.pt are written
  --epochs           passes over the training frames (≈ 3.4 M frames per epoch)
  --batch-size       GLOBAL batch size, split evenly over the processes (must be divisible by
                     their number): 256 on 8 GPUs means 32 frames per GPU per step
  --lr, --weight-decay, --warmup-epochs, --grad-clip                  optimisation
  --n-iter, --n-channels, --n-primal, --n-dual                         model size
  --kappa            internal scale f = κ·p
  --tikhonov-rel     λ of the Tikhonov warm start relative to σ_max(F)²
  --lambda-dc        weight of the data-consistency term
  --qmin-lo, --qmin-hi  beamstop q_min [Å⁻¹] drawn uniformly in this range for augmented samples
  --p-full           fraction of samples with no beamstop (q_min = 0). --qmin-hi 0 disables
                     beamstop augmentation entirely.
  --val-frames       validation frames evaluated per epoch (fixed subset); the same number of
                     training frames is evaluated for the train-subset metrics
  --num-workers      DataLoader workers per process
  --max-files, --max-steps   limits for smoke tests (default: no limit)
  --resume           continue from checkpoint-dir/latest_checkpoint.pt (fresh start if it does not
                     exist). The learning rate is a function of training progress in epochs (linear
                     warmup over --warmup-epochs, then cosine decay to 0 at --epochs), so --epochs, the
                     number of GPUs and the batch size may all differ from the original run. With
                     --warmup-epochs 1, a 1-epoch run followed by a resume with --epochs N is identical
                     to an N-epoch run. --lr and --weight-decay are taken from the command line, not
                     from the checkpoint (the defaults are those of the original run).

Usage
-----
    python models/lpd/training/train_lpd_v2.py                                    # one device, defaults
    python models/lpd/training/train_lpd_v2.py --max-files 6 --max-steps 20 --val-frames 64   # smoke test
    torchrun --standalone --nproc_per_node=8 models/lpd/training/train_lpd_v2.py --resume     # 8 GPUs
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.lpd.dataset.dataset_lpd_v2 import LPDv2Dataset, split_files  # noqa: E402
from models.lpd.code.learned_primal_dual_v2 import (  # noqa: E402
    LearnedPrimalDualV2, process_scattering, r2_blurred, w1_loss)

VAL_SIGMAS = (0.0, 0.025, 0.05)
VAL_CONDITIONS = {"clean": 0.0, "bs_q08": 0.8}


def setup_distributed():
    """
    Returns (world_size, rank, device). torchrun sets WORLD_SIZE / RANK / LOCAL_RANK; without
    them this is a plain single-process run. GPUs use NCCL (one process per GPU); a multi-process
    run without GPUs uses gloo (smoke tests only).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    timeout = datetime.timedelta(minutes=30)        # the others wait while process 0 validates/saves
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", timeout=timeout)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timeout)
    return world, rank, device


def barrier(world, device):
    if world > 1:
        dist.barrier(device_ids=[device.index]) if device.type == "cuda" else dist.barrier()


def reduce_sum(t, world):
    """Sum of tensor t over processes (a copy; t itself is unchanged)."""
    if world > 1:
        t = t.clone()
        dist.all_reduce(t)
    return t


def lr_factor(progress, args):
    """LR multiplier after `progress` epochs (fractional): linear warmup, then cosine decay to 0."""
    w = args.warmup_epochs
    if progress < w:
        return progress / w
    return 0.5 * (1 + math.cos(math.pi * (progress - w) / max(args.epochs - w, 1e-9)))


def sample_qmin(B, args, device):
    lo, hi = args.qmin_lo, args.qmin_hi
    if hi <= 0:
        return torch.zeros(B, device=device)
    qmin = lo + (hi - lo) * torch.rand(B, device=device)
    full = torch.rand(B, device=device) < args.p_full
    return torch.where(full, torch.zeros_like(qmin), qmin)


def w1_rcut_and_junk(p_hat, p, dR, margin=1.0):
    """Harness-compatible W1 (up to longest true distance + margin) and junk beyond it."""
    nR = p.shape[-1]
    last = nR - 1 - torch.argmax((p.flip(-1) != 0).to(torch.int8), dim=-1)
    cut = (last + int(math.ceil(margin / dR)) + 1).clamp(max=nR)
    inside = torch.arange(nR, device=p.device)[None, :] < cut[:, None]
    w1 = (torch.cumsum(p_hat - p, -1).abs() * inside).sum(-1) * dR
    junk = (p_hat.abs() * ~inside).sum(-1)
    return w1, junk


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


@torch.no_grad()
def validate(model, loader, q, dR, device):
    model.eval()
    out = {c: {"W1": [], "junk": [], **{f"R2@{s:g}": [] for s in VAL_SIGMAS}} for c in VAL_CONDITIONS}
    for batch in loader:
        b = to_device(batch, device)
        B = b["S0"].shape[0]
        for cond, qmin in VAL_CONDITIONS.items():
            dS0, W = process_scattering(b["S0"], b["self_scatter"], b["fe"], q,
                                        torch.full((B,), qmin, device=device))
            p_hat, _ = model(dS0, W)
            w1, junk = w1_rcut_and_junk(p_hat, b["p"], dR)
            out[cond]["W1"].append(w1.cpu())
            out[cond]["junk"].append(junk.cpu())
            for s in VAL_SIGMAS:
                out[cond][f"R2@{s:g}"].append(r2_blurred(p_hat, b["p"], s, dR).cpu())
    model.train()
    return {c: {k: float(torch.cat(v).median()) for k, v in m.items()} for c, m in out.items()}


def main(args):
    world, rank, device = setup_distributed()
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    if args.batch_size % world:
        raise ValueError(f"--batch-size {args.batch_size} is not divisible by {world} processes")
    local_bs = args.batch_size // world
    # Per-process seeds → independent beamstop draws. Initial weights may differ, but DDP copies
    # process 0's weights to all processes when the model is wrapped.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True       # input shapes are fixed: autotune convolutions once
    ckpt_dir = ROOT / args.checkpoint_dir
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log(f"device: {device}   processes: {world}   frames per process per step: {local_bs}   "
        f"args: {json.dumps(vars(args))}")

    val_files, train_files = split_files(ROOT / args.data_dir, ROOT / args.split)
    if args.max_files:
        train_files = train_files[:args.max_files]
        val_files = val_files[:max(1, args.max_files // 3)]
    log(f"train molecules: {len(train_files)}   val molecules: {len(val_files)}")
    log("loading training set ...")
    train_ds = LPDv2Dataset(train_files, verbose=is_main)

    common = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.num_workers > 0)
    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                  seed=args.seed, drop_last=True) if world > 1 else None)
    train_loader = DataLoader(train_ds, batch_size=local_bs, shuffle=sampler is None, sampler=sampler,
                              drop_last=True, **common)
    val_loader = train_eval_loader = None
    if is_main:                                     # only process 0 validates
        log("loading validation set ...")
        val_ds = LPDv2Dataset(val_files)
        val_idx = np.random.default_rng(0).choice(len(val_ds), size=min(args.val_frames, len(val_ds)),
                                                  replace=False)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
                                **common)
        # fixed training frames, scored with the validation metrics (generalisation gap)
        tr_idx = np.random.default_rng(1).choice(len(train_ds), size=min(args.val_frames, len(train_ds)),
                                                 replace=False)
        train_eval_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=args.batch_size,
                                       shuffle=False, **common)

    q = torch.tensor(train_ds.q, device=device)
    dR = float(train_ds.R_grid[1] - train_ds.R_grid[0])
    # net: the plain module (state_dict, scale_data, data_residual, validation);
    # model: what the training step calls (the DDP wrapper when world > 1).
    net = LearnedPrimalDualV2(train_ds.q, train_ds.R_grid, n_primal=args.n_primal,
                              n_dual=args.n_dual, n_iter=args.n_iter, n_channels=args.n_channels,
                              kappa=args.kappa, tikhonov_rel=args.tikhonov_rel).to(device)
    log(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) if not args.max_steps else min(len(train_loader), args.max_steps)

    start_epoch, best = 1, float("inf")
    latest = ckpt_dir / "latest_checkpoint.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device)
        net.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])     # Adam moments
        for pg in opt.param_groups:                         # hyper-parameters: from the command line
            pg["weight_decay"] = args.weight_decay
        start_epoch, best = ck["epoch"] + 1, ck["best_val_w1"]
        log(f"resumed from epoch {ck['epoch']} (best clean val W1 {best:.4f}; "
            f"lr now {args.lr * lr_factor(ck['epoch'], args):.2e})")

    # Buffers (Debye operator, coordinates) are constants built identically on every process,
    # so they are not re-broadcast at each forward.
    model = (DDP(net, device_ids=[device.index] if device.type == "cuda" else None,
                 broadcast_buffers=False) if world > 1 else net)

    if start_epoch == 1 and is_main:
        v = validate(net, val_loader, q, dR, device)
        te = validate(net, train_eval_loader, q, dR, device)
        log(f"epoch 0 (warm start only) | val {json.dumps(v)}\n  train-subset {json.dumps(te)}")

    def save(path, epoch, val, train_eval):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"epoch": epoch, "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "config": net.config,
                    # tensors, not numpy arrays: torch ≥ 2.6 loads with weights_only=True
                    "q": torch.tensor(train_ds.q), "R_grid": torch.tensor(train_ds.R_grid),
                    "args": vars(args), "val": val, "train_eval": train_eval,
                    "best_val_w1": best, "world_size": world}, tmp)
        os.replace(tmp, path)       # atomic: a job killed mid-save leaves the previous file intact

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        acc = torch.zeros(4, dtype=torch.float64, device=device)   # Σ loss, Σ W1, Σ dc, steps
        one = torch.ones((), device=device)
        skipped = 0
        for step, batch in enumerate(train_loader, 1):
            if args.max_steps and step > args.max_steps:
                break
            lr = args.lr * lr_factor(epoch - 1 + (step - 1) / steps_per_epoch, args)
            for pg in opt.param_groups:
                pg["lr"] = lr
            b = to_device(batch, device)
            B = b["S0"].shape[0]
            dS0, W = process_scattering(b["S0"], b["self_scatter"], b["fe"], q, sample_qmin(B, args, device))
            p_hat, f_raw = model(dS0, W)
            g, Wt = net.scale_data(dS0, W)
            w1 = w1_loss(p_hat, b["p"], dR).mean()
            dc = net.data_residual(f_raw, g, Wt).mean()
            loss = w1 + args.lambda_dc * dc

            opt.zero_grad(set_to_none=True)
            loss.backward()                 # DDP averages the gradients over processes here
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            if not torch.isfinite(gn):      # same value on every process → all skip together
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.step()
            acc += torch.stack([loss.detach(), w1.detach(), dc.detach(), one]).double()
            if step % args.log_every == 0:
                a = reduce_sum(acc, world)
                n = max(1.0, float(a[3]))
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} | loss {a[0]/n:.4f} "
                    f"W1 {a[1]/n:.4f} dc {a[2]/n:.4f} | lr {lr:.2e} | {time.time() - t0:.0f} s")

        a = reduce_sum(acc, world)
        n = max(1.0, float(a[3]))
        if is_main:
            v = validate(net, val_loader, q, dR, device)
            te = validate(net, train_eval_loader, q, dR, device)
            msg = (f"epoch {epoch:03d}/{args.epochs} | train loss {a[0]/n:.4f} W1 {a[1]/n:.4f} "
                   f"dc {a[2]/n:.4f} | skipped {skipped} | {time.time() - t0:.0f} s\n"
                   f"  val {json.dumps(v)}\n"
                   f"  train-subset {json.dumps(te)}")
            if v["clean"]["W1"] < best:
                best = v["clean"]["W1"]
                save(ckpt_dir / "best_model.pt", epoch, v, te)
                msg += f"\n  new best (clean val median W1 = {best:.4f} Å)"
            save(latest, epoch, v, te)
            log(msg)
        barrier(world, device)

    log(f"done. best clean val median W1 = {best:.4f} Å")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/trajectory_data_cap200")
    p.add_argument("--split", default="data/splits/split_lpd.npy")
    p.add_argument("--checkpoint-dir", default="models/lpd/code/checkpoints/lpd_v2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--n-channels", type=int, default=32)
    p.add_argument("--n-primal", type=int, default=5)
    p.add_argument("--n-dual", type=int, default=5)
    p.add_argument("--kappa", type=float, default=100.0)
    p.add_argument("--tikhonov-rel", type=float, default=1e-6)
    p.add_argument("--lambda-dc", type=float, default=0.1)
    p.add_argument("--qmin-lo", type=float, default=0.3)
    p.add_argument("--qmin-hi", type=float, default=1.0)
    p.add_argument("--p-full", type=float, default=0.3)
    p.add_argument("--val-frames", type=int, default=20000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    main(p.parse_args())
