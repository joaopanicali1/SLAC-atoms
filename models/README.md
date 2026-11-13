# Models

Each model family has the same internal layout:

```text
<model>/
  code/       model implementation and reusable best checkpoint(s)
  training/   training script(s), Slurm submit files, and training logs
  dataset/    PyTorch dataset used by that model
```

Current model folders:

```text
beamstop/  low-q beamstop recovery model
lpd/       learned primal-dual pair-density recovery model (v1 and v2)
unet/      pair-density UNet model
nsk/       NSK-l1 baseline as published (code/ only: no training, no dataset)
```

The final beamstop model is:

```text
models/beamstop/code/beamstop_final.pt
models/beamstop/code/beamstop_final.json
```

The committed UNet checkpoints are:

```text
models/unet/code/unet_best_model_epoch208.pt
models/unet/code/unet_v1.pt
```

The final LPD checkpoints are:

```text
models/lpd/code/lpd_final.pt      LPD v1
models/lpd/code/lpd_v2_final.pt   LPD v2 (epoch 8; recommended)
```
