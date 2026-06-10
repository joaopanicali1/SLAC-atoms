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
lpd/       learned primal-dual pair-density recovery model
unet/      pair-density UNet model
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

The final LPD checkpoint is:

```text
models/lpd/code/lpd_final.pt
```
