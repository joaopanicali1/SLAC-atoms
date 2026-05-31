import modal
from pathlib import Path

data_volume = modal.Volume.from_name("mol-scattering-data")
ckpt_volume = modal.Volume.from_name("mol-scattering-checkpoints")

app = modal.App("mol-scan", image=modal.Image.debian_slim().pip_install("numpy"))

@app.function(volumes={"/data": data_volume, "/checkpoints": ckpt_volume},
              timeout=60 * 30,)
def scan_and_clean():
    import numpy as np
    from pathlib import Path

    data_dir = Path("/data/trajectory_data")
    bad_files = []

    files = sorted(data_dir.glob("mol_*_pd.npz"))
    print(f"Scanning {len(files)} files...")

    for i, fpath in enumerate(files):
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        try:
            d = np.load(fpath)
            _ = d["pd_true"][:]
            _ = d["pd_limited"][:]
        except Exception as e:
            print(f"BAD: {fpath.name} — {e}")
            bad_files.append(fpath)

    print(f"\n{len(bad_files)} bad files found")
    for fpath in bad_files:
        print(f"Deleting {fpath.name}")
        fpath.unlink()

    if bad_files:
        data_volume.commit()
        # also delete the index cache so it gets rebuilt without the bad files
        index_cache = Path("/checkpoints/dataset_index.npy")
        if index_cache.exists():
            index_cache.unlink()
            ckpt_volume.commit()
            print("Deleted index cache — will rebuild on next run")

    print("Done.")

@app.local_entrypoint()
def main():
    scan_and_clean.remote()