"""
Regenerate trajectory data from mol_data_new with a max_frames cap.

Usage:
    python regenerate_trajectories.py \
        --data-dir data/mol_data_new \
        --out-dir  data/trajectory_data\
        --max-frames 500 \
        --workers 8

Recommended: --max-frames 500  (median period = 370 frames, max = 6670)
Skips molecules whose output .npz already exists — safe to resume after interruption.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from functools import partial

from processing.analyse_mol_v2 import process_molecule 


def _worker(npz_path: Path, out_dir: Path, max_frames: int) -> tuple[str, str or None]:
    """Top-level function (required for multiprocessing pickling)."""
    result = process_molecule(
        npz_path=npz_path,
        out_dir=out_dir,
        max_frames=max_frames,
    )
    return npz_path.name, result


def main():
    parser = argparse.ArgumentParser(description="Regenerate trajectory data with max_frames cap")
    parser.add_argument("--data-dir",   default="data/mol_data_new",
                        help="Input directory with mol_*_physics.npz files")
    parser.add_argument("--out-dir",    default="data/trajectory_data_v2",
                        help="Output directory for mol_*_pd.npz files")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Cap trajectory length in frames (default 500)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="Number of parallel worker processes (default 8)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print what would be processed without running")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        sys.exit(f"ERROR: data directory not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_npz = sorted(data_dir.glob("*.npz"))
    if not all_npz:
        sys.exit(f"ERROR: no .npz files found in {data_dir}")

    # filter out already-completed molecules
    def _is_done(p: Path) -> bool:
        import re
        m = re.search(r'mol_(\d+)_physics', p.stem)
        if m is None:
            return False
        mol_idx = int(m.group(1))
        return (out_dir / f"mol_{mol_idx:06d}_pd.npz").exists()

    pending = [p for p in all_npz if not _is_done(p)]
    done    = len(all_npz) - len(pending)

    print(f"Input dir: {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"max_frames: {args.max_frames}")
    print(f"Workers: {args.workers}")
    print(f"Total files: {len(all_npz)}  |  Already done: {done}  |  To process: {len(pending)}")

    if args.dry_run:
        for p in pending:
            print(f"  would process: {p.name}")
        return

    if not pending:
        print("Nothing to do — all outputs already exist.")
        return

    worker_fn = partial(_worker, out_dir=out_dir, max_frames=args.max_frames)

    n_ok = 0
    n_skip = 0
    n_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker_fn, p): p for p in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            fname, result = fut.result()
            if result is None:
                n_ok += 1
                status = "OK"
            elif result.startswith("SKIP"):
                n_skip += 1
                status = result
            else:
                n_err += 1
                status = f"ERROR: {result}"
            print(f"[{i:>5}/{len(pending)}] {fname}  →  {status}")

    print(f"\nDone.  OK={n_ok}  Skipped={n_skip}  Errors={n_err}")


if __name__ == "__main__":
    main()