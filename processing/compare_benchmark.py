"""
Read all benchmark CSVs and print a speedup table across node counts.
Run any time — shows "pending" for results not yet available.
  python processing/compare_benchmark.py
"""
import os
import csv
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(PROJECT_ROOT, "processing")

NODE_CONFIGS = {
    "large":  [32, 64],
    "xlarge": [32, 64, 128],
}


def load_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {int(r["mol_idx"]): r for r in csv.DictReader(f)}


def fmt(val, suffix=""):
    try:
        return f"{float(val):6.1f}{suffix}"
    except (ValueError, TypeError):
        return f"{'—':>7s}"


for tier, nprocs_list in NODE_CONFIGS.items():
    data = {}  # nprocs -> {mol_idx -> row}
    for np in nprocs_list:
        csv_path = os.path.join(PROC_DIR, f"benchmark_results_{tier}_{np}cores.csv")
        data[np] = load_csv(csv_path)

    # Collect all mol indices seen across any config
    all_mol_idx = sorted({i for rows in data.values() for i in rows})
    if not all_mol_idx:
        print(f"\n{tier.upper()} — no results yet\n")
        continue

    header_nodes = "  ".join(f"{np//32:>1d}N/{np:>3d}c(min)" for np in nprocs_list)
    speedup_hdrs = "  ".join(f"x{np//32:d}v{nprocs_list[0]//32:d}" for np in nprocs_list[1:])

    print(f"\n{'='*72}")
    print(f"{tier.upper()} tier")
    print(f"{'Atoms':>6s}  {'Formula':22s}  {header_nodes}  {speedup_hdrs}")
    print("-" * 72)

    base_np = nprocs_list[0]
    for mi in all_mol_idx:
        base_row = data[base_np].get(mi, {})
        formula  = base_row.get("formula", "?")
        atoms    = base_row.get("actual_atoms", "?")

        parts = [f"{atoms:>6s}  {formula:22s}"]
        base_wall = None
        for np in nprocs_list:
            row = data[np].get(mi, {})
            wall = row.get("wall_min", "pending")
            parts.append(fmt(wall, " "))
            if np == base_np:
                try:
                    base_wall = float(wall)
                except (ValueError, TypeError):
                    pass

        for np in nprocs_list[1:]:
            row = data[np].get(mi, {})
            try:
                wall = float(row.get("wall_min", "nan"))
                sp   = base_wall / wall if base_wall and wall > 0 else float("nan")
                parts.append(f"{sp:>6.2f}x")
            except (ValueError, TypeError):
                parts.append(f"{'pending':>7s}")

        print("  ".join(parts))

print()
print("Speedup > 1.5x  → extra nodes worthwhile for that molecule size")
print("Speedup < 1.2x  → communication overhead dominates; stick to 1 node")