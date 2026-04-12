#!/usr/bin/env python3
"""
extract_all_metrics.py
Extract ncu metrics from every .ncu-rep file produced by run_all.sh.

For each run_XX.ncu-rep found under an ncu_results_* directory, two CSVs
are written alongside the report:
  run_XX_metrics.csv          — one row per kernel launch
  run_XX_metrics_grouped.csv  — metrics averaged per kernel name

Usage
-----
  python extract_all_metrics.py [RESULTS_DIR]

  RESULTS_DIR  path to an ncu_results_* directory.
               Defaults to the most-recently modified ncu_results_* directory
               found next to this script.
"""

from __future__ import annotations

import sys
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: reuse extract_ncu_metrics from the shared python/ directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE.parent.parent / "python"          # …/AccelProf/python
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

try:
    from extract_ncu_metrics import (
        extract_actions,
        group_by_kernel,
        write_csv,
        write_grouped_csv,
        COLUMNS,
        GROUPED_COLUMNS,
    )
except ImportError as exc:
    sys.exit(f"ERROR: Cannot import extract_ncu_metrics from {_PYTHON_DIR}.\n{exc}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv_file(rows: list[dict], cols: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_report(rep: Path) -> tuple[int, int]:
    """Extract metrics from one .ncu-rep and write two CSVs. Returns (launch_count, kernel_count)."""
    rows = list(extract_actions(rep))
    if not rows:
        print(f"    WARNING: no kernel actions found in {rep.name}")
        return 0, 0

    grouped = group_by_kernel(rows)

    out_launch  = rep.with_name(rep.stem + "_metrics.csv")
    out_grouped = rep.with_name(rep.stem + "_metrics_grouped.csv")

    write_csv_file(rows,    COLUMNS,         out_launch)
    write_csv_file(grouped, GROUPED_COLUMNS, out_grouped)

    return len(rows), len(grouped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_results_dir(script_dir: Path) -> Path | None:
    candidates = sorted(script_dir.glob("ncu_results_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = find_latest_results_dir(script_dir)
        if results_dir is None:
            sys.exit(
                f"ERROR: No ncu_results_* directory found under {script_dir}.\n"
                "Run run_all.sh first, or pass the results directory as an argument."
            )

    if not results_dir.is_dir():
        sys.exit(f"ERROR: Not a directory: {results_dir}")

    print(f"Results directory: {results_dir}\n")

    total_pass = total_fail = total_skip = 0

    for app_dir in sorted(results_dir.iterdir()):
        if not app_dir.is_dir():
            continue

        reports = sorted(app_dir.glob("run_*.ncu-rep"))
        if not reports:
            print(f"[ {app_dir.name} ]  no .ncu-rep files — skipping")
            total_skip += 1
            continue

        print(f"[ {app_dir.name} ]")
        for rep in reports:
            try:
                n_launches, n_kernels = process_report(rep)
                print(f"  {rep.name:<35}  {n_launches:>4} launches  {n_kernels:>3} unique kernels"
                      f"  ->  {rep.stem}_metrics.csv  +  {rep.stem}_metrics_grouped.csv")
                total_pass += 1
            except Exception as exc:
                print(f"  {rep.name:<35}  FAILED: {exc}")
                total_fail += 1
        print()

    print("=" * 60)
    print(f"  Done.  OK={total_pass}  FAILED={total_fail}  SKIPPED={total_skip}")
    print(f"  CSVs written under: {results_dir}")
    print("=" * 60)

    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
