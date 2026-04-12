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

import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# ncu_report bootstrap
# ---------------------------------------------------------------------------
_NCU_PYTHON = Path("/usr/local/cuda-13.0/nsight-compute-2025.3.0/extras/python")
if _NCU_PYTHON.exists() and str(_NCU_PYTHON) not in sys.path:
    sys.path.insert(0, str(_NCU_PYTHON))

try:
    import ncu_report
except ImportError as exc:
    sys.exit(
        f"ERROR: Cannot import ncu_report. "
        f"Ensure Nsight Compute is installed and {_NCU_PYTHON} exists.\n{exc}"
    )

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------
M_GRID_X     = "launch__grid_dim_x"
M_GRID_Y     = "launch__grid_dim_y"
M_GRID_Z     = "launch__grid_dim_z"
M_BLOCK_X    = "launch__block_dim_x"
M_BLOCK_Y    = "launch__block_dim_y"
M_BLOCK_Z    = "launch__block_dim_z"
M_CYCLES     = "smsp__cycles_elapsed.avg"
M_GLOBAL_LD  = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
M_GLOBAL_ST  = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
M_SHARED_LD  = "smsp__sass_inst_executed_op_shared_ld.sum"
M_SHARED_ST  = "smsp__sass_inst_executed_op_shared_st.sum"
M_LOCAL_LD   = "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum"
M_LOCAL_ST   = "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"
M_L1_L2_LD   = "lts__t_sectors_srcunit_tex_op_read.sum"
M_L1_L2_ST   = "lts__t_sectors_srcunit_tex_op_write.sum"
M_L2_DRAM_LD = "dram__sectors_read.sum"
M_L2_DRAM_ST = "dram__sectors_write.sum"

COLUMNS = [
    "launch_id", "kernel_name", "grid_size", "block_size",
    "cycles",
    "global_ld_instr", "global_st_instr",
    "shared_ld_instr", "shared_st_instr",
    "local_ld_instr",  "local_st_instr",
    "l1_to_l2_ld_sectors", "l1_to_l2_st_sectors",
    "l2_to_dram_ld_sectors", "l2_to_dram_st_sectors",
]

GROUPED_COLUMNS = [
    "kernel_name", "invocations",
    "cycles_avg",
    "global_ld_instr_avg", "global_st_instr_avg",
    "shared_ld_instr_avg", "shared_st_instr_avg",
    "local_ld_instr_avg",  "local_st_instr_avg",
    "l1_to_l2_ld_sectors_avg", "l1_to_l2_st_sectors_avg",
    "l2_to_dram_ld_sectors_avg", "l2_to_dram_st_sectors_avg",
]

NUMERIC_COLS = [c for c in COLUMNS if c not in ("launch_id", "kernel_name", "grid_size", "block_size")]

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _metric_val(action, name):
    m = action.metric_by_name(name)
    if m is None:
        return None
    k = m.kind()
    if k in (ncu_report.IMetric.ValueKind_UINT64, ncu_report.IMetric.ValueKind_UINT32):
        return m.as_uint64()
    if k in (ncu_report.IMetric.ValueKind_FLOAT, ncu_report.IMetric.ValueKind_DOUBLE):
        return m.as_double()
    return None


def extract_actions(report_path: Path) -> list[dict]:
    ctx = ncu_report.load_report(str(report_path))
    rows = []
    launch_id = 0
    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            gx = _metric_val(action, M_GRID_X)
            gy = _metric_val(action, M_GRID_Y)
            gz = _metric_val(action, M_GRID_Z)
            bx = _metric_val(action, M_BLOCK_X)
            by = _metric_val(action, M_BLOCK_Y)
            bz = _metric_val(action, M_BLOCK_Z)
            rows.append({
                "launch_id":             launch_id,
                "kernel_name":           action.name(ncu_report.IAction.NameBase_DEMANGLED),
                "grid_size":             f"({gx},{gy},{gz})",
                "block_size":            f"({bx},{by},{bz})",
                "cycles":                _metric_val(action, M_CYCLES),
                "global_ld_instr":       _metric_val(action, M_GLOBAL_LD),
                "global_st_instr":       _metric_val(action, M_GLOBAL_ST),
                "shared_ld_instr":       _metric_val(action, M_SHARED_LD),
                "shared_st_instr":       _metric_val(action, M_SHARED_ST),
                "local_ld_instr":        _metric_val(action, M_LOCAL_LD),
                "local_st_instr":        _metric_val(action, M_LOCAL_ST),
                "l1_to_l2_ld_sectors":   _metric_val(action, M_L1_L2_LD),
                "l1_to_l2_st_sectors":   _metric_val(action, M_L1_L2_ST),
                "l2_to_dram_ld_sectors": _metric_val(action, M_L2_DRAM_LD),
                "l2_to_dram_st_sectors": _metric_val(action, M_L2_DRAM_ST),
            })
            launch_id += 1
    return rows


def group_by_kernel(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["kernel_name"]].append(row)
    # Preserve first-appearance order
    order = {r["kernel_name"]: i for i, r in enumerate(rows)}
    grouped = []
    for name, group in sorted(buckets.items(), key=lambda kv: order[kv[0]]):
        n = len(group)
        agg: dict = {"kernel_name": name, "invocations": n}
        for col in NUMERIC_COLS:
            vals = [r[col] for r in group if r.get(col) is not None]
            agg[f"{col}_avg"] = (sum(vals) / len(vals)) if vals else None
        grouped.append(agg)
    return grouped


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_csv_file(rows: list[dict], cols: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-report processing
# ---------------------------------------------------------------------------

def process_report(rep: Path) -> tuple[int, int]:
    rows = extract_actions(rep)
    if not rows:
        print(f"    WARNING: no kernel actions found in {rep.name}")
        return 0, 0

    grouped = group_by_kernel(rows)

    write_csv_file(rows,    COLUMNS,         rep.with_name(rep.stem + "_metrics.csv"))
    write_csv_file(grouped, GROUPED_COLUMNS, rep.with_name(rep.stem + "_metrics_grouped.csv"))

    return len(rows), len(grouped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_results_dir(script_dir: Path) -> Path | None:
    candidates = sorted(
        script_dir.glob("ncu_results_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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
                print(
                    f"  {rep.name:<35}  {n_launches:>4} launches  "
                    f"{n_kernels:>3} unique kernels  ->  "
                    f"{rep.stem}_metrics.csv  +  {rep.stem}_metrics_grouped.csv"
                )
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
