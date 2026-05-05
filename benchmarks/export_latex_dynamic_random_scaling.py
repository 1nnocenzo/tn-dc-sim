"""Export LaTeX tables from dynamic random scaling benchmark CSV results.

The script filters rows by Dmax (default: 8), then builds one table for MPS
and one table for TTN. Rows are aggregated by depth, averaging circuit size,
runtime, and peak-memory metrics across repeated seeds and circuit instances.

Example:
    python benchmarks/export_latex_dynamic_random_scaling.py \
        --input benchmarks/dynamic_random_scaling.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX tables for dynamic scaling benchmark results.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file produced by benchmark_dynamic_random_scaling.py.",
    )
    parser.add_argument(
        "--dmax",
        type=int,
        default=8,
        help="Bond dimension value to filter on (default: 8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required value for column '{key}'")
    return float(value)


def _int_value(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required value for column '{key}'")
    return int(value)


def load_rows(path: Path, dmax: int) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    parsed: list[dict[str, object]] = []
    for row in rows:
        if _int_value(row, "Dmax") != dmax:
            continue

        network_type = str(row["network_type"]).strip().lower()
        if network_type not in {"mps", "tree"}:
            continue

        parsed.append(
            {
                "network_type": network_type,
                "depth": _int_value(row, "depth"),
                "size": _int_value(row, "circuit_size"),
                "runtime_single": _float_value(row, "single_path_mean_runtime_s")
                * _int_value(row, "single_path_repeat_count"),
                "peak_single": _float_value(row, "single_path_mean_peak_bytes_estimate"),
                "runtime_multi": _float_value(row, "multi_path_runtime_s"),
                "peak_multi": _float_value(row, "multi_path_peak_bytes_estimate"),
            }
        )

    if not parsed:
        raise ValueError(f"No rows found for Dmax={dmax} in {path}")

    return parsed


def summarize(rows: list[dict[str, object]], network_type: str) -> list[dict[str, float | int]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row["network_type"] != network_type:
            continue

        depth = int(row["depth"])
        grouped[depth]["runtime_single"].append(float(row["runtime_single"]))
        grouped[depth]["peak_single"].append(float(row["peak_single"]))
        grouped[depth]["runtime_multi"].append(float(row["runtime_multi"]))
        grouped[depth]["peak_multi"].append(float(row["peak_multi"]))

    summary_rows: list[dict[str, float | int]] = []
    for depth, acc in grouped.items():
        runtime_single_values = acc["runtime_single"]
        peak_single_values = acc["peak_single"]
        runtime_multi_values = acc["runtime_multi"]
        peak_multi_values = acc["peak_multi"]

        if not runtime_single_values:
            continue

        def _mean_std(values: list[float]) -> tuple[float, float]:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return mean, sqrt(variance)

        runtime_single_mean, runtime_single_std = _mean_std(runtime_single_values)
        peak_single_mean, peak_single_std = _mean_std(peak_single_values)
        runtime_multi_mean, runtime_multi_std = _mean_std(runtime_multi_values)
        peak_multi_mean, peak_multi_std = _mean_std(peak_multi_values)

        summary_rows.append(
            {
                "depth": depth,
                "runtime_single_mean": runtime_single_mean,
                "runtime_single_std": runtime_single_std,
                "peak_single_mean": peak_single_mean,
                "peak_single_std": peak_single_std,
                "runtime_multi_mean": runtime_multi_mean,
                "runtime_multi_std": runtime_multi_std,
                "peak_multi_mean": peak_multi_mean,
                "peak_multi_std": peak_multi_std,
            }
        )

    return sorted(summary_rows, key=lambda item: int(item["depth"]))


def _bytes_to_megabytes(value_bytes: float) -> float:
    return value_bytes / 1_000_000.0


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{_fmt(mean)} $\\pm$ {_fmt(std)}"


def _latex_table(summary_rows: list[dict[str, float | int]], dmax: int) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{c|cccc|cccc}")
    lines.append(r"\hline")
    lines.append(r" & \multicolumn{4}{c|}{MPS} & \multicolumn{4}{c}{TTN} \\")
    lines.append(r"\hline")
    lines.append(r"depth & runtime sp [s] & memory sp [MB] & runtime mp [s] & memory mp [MB] & runtime sp [s] & memory sp [MB] & runtime mp [s] & memory mp [MB] \\")
    lines.append(r"\hline")

    for row in summary_rows:
        depth = int(row["depth"])
        mps_runtime_single = _fmt_mean_std(float(row["mps_runtime_single_mean"]), float(row["mps_runtime_single_std"]))
        mps_memory_single = _fmt_mean_std(
            _bytes_to_megabytes(float(row["mps_peak_single_mean"])),
            _bytes_to_megabytes(float(row["mps_peak_single_std"])),
        )
        mps_runtime_multi = _fmt_mean_std(float(row["mps_runtime_multi_mean"]), float(row["mps_runtime_multi_std"]))
        mps_memory_multi = _fmt_mean_std(
            _bytes_to_megabytes(float(row["mps_peak_multi_mean"])),
            _bytes_to_megabytes(float(row["mps_peak_multi_std"])),
        )
        ttn_runtime_single = _fmt_mean_std(float(row["ttn_runtime_single_mean"]), float(row["ttn_runtime_single_std"]))
        ttn_memory_single = _fmt_mean_std(
            _bytes_to_megabytes(float(row["ttn_peak_single_mean"])),
            _bytes_to_megabytes(float(row["ttn_peak_single_std"])),
        )
        ttn_runtime_multi = _fmt_mean_std(float(row["ttn_runtime_multi_mean"]), float(row["ttn_runtime_multi_std"]))
        ttn_memory_multi = _fmt_mean_std(
            _bytes_to_megabytes(float(row["ttn_peak_multi_mean"])),
            _bytes_to_megabytes(float(row["ttn_peak_multi_std"])),
        )

        lines.append(
            " & ".join(
                [
                    str(depth),
                    mps_runtime_single,
                    mps_memory_single,
                    mps_runtime_multi,
                    mps_memory_multi,
                    ttn_runtime_single,
                    ttn_memory_single,
                    ttn_runtime_multi,
                    ttn_memory_multi,
                ]
            )
            + r" \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{Computational resources for MPS and TTN simulations for both single-path (sp) and multi-path (mp) methods on random dynamic circuits with $D_{{\max}}={dmax}$.}}")
    lines.append(rf"\label{{tab:dynamic-scaling-combined-dmax-{dmax}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_output(rows: list[dict[str, object]], dmax: int) -> str:
    mps_rows = summarize(rows, "mps")
    ttn_rows = summarize(rows, "tree")

    if not mps_rows:
        raise ValueError(f"No MPS rows found for Dmax={dmax}")
    if not ttn_rows:
        raise ValueError(f"No TTN rows found for Dmax={dmax}")

    mps_by_depth = {int(row["depth"]): row for row in mps_rows}
    ttn_by_depth = {int(row["depth"]): row for row in ttn_rows}
    if set(mps_by_depth) != set(ttn_by_depth):
        missing_mps = sorted(set(ttn_by_depth) - set(mps_by_depth))
        missing_ttn = sorted(set(mps_by_depth) - set(ttn_by_depth))
        raise ValueError(f"Mismatched depths between MPS and TTN rows: missing MPS={missing_mps}, missing TTN={missing_ttn}")

    combined_rows: list[dict[str, float | int]] = []
    for depth in sorted(mps_by_depth):
        mps_row = mps_by_depth[depth]
        ttn_row = ttn_by_depth[depth]
        combined_rows.append(
            {
                "depth": depth,
                "mps_runtime_single_mean": mps_row["runtime_single_mean"],
                "mps_runtime_single_std": mps_row["runtime_single_std"],
                "mps_peak_single_mean": mps_row["peak_single_mean"],
                "mps_peak_single_std": mps_row["peak_single_std"],
                "mps_runtime_multi_mean": mps_row["runtime_multi_mean"],
                "mps_runtime_multi_std": mps_row["runtime_multi_std"],
                "mps_peak_multi_mean": mps_row["peak_multi_mean"],
                "mps_peak_multi_std": mps_row["peak_multi_std"],
                "ttn_runtime_single_mean": ttn_row["runtime_single_mean"],
                "ttn_runtime_single_std": ttn_row["runtime_single_std"],
                "ttn_peak_single_mean": ttn_row["peak_single_mean"],
                "ttn_peak_single_std": ttn_row["peak_single_std"],
                "ttn_runtime_multi_mean": ttn_row["runtime_multi_mean"],
                "ttn_runtime_multi_std": ttn_row["runtime_multi_std"],
                "ttn_peak_multi_mean": ttn_row["peak_multi_mean"],
                "ttn_peak_multi_std": ttn_row["peak_multi_std"],
            }
        )

    return _latex_table(combined_rows, dmax)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.dmax)
    output_text = build_output(rows, args.dmax)

    if args.output is None:
        print(output_text)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")
    print(f"Wrote LaTeX tables to {args.output}")


if __name__ == "__main__":
    main()

