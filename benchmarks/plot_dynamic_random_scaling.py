"""Plot results produced by benchmark_dynamic_random_scaling.py.

The script reads the CSV output and generates a small set of summary figures
for runtime, fidelity, branch count, and branch probability behavior.

Example:

    python benchmarks/plot_dynamic_random_scaling.py \
        --input benchmarks/dynamic_random_scaling.csv \
        --output-dir benchmarks/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dynamic random scaling benchmark results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/dynamic_random_scaling.csv"),
        help="CSV file produced by benchmark_dynamic_random_scaling.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/figures"),
        help="Directory where plots will be saved.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as file_in:
        rows = list(csv.DictReader(file_in))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        parsed_row = dict(row)
        parsed_row["network_structure_label"] = _structure_label(str(parsed_row["network_structure"]))
        parsed_row["circuit_seed"] = int(parsed_row["circuit_seed"])
        parsed_row["no_qubits"] = int(parsed_row["no_qubits"])
        parsed_row["depth"] = int(parsed_row["depth"])
        parsed_row["Dmax"] = int(parsed_row["Dmax"])
        parsed_row["compression_steps"] = int(parsed_row["compression_steps"])
        parsed_row["no_sweeps"] = int(parsed_row["no_sweeps"])
        parsed_row["max_ops_per_branch"] = int(parsed_row["max_ops_per_branch"])
        parsed_row["single_path_repeat_count"] = int(parsed_row["single_path_repeat_count"])
        parsed_row["single_path_mean_fidelity"] = float(parsed_row["single_path_mean_fidelity"])
        parsed_row["single_path_std_fidelity"] = float(parsed_row["single_path_std_fidelity"])
        parsed_row["single_path_mean_branch_probability"] = float(parsed_row["single_path_mean_branch_probability"])
        parsed_row["single_path_std_branch_probability"] = float(parsed_row["single_path_std_branch_probability"])
        parsed_row["single_path_mean_runtime_s"] = float(parsed_row["single_path_mean_runtime_s"])
        parsed_row["single_path_mean_final_tensor_elements"] = float(parsed_row["single_path_mean_final_tensor_elements"])
        parsed_row["single_path_std_final_tensor_elements"] = float(parsed_row["single_path_std_final_tensor_elements"])
        parsed_row["single_path_mean_peak_tensor_elements"] = float(parsed_row["single_path_mean_peak_tensor_elements"])
        parsed_row["single_path_std_peak_tensor_elements"] = float(parsed_row["single_path_std_peak_tensor_elements"])
        parsed_row["single_path_mean_final_bytes_estimate"] = float(parsed_row["single_path_mean_final_bytes_estimate"])
        parsed_row["single_path_std_final_bytes_estimate"] = float(parsed_row["single_path_std_final_bytes_estimate"])
        parsed_row["single_path_mean_peak_bytes_estimate"] = float(parsed_row["single_path_mean_peak_bytes_estimate"])
        parsed_row["single_path_std_peak_bytes_estimate"] = float(parsed_row["single_path_std_peak_bytes_estimate"])
        parsed_row["multi_path_fidelity"] = float(parsed_row["multi_path_fidelity"])
        parsed_row["multi_path_pruning_error"] = float(parsed_row["multi_path_pruning_error"])
        parsed_row["multi_path_runtime_s"] = float(parsed_row["multi_path_runtime_s"])
        parsed_row["multi_path_final_tensor_elements"] = int(parsed_row["multi_path_final_tensor_elements"])
        parsed_row["multi_path_peak_tensor_elements"] = int(parsed_row["multi_path_peak_tensor_elements"])
        parsed_row["multi_path_final_bytes_estimate"] = int(parsed_row["multi_path_final_bytes_estimate"])
        parsed_row["multi_path_peak_bytes_estimate"] = int(parsed_row["multi_path_peak_bytes_estimate"])
        parsed_row["branch_count"] = int(parsed_row["branch_count"])
        parsed_row["circuit_size"] = int(parsed_row["circuit_size"])
        parsed_rows.append(parsed_row)

    return parsed_rows


def _structure_label(raw: str) -> str:
    try:
        structure = json.loads(raw)
    except Exception:
        return str(raw)
    return "-".join(str(value) for value in structure)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mean_by_key(rows: list[dict[str, object]], keys: tuple[str, ...], metric: str) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        buckets.setdefault(key, []).append(float(row[metric]))

    summary_rows: list[dict[str, object]] = []
    for key, values in buckets.items():
        summary_row = {keys[index]: key[index] for index in range(len(keys))}
        summary_row[metric] = sum(values) / len(values)
        summary_rows.append(summary_row)

    return sorted(summary_rows, key=lambda row: tuple(row[key_name] for key_name in keys))


def plot_metric_by_qubits(rows: list[dict[str, object]], metric: str, ylabel: str, output_path: Path) -> None:
    summary = _mean_by_key(rows, ("network_type", "Dmax", "no_qubits"), metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        for dmax in sorted({row["Dmax"] for row in summary if row["network_type"] == network_type}):
            group = [row for row in summary if row["network_type"] == network_type and row["Dmax"] == dmax]
            group = sorted(group, key=lambda row: row["no_qubits"])
            label = f"{network_type}, Dmax={dmax}"
            ax.plot([row["no_qubits"] for row in group], [row[metric] for row in group], marker="o", linewidth=2, label=label)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs qubits")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_compression(rows: list[dict[str, object]], metric: str, ylabel: str, output_path: Path) -> None:
    summary = _mean_by_key(rows, ("network_type", "Dmax", "compression_steps"), metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        for dmax in sorted({row["Dmax"] for row in summary if row["network_type"] == network_type}):
            group = [row for row in summary if row["network_type"] == network_type and row["Dmax"] == dmax]
            group = sorted(group, key=lambda row: row["compression_steps"])
            label = f"{network_type}, Dmax={dmax}"
            ax.plot(
                [row["compression_steps"] for row in group],
                [row[metric] for row in group],
                marker="o",
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Compression steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs compression steps")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_dmax(rows: list[dict[str, object]], metric: str, ylabel: str, output_path: Path, *, use_log_scale: bool = False) -> None:
    summary = _mean_by_key(rows, ("network_type", "Dmax"), metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        group = [row for row in summary if row["network_type"] == network_type]
        group = sorted(group, key=lambda row: row["Dmax"])
        label = "TTN" if network_type == "tree" else network_type.upper()
        ax.plot(
            [row["Dmax"] for row in group],
            [row[metric] for row in group],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Dmax")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Dmax")
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_vs_multi_by_dmax(
    rows: list[dict[str, object]],
    metric_single: str,
    metric_multi: str,
    ylabel: str,
    output_path: Path,
    *,
    use_log_scale: bool = False,
    single_agg: str = "mean",
    multi_agg: str = "mean",
) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["network_type"], row["Dmax"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (network_type, dmax), group in grouped.items():
        if single_agg == "sum":
            single_value = sum(float(row[metric_single]) for row in group)
        else:
            single_value = sum(float(row[metric_single]) for row in group) / len(group)

        if multi_agg == "sum":
            multi_value = sum(float(row[metric_multi]) for row in group)
        else:
            multi_value = sum(float(row[metric_multi]) for row in group) / len(group)

        summary.append(
            {
                "network_type": network_type,
                "Dmax": dmax,
                metric_single: single_value,
                metric_multi: multi_value,
            }
        )

    summary = sorted(summary, key=lambda row: (row["network_type"], row["Dmax"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        group = [row for row in summary if row["network_type"] == network_type]
        group = sorted(group, key=lambda row: row["Dmax"])
        label_prefix = "TTN" if network_type == "tree" else network_type.upper()
        ax.plot(
            [row["Dmax"] for row in group],
            [row[metric_single] for row in group],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=f"{label_prefix} single",
        )
        ax.plot(
            [row["Dmax"] for row in group],
            [row[metric_multi] for row in group],
            marker="s",
            linewidth=2,
            label=f"{label_prefix} multibranch",
        )

    ax.set_xlabel("Dmax")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Dmax")
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_vs_multi_by_depth(rows: list[dict[str, object]], network_type: str, output_path: Path) -> None:
    network_rows = [row for row in rows if row["network_type"] == network_type]
    if not network_rows:
        raise ValueError(f"No rows found for network_type={network_type!r}")

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in network_rows:
        key = (row["Dmax"], row["depth"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dmax, depth), group in grouped.items():
        summary.append(
            {
                "Dmax": dmax,
                "depth": depth,
                "single_path_mean_fidelity": sum(float(row["single_path_mean_fidelity"]) for row in group) / len(group),
                "multi_path_fidelity": sum(float(row["multi_path_fidelity"]) for row in group) / len(group),
            }
        )

    summary = sorted(summary, key=lambda row: (row["Dmax"], row["depth"]))
    network_label = "MPS" if network_type == "mps" else "TTN"

    fig, ax = plt.subplots(figsize=(8, 5))
    for dmax in sorted({row["Dmax"] for row in summary}):
        group = [row for row in summary if row["Dmax"] == dmax]
        group = sorted(group, key=lambda row: row["depth"])
        ax.plot(
            [row["depth"] for row in group],
            [row["single_path_mean_fidelity"] for row in group],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=f"Dmax={dmax} single",
        )
        ax.plot(
            [row["depth"] for row in group],
            [row["multi_path_fidelity"] for row in group],
            marker="s",
            linewidth=2,
            label=f"Dmax={dmax} multi",
        )

    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"{network_label} fidelity vs circuit depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_branch_count(rows: list[dict[str, object]], output_path: Path) -> None:
    summary = _mean_by_key(rows, ("network_type", "Dmax", "no_qubits"), "branch_count")

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        for dmax in sorted({row["Dmax"] for row in summary if row["network_type"] == network_type}):
            group = [row for row in summary if row["network_type"] == network_type and row["Dmax"] == dmax]
            group = sorted(group, key=lambda row: row["no_qubits"])
            label = f"{network_type}, Dmax={dmax}"
            ax.plot([row["no_qubits"] for row in group], [row["branch_count"] for row in group], marker="o", linewidth=2, label=label)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Average branch count")
    ax.set_title("Branch count vs qubits")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_vs_multi_runtime(rows: list[dict[str, object]], output_path: Path) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["network_type"], row["Dmax"], row["no_qubits"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (network_type, dmax, no_qubits), group in grouped.items():
        summary.append(
            {
                "network_type": network_type,
                "Dmax": dmax,
                "no_qubits": no_qubits,
                "single_path_mean_runtime_s": sum(float(row["single_path_mean_runtime_s"]) for row in group) / len(group),
                "multi_path_runtime_s": sum(float(row["multi_path_runtime_s"]) for row in group) / len(group),
            }
        )

    summary = sorted(summary, key=lambda row: (row["network_type"], row["Dmax"], row["no_qubits"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({row["network_type"] for row in summary}):
        for dmax in sorted({row["Dmax"] for row in summary if row["network_type"] == network_type}):
            group = [row for row in summary if row["network_type"] == network_type and row["Dmax"] == dmax]
            group = sorted(group, key=lambda row: row["no_qubits"])
            label_single = f"single {network_type}, Dmax={dmax}"
            label_multi = f"multi {network_type}, Dmax={dmax}"
            ax.plot([row["no_qubits"] for row in group], [row["single_path_mean_runtime_s"] for row in group], marker="o", linewidth=2, linestyle="--", label=label_single)
            ax.plot([row["no_qubits"] for row in group], [row["multi_path_runtime_s"] for row in group], marker="s", linewidth=2, label=label_multi)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Single-path vs multi-path runtime")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output_dir)
    rows = load_results(args.input)

    plot_metric_by_qubits(rows, "multi_path_fidelity", "Multi-path fidelity", args.output_dir / "multi_path_fidelity_vs_qubits.png")
    plot_metric_by_qubits(rows, "single_path_mean_fidelity", "Single-path mean fidelity", args.output_dir / "single_path_fidelity_vs_qubits.png")
    plot_single_vs_multi_by_dmax(
        rows,
        "single_path_mean_fidelity",
        "multi_path_fidelity",
        "Fidelity",
        args.output_dir / "fidelity_vs_dmax.png",
    )
    plot_single_vs_multi_by_depth(rows, "mps", args.output_dir / "mps_fidelity_vs_depth.png")
    plot_single_vs_multi_by_depth(rows, "tree", args.output_dir / "ttn_fidelity_vs_depth.png")
    plot_single_vs_multi_by_dmax(
        rows,
        "single_path_mean_peak_bytes_estimate",
        "multi_path_peak_bytes_estimate",
        "Peak memory footprint [bytes]",
        args.output_dir / "memory_footprint_vs_dmax.png",
        use_log_scale=True,
    )
    plot_single_vs_multi_by_dmax(
        rows,
        "single_path_mean_runtime_s",
        "multi_path_runtime_s",
        "Runtime [s]",
        args.output_dir / "runtime_vs_dmax.png",
        use_log_scale=True,
        single_agg="sum",
    )
    plot_metric_by_compression(rows, "multi_path_fidelity", "Multi-path fidelity", args.output_dir / "multi_path_fidelity_vs_compression.png")
    plot_metric_by_compression(rows, "single_path_mean_fidelity", "Single-path mean fidelity", args.output_dir / "single_path_fidelity_vs_compression.png")
    plot_branch_count(rows, args.output_dir / "branch_count_vs_qubits.png")
    plot_single_vs_multi_runtime(rows, args.output_dir / "runtime_single_vs_multi.png")

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()