"""Plot results produced by benchmark_dynamic_random_scaling.py.

The script reads the CSV output and generate figures for fidelity curves.

Example:

    python benchmarks/plot_dynamic_random_scaling.py \
        --input benchmarks/dynamic_random_scaling.csv \
        --output-dir benchmarks/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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

    fig, ax = plt.subplots(figsize=(10, 5))
    for dmax in sorted({row["Dmax"] for row in summary}):
        if dmax == 4:
            continue
        group = [row for row in summary if row["Dmax"] == dmax]
        group = sorted(group, key=lambda row: row["depth"])
        ax.plot(
            [row["depth"] for row in group],
            [row["single_path_mean_fidelity"] for row in group],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=f"$\\chi$={dmax} single",
        )
        ax.plot(
            [row["depth"] for row in group],
            [row["multi_path_fidelity"] for row in group],
            marker="s",
            linewidth=2,
            label=f"$\\chi$={dmax} multi",
        )

    ax.set_xlabel("Circuit depth", fontsize=18)
    ax.set_ylabel("Fidelity", fontsize=18)
    #ax.set_title(f"{network_label} fidelity vs circuit depth")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)




def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output_dir)
    rows = load_results(args.input)

    plot_single_vs_multi_by_depth(rows, "mps", args.output_dir / "mps_fidelity_vs_depth.png")
    plot_single_vs_multi_by_depth(rows, "tree", args.output_dir / "ttn_fidelity_vs_depth.png")

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()