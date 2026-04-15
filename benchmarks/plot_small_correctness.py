"""Plot results produced by benchmark_small_correctness.py.

Example:

    python benchmarks/benchmark_small_correctness.py \
        --output-jsonl benchmarks/small_correctness_results.jsonl

    python benchmarks/plot_small_correctness.py \
        --input benchmarks/small_correctness_results.jsonl \
        --output-dir benchmarks/figures_small_correctness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot small correctness benchmark results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/small_correctness_results.jsonl"),
        help="JSONL file produced by benchmark_small_correctness.py --output-jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/figures_small_correctness"),
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file_in:
        for line in file_in:
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                continue
            rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    return rows


def mean_by_group(rows: list[dict[str, object]], keys: tuple[str, ...], metric: str) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        buckets.setdefault(key, []).append(float(row[metric]))

    out: list[dict[str, object]] = []
    for key, values in buckets.items():
        entry = {keys[index]: key[index] for index in range(len(keys))}
        entry[metric] = sum(values) / len(values)
        out.append(entry)

    return sorted(out, key=lambda entry: tuple(entry[key_name] for key_name in keys))


def plot_ghz_fidelity(rows: list[dict[str, object]], output_path: Path) -> None:
    ghz_rows = [row for row in rows if row.get("case") == "ghz_dynamic"]
    summary = mean_by_group(ghz_rows, ("network_type", "Dmax", "no_qubits"), "mean_exact_state_fidelity")

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({entry["network_type"] for entry in summary}):
        for dmax in sorted({entry["Dmax"] for entry in summary if entry["network_type"] == network_type}):
            group = [entry for entry in summary if entry["network_type"] == network_type and entry["Dmax"] == dmax]
            group = sorted(group, key=lambda entry: int(entry["no_qubits"]))
            ax.plot(
                [int(entry["no_qubits"]) for entry in group],
                [float(entry["mean_exact_state_fidelity"]) for entry in group],
                marker="o",
                linewidth=2,
                label=f"{network_type}, Dmax={dmax}",
            )

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Mean exact-state fidelity")
    ax.set_title("Dynamic GHZ correctness")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ghz_branch_count(rows: list[dict[str, object]], output_path: Path) -> None:
    ghz_rows = [row for row in rows if row.get("case") == "ghz_dynamic"]
    summary = mean_by_group(ghz_rows, ("network_type", "Dmax", "no_qubits"), "branch_count")

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({entry["network_type"] for entry in summary}):
        for dmax in sorted({entry["Dmax"] for entry in summary if entry["network_type"] == network_type}):
            group = [entry for entry in summary if entry["network_type"] == network_type and entry["Dmax"] == dmax]
            group = sorted(group, key=lambda entry: int(entry["no_qubits"]))
            ax.plot(
                [int(entry["no_qubits"]) for entry in group],
                [float(entry["branch_count"]) for entry in group],
                marker="o",
                linewidth=2,
                label=f"{network_type}, Dmax={dmax}",
            )

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Average branch count")
    ax.set_title("Dynamic GHZ branch count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_teleportation_fidelity(rows: list[dict[str, object]], output_path: Path) -> None:
    tele_rows = [row for row in rows if row.get("case") == "teleportation"]
    multi_summary = mean_by_group(tele_rows, ("network_type", "Dmax", "compression_steps"), "multi_path_fidelity")
    single_summary = mean_by_group(tele_rows, ("network_type", "Dmax", "compression_steps"), "single_path_mean_fidelity")

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({entry["network_type"] for entry in multi_summary}):
        for dmax in sorted({entry["Dmax"] for entry in multi_summary if entry["network_type"] == network_type}):
            multi_group = [entry for entry in multi_summary if entry["network_type"] == network_type and entry["Dmax"] == dmax]
            single_group = [entry for entry in single_summary if entry["network_type"] == network_type and entry["Dmax"] == dmax]
            multi_group = sorted(multi_group, key=lambda entry: int(entry["compression_steps"]))
            single_group = sorted(single_group, key=lambda entry: int(entry["compression_steps"]))

            ax.plot(
                [int(entry["compression_steps"]) for entry in multi_group],
                [float(entry["multi_path_fidelity"]) for entry in multi_group],
                marker="o",
                linewidth=2,
                label=f"multi {network_type}, Dmax={dmax}",
            )
            ax.plot(
                [int(entry["compression_steps"]) for entry in single_group],
                [float(entry["single_path_mean_fidelity"]) for entry in single_group],
                marker="s",
                linewidth=2,
                linestyle="--",
                label=f"single {network_type}, Dmax={dmax}",
            )

    ax.set_xlabel("Compression steps")
    ax.set_ylabel("Fidelity")
    ax.set_title("Teleportation: single-path vs multi-path")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_teleportation_branch_probability(rows: list[dict[str, object]], output_path: Path) -> None:
    tele_rows = [row for row in rows if row.get("case") == "teleportation"]
    summary = mean_by_group(tele_rows, ("network_type", "Dmax", "compression_steps"), "single_path_mean_branch_probability")

    fig, ax = plt.subplots(figsize=(8, 5))
    for network_type in sorted({entry["network_type"] for entry in summary}):
        for dmax in sorted({entry["Dmax"] for entry in summary if entry["network_type"] == network_type}):
            group = [entry for entry in summary if entry["network_type"] == network_type and entry["Dmax"] == dmax]
            group = sorted(group, key=lambda entry: int(entry["compression_steps"]))
            ax.plot(
                [int(entry["compression_steps"]) for entry in group],
                [float(entry["single_path_mean_branch_probability"]) for entry in group],
                marker="o",
                linewidth=2,
                label=f"{network_type}, Dmax={dmax}",
            )

    ax.set_xlabel("Compression steps")
    ax.set_ylabel("Mean sampled branch probability")
    ax.set_title("Teleportation sampled branch probability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)

    plot_ghz_fidelity(rows, args.output_dir / "ghz_dynamic_fidelity_vs_qubits.png")
    plot_ghz_branch_count(rows, args.output_dir / "ghz_dynamic_branch_count_vs_qubits.png")
    plot_teleportation_fidelity(rows, args.output_dir / "teleportation_fidelity_single_vs_multi.png")
    plot_teleportation_branch_probability(rows, args.output_dir / "teleportation_branch_probability_vs_compression.png")

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()