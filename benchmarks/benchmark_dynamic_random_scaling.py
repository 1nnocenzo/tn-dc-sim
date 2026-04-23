"""Scaling benchmark for dynamic random circuits.

This script sweeps circuit size, depth, seed, bond dimension, tensor-network
type, compression steps, and tree structure. It compares the single-path and
multi-path dynamic simulators on the same family of randomly generated dynamic
circuits.

Run from the repository root, for example:

    python benchmarks/benchmark_dynamic_random_scaling.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from qiskit import qasm3, transpile
from qiskit.circuit.random import random_circuit as random_circuit_qiskit



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation import (  # noqa: E402
    DMRG_dynamic_all_branches_from_qasm3,
    DMRG_dynamic_single_path_from_qasm3,
    DMRG_from_qasm3,
)
from src.utils.TN_gen import D_mps, D_tree  # noqa: E402
from src.utils.my_random_circuit import random_circuit  # noqa: E402
from src.utils.quantum_gates import supported_ir_gates

@dataclass
class ScalingRow:
    circuit_seed: int
    no_qubits: int
    depth: int
    network_type: str
    network_structure: str
    Dmax: int
    compression_steps: int
    no_sweeps: int
    max_ops_per_branch: int
    single_path_repeat_count: int
    single_path_mean_fidelity: float
    single_path_std_fidelity: float
    single_path_mean_branch_probability: float
    single_path_std_branch_probability: float
    single_path_mean_runtime_s: float
    single_path_mean_final_tensor_elements: float
    single_path_std_final_tensor_elements: float
    single_path_mean_peak_tensor_elements: float
    single_path_std_peak_tensor_elements: float
    single_path_mean_final_bytes_estimate: float
    single_path_std_final_bytes_estimate: float
    single_path_mean_peak_bytes_estimate: float
    single_path_std_peak_bytes_estimate: float
    multi_path_fidelity: float
    multi_path_pruning_error: float
    multi_path_runtime_s: float
    multi_path_final_tensor_elements: int
    multi_path_peak_tensor_elements: int
    multi_path_final_bytes_estimate: int
    multi_path_peak_bytes_estimate: int
    branch_count: int
    circuit_size: int


def candidate_ttn_structures(no_qubits: int) -> list[list[int]]:
    #return [[1, 16, no_qubits]] 
    return [[1, 3, 9, no_qubits]]
    #return [[1, no_qubits // 2, no_qubits]]


def _qc_is_dynamic(qc) -> bool:
    for instruction in qc.data:
        op = instruction.operation
        op_name = getattr(op, "name", "")
        if op_name == "reset":
            return True
        if op_name in {"if_else", "if_test"}:
            return True
        if getattr(op, "condition", None) is not None:
            return True
    return False


def _qasm_is_dynamic(qasm_text: str) -> bool:
    return ("if (" in qasm_text) or ("reset " in qasm_text)


def _qc_has_conditional(qc) -> bool:
    for instruction in qc.data:
        op = instruction.operation
        op_name = getattr(op, "name", "")
        if op_name in {"if_else", "if_test"}:
            return True
        if getattr(op, "condition", None) is not None:
            return True
    return False


def _qasm_has_conditional(qasm_text: str) -> bool:
    return len(re.findall(r"\bif\s*\(", qasm_text)) > 0


def build_dynamic_random_qasm(no_qubits: int, depth: int, seed: int, max_ops_per_branch: int) -> str:
    
    max_attempts = 10000
    for attempt in range(max_attempts):
        trial_seed = seed + attempt

        qc = random_circuit(
            num_qubits=no_qubits,
            depth=depth,
            max_operands=2,
            conditional=True,
            reset=False,
            seed=trial_seed,
            num_operand_distribution={1: 0.65, 2: 0.35},
            max_ops_per_branch=max_ops_per_branch,
        )

        # Force circuits to include at least one conditional operation.
        if not _qc_has_conditional(qc):
            continue

        qc = transpile(qc, optimization_level=0, basis_gates=supported_ir_gates())
        size = qc.size()
        print("Size transpiled circuit: ",size)

        qasm_text = qasm3.dumps(qc)
        if _qasm_has_conditional(qasm_text):
            return qasm_text, trial_seed, size
    
        
    

    raise RuntimeError(
        "Unable to generate a random circuit containing at least one conditional operation "
        f"after {max_attempts} attempts."
    )


def _count_circuit_characteristics(qasm_text: str, max_branches: int | None = None) -> tuple[int, int, int]:
    """Count dynamic operations, conditional operations, and branch estimate from QASM.

    Returned values are:
        - dynamic operations: measurements + resets
        - conditional operations: number of operations inside ``if (...) { ... }`` blocks
        - branches: estimated from measurement and reset events that introduce branching
    """
    num_measure_ops = 0
    num_reset_ops = 0
    num_conditional_ops = 0
    num_branch_events = 0
    conditional_branch_events = 0

    # Parse if-blocks line by line to handle complex conditions, e.g.:
    # if (c[5] & c[6] ^ ~c[12]) {
    #   z q[15];
    # }
    in_if_block = False
    if_block_depth = 0
    current_if_ops = 0

    for raw_line in qasm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if in_if_block:
            open_braces = line.count("{")
            close_braces = line.count("}")

            # Count executable statements inside the current if-block.
            if line not in {"{", "}"} and line.endswith(";") and not line.startswith("if"):
                current_if_ops += 1

            if_block_depth += open_braces - close_braces
            if if_block_depth <= 0:
                conditional_branch_events += current_if_ops
                in_if_block = False
                current_if_ops = 0
            continue

        if re.match(r"^if\s*\(.*\)\s*\{\s*$", line):
            in_if_block = True
            if_block_depth = 1
            current_if_ops = 0
            continue

        if "measure" in line:
            num_measure_ops += 1
            num_branch_events += 1
            continue

        if line.startswith("reset "):
            num_reset_ops += 1
            num_branch_events += 1
            continue

    num_conditional_ops = conditional_branch_events
    num_branch_events += conditional_branch_events
    num_branches = 2 ** num_branch_events if num_branch_events > 0 else 1
    if max_branches is not None:
        num_branches = min(num_branches, max_branches)

    return num_measure_ops, num_reset_ops, num_conditional_ops, num_branches


def characterize_random_circuits(
    no_qubits: int,
    depth: int,
    seeds: list[int],
    max_ops_per_branch: int,
    max_branches: int | None = None,
) -> dict[str, float]:
    """Generate random circuits and characterize them over multiple seeds.
    
    Args:
        no_qubits: Number of qubits
        depth: Circuit depth
        seeds: List of seeds to use for circuit generation
        max_ops_per_branch: Maximum operations per conditional branch
    
    Returns:
        Dictionary with keys:
            - dynamic_ops_mean, dynamic_ops_std
            - conditional_ops_mean, conditional_ops_std
            - branches_mean, branches_std
    """
    measurement_ops_list = []
    reset_ops_list = []
    conditional_ops_list = []
    branches_list = []
    
    seeds = [x * depth for x in seeds]  # Space out seeds by depth to get more variability in circuit structure
    for i, seed in enumerate(seeds):
        qasm_text, trial_seed, _ = build_dynamic_random_qasm(
            no_qubits=no_qubits,
            depth=depth,
            seed=seed,
            max_ops_per_branch=max_ops_per_branch,
        )
        if i < len(seeds) - 1:
            seeds[i + 1] = trial_seed + 1
        measurement_ops, reset_ops, conditional_ops, branches = _count_circuit_characteristics(
            qasm_text=qasm_text,
            max_branches=max_branches,
        )

        measurement_ops_list.append(measurement_ops)
        reset_ops_list.append(reset_ops)
        conditional_ops_list.append(conditional_ops)
        branches_list.append(branches)
    
    return {
        "measurement_ops_mean": float(np.mean(measurement_ops_list)) if measurement_ops_list else 0.0,
        "measurement_ops_std": float(np.std(measurement_ops_list)) if measurement_ops_list else 0.0,
        "reset_ops_mean": float(np.mean(reset_ops_list)) if reset_ops_list else 0.0,
        "reset_ops_std": float(np.std(reset_ops_list)) if reset_ops_list else 0.0,
        "conditional_ops_mean": float(np.mean(conditional_ops_list)) if conditional_ops_list else 0.0,
        "conditional_ops_std": float(np.std(conditional_ops_list)) if conditional_ops_list else 0.0,
        "branches_mean": float(np.mean(branches_list)) if branches_list else 0.0,
        "branches_std": float(np.std(branches_list)) if branches_list else 0.0,
    }


def run_single_path(
    qasm_text: str,
    D,
    network_structure: list[int],
    network_type: str,
    compression_steps: int,
    no_sweeps: int,
    repeat_count: int,
    base_seed: int,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    fidelities = []
    branch_probabilities = []
    runtimes = []
    final_tensor_elements = []
    peak_tensor_elements = []
    final_bytes_estimate = []
    peak_bytes_estimate = []

    for repeat in range(repeat_count):
        run_seed = base_seed * 10_000 + repeat
        start = time.perf_counter()
        fidelity, branch_probability, memory_stats = DMRG_dynamic_single_path_from_qasm3(
            compression_steps=compression_steps,
            no_sweeps=no_sweeps,
            D=D,
            network_structure=network_structure,
            qasm_text=qasm_text,
            network_type=network_type,
            seed=run_seed,
            return_branch_probability=True,
            return_memory_stats=True,
        )
        runtimes.append(time.perf_counter() - start)
        fidelities.append(float(fidelity))
        branch_probabilities.append(float(branch_probability))
        final_tensor_elements.append(float(memory_stats["final_tensor_elements"]))
        peak_tensor_elements.append(float(memory_stats["peak_tensor_elements"]))
        final_bytes_estimate.append(float(memory_stats["final_bytes_estimate"]))
        peak_bytes_estimate.append(float(memory_stats["peak_bytes_estimate"]))

    return (
        fidelities,
        branch_probabilities,
        runtimes,
        final_tensor_elements,
        peak_tensor_elements,
        final_bytes_estimate,
        peak_bytes_estimate,
    )


def run_multi_path(
    qasm_text: str,
    D,
    network_structure: list[int],
    network_type: str,
    compression_steps: int,
    no_sweeps: int,
    base_seed: int,
    max_branches: int | None,
    probability_cutoff: float,
) -> tuple[float, float, float, int, dict[str, int]]:
    start = time.perf_counter()
    fidelity, pruning_error, branches, memory_stats = DMRG_dynamic_all_branches_from_qasm3(
        compression_steps=compression_steps,
        no_sweeps=no_sweeps,
        D=D,
        network_structure=network_structure,
        qasm_text=qasm_text,
        network_type=network_type,
        seed=base_seed,
        max_branches=max_branches,
        probability_cutoff=probability_cutoff,
        return_branches=True,
        return_pruning_error=True,
        return_memory_stats=True,
    )
    runtime = time.perf_counter() - start
    return float(fidelity), float(pruning_error), float(runtime), len(branches), memory_stats


def count_total_configurations(args: argparse.Namespace) -> int:
    total = 0
    for no_qubits in args.qubits:
        ttn_structures = candidate_ttn_structures(no_qubits)
        for _depth in args.depths:
            for _seed in args.seeds:
                for network_type in args.network_types:
                    structures = ttn_structures if network_type == "tree" else [[1, no_qubits]]
                    total += (
                        len(structures)
                        * len(args.dmax)
                        * len(args.compression_steps)
                        * len(args.no_sweeps)
                    )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic random circuit scaling benchmarks.")
    parser.add_argument("--qubits", type=int, nargs="+", default=[27], help="Qubit counts to test.")
    parser.add_argument("--depths", type=int, nargs="+", default=[2,3,4,5,6,7,8], help="Circuit depths to test.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3], help="Circuit seeds to test.")
    parser.add_argument(
        "--network-types",
        type=str,
        nargs="+",
        default=["tree"],
        choices=["tree", "mps"],
        help="Network types to benchmark.",
    )
    parser.add_argument("--dmax", type=int, nargs="+", default=[4,8,16], help="Bond dimensions to test.")
    parser.add_argument(
        "--compression-steps",
        type=int,
        nargs="+",
        default=[20],
        help="Compression steps to test.",
    )
    parser.add_argument("--no-sweeps", type=int, nargs="+", default=[2], help="Sweep counts to test.")
    parser.add_argument(
        "--single-path-repeats",
        type=int,
        default=10, 
        help="Number of single-path samples per benchmark configuration.",
    )
    parser.add_argument(
        "--max-ops-per-branch",
        type=int,
        default=1,
        help="Maximum operations per randomly generated conditional branch.",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=8,
        help="Optional cap on the number of branches retained by the multi-path simulator.",
    )
    parser.add_argument(
        "--probability-cutoff",
        type=float,
        default=0.0,
        help="Optional probability pruning cutoff for the multi-path simulator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dynamic_random_scaling.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--characterize",
        action="store_true",
        help="Characterize random circuits (dynamic ops, conditional ops, branches) without running simulator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Handle characterization mode
    if args.characterize:
        print("\n=== Circuit Characterization Mode ===\n")
        for no_qubits in args.qubits:
            for depth in args.depths:
                print(f"\nCharacterizing circuits: qubits={no_qubits}, depth={depth}, seeds={args.seeds}")
                result = characterize_random_circuits(
                    no_qubits=no_qubits,
                    depth=depth,
                    seeds=args.seeds,
                    max_ops_per_branch=args.max_ops_per_branch,
                    max_branches=args.max_branches,
                )
                #print(f"  Dynamic operations:    mean={result['dynamic_ops_mean']:.2f}, std={result['dynamic_ops_std']:.2f}")
                print(f"  Measurement ops:       mean={result['measurement_ops_mean']:.2f}, std={result['measurement_ops_std']:.2f}")
                print(f"  Reset ops:             mean={result['reset_ops_mean']:.2f}, std={result['reset_ops_std']:.2f}")
                print(f"  Conditional operations: mean={result['conditional_ops_mean']:.2f}, std={result['conditional_ops_std']:.2f}")
                print(f"  Branches:              mean={result['branches_mean']:.2f}, std={result['branches_std']:.2f}")
        return
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_configurations = count_total_configurations(args)

    rows: list[ScalingRow] = []
    config_index = 0
    for no_qubits in args.qubits:
        ttn_structures = candidate_ttn_structures(no_qubits) # scegliere a priori
        for depth in args.depths:
            for i, seed in enumerate(args.seeds):
                seeds = [x * depth for x in args.seeds]
                qasm_text, trial_seed, size = build_dynamic_random_qasm(
                    no_qubits=no_qubits,
                    depth=depth,
                    seed=seed,
                    max_ops_per_branch=args.max_ops_per_branch,
                )
                if i < len(seeds) - 1:
                    seeds[i + 1] = trial_seed + 1
                circuit_size = size

                for network_type in args.network_types:
                    if network_type == "tree":
                        structures = ttn_structures
                    else:
                        structures = [[1, no_qubits]]

                    for structure in structures:
                        structure_string = json.dumps(structure)
                        for dmax in args.dmax:
                            if network_type == "tree":
                                D = D_tree(structure, dmax)
                            else:
                                D = D_mps(no_qubits, dmax)

                            for compression_steps in args.compression_steps:
                                for no_sweeps in args.no_sweeps:
                                    config_index += 1
                                    print(
                                        f"\nConfiguration {config_index}/{total_configurations}: "
                                        f"qubits={no_qubits}, depth={depth}, size={circuit_size}, seed={seed}, "
                                        f"network_type={network_type}, structure={structure_string}, "
                                        f"Dmax={dmax}, compression_steps={compression_steps}, no_sweeps={no_sweeps}"
                                    )
                                    '''
                                    DMRG_from_qasm3(qasm_text=qasm_text,
                                        D=D,
                                        network_structure=structure,
                                        network_type=network_type,
                                        compression_steps=compression_steps,
                                        no_sweeps=no_sweeps)
                                    exit()
                                    '''
                                    (
                                        single_fidelities,
                                        single_branch_probs,
                                        single_runtimes,
                                        single_final_tensor_elements,
                                        single_peak_tensor_elements,
                                        single_final_bytes_estimate,
                                        single_peak_bytes_estimate,
                                    ) = run_single_path(
                                        qasm_text=qasm_text,
                                        D=D,
                                        network_structure=structure,
                                        network_type=network_type,
                                        compression_steps=compression_steps,
                                        no_sweeps=no_sweeps,
                                        repeat_count=args.single_path_repeats,
                                        base_seed=seed,
                                    )
                                    (
                                        multi_fidelity,
                                        pruning_error,
                                        multi_runtime,
                                        branch_count,
                                        multi_memory_stats,
                                    ) = run_multi_path(
                                        qasm_text=qasm_text,
                                        D=D,
                                        network_structure=structure,
                                        network_type=network_type,
                                        compression_steps=compression_steps,
                                        no_sweeps=no_sweeps,
                                        base_seed=seed,
                                        max_branches=args.max_branches,
                                        probability_cutoff=args.probability_cutoff,
                                    )

                                    
                                    row = ScalingRow(
                                        circuit_seed=seed,
                                        no_qubits=no_qubits,
                                        depth=depth,
                                        network_type=network_type,
                                        network_structure=structure_string,
                                        Dmax=dmax,
                                        compression_steps=compression_steps,
                                        no_sweeps=no_sweeps,
                                        max_ops_per_branch=args.max_ops_per_branch,
                                        single_path_repeat_count=args.single_path_repeats,
                                        single_path_mean_fidelity=float(np.mean(single_fidelities)),
                                        single_path_std_fidelity=float(np.std(single_fidelities)),
                                        single_path_mean_branch_probability=float(np.mean(single_branch_probs)),
                                        single_path_std_branch_probability=float(np.std(single_branch_probs)),
                                        single_path_mean_runtime_s=float(np.mean(single_runtimes)),
                                        single_path_mean_final_tensor_elements=float(np.mean(single_final_tensor_elements)),
                                        single_path_std_final_tensor_elements=float(np.std(single_final_tensor_elements)),
                                        single_path_mean_peak_tensor_elements=float(np.mean(single_peak_tensor_elements)),
                                        single_path_std_peak_tensor_elements=float(np.std(single_peak_tensor_elements)),
                                        single_path_mean_final_bytes_estimate=float(np.mean(single_final_bytes_estimate)),
                                        single_path_std_final_bytes_estimate=float(np.std(single_final_bytes_estimate)),
                                        single_path_mean_peak_bytes_estimate=float(np.mean(single_peak_bytes_estimate)),
                                        single_path_std_peak_bytes_estimate=float(np.std(single_peak_bytes_estimate)),
                                        multi_path_fidelity=multi_fidelity,
                                        multi_path_pruning_error=pruning_error,
                                        multi_path_runtime_s=multi_runtime,
                                        multi_path_final_tensor_elements=int(multi_memory_stats["final_tensor_elements"]),
                                        multi_path_peak_tensor_elements=int(multi_memory_stats["peak_tensor_elements"]),
                                        multi_path_final_bytes_estimate=int(multi_memory_stats["final_bytes_estimate"]),
                                        multi_path_peak_bytes_estimate=int(multi_memory_stats["peak_bytes_estimate"]),
                                        branch_count=branch_count,
                                        circuit_size=circuit_size,
                                    )
                                    rows.append(row)
                                    print("\n")
                                    print(json.dumps(asdict(row), sort_keys=True))

    with args.output.open("w", newline="", encoding="utf-8") as file_out:
        writer = csv.DictWriter(file_out, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()