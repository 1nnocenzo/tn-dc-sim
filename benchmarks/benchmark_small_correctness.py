"""Small-size correctness benchmarks for dynamic and non-dynamic circuits.

This script focuses on two sanity checks:

1. GHZ state preparation, compared against exact statevector simulation.
2. Quantum teleportation with mid-circuit measurements and feed-forward,
   compared branch-by-branch against the expected teleported output state.

Run from the repository root, for example:

    python benchmarks/benchmark_small_correctness.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation import (  # noqa: E402
    DMRG_dynamic_all_branches_from_qasm3,
    DMRG_dynamic_single_path_from_qasm3,
    DMRG_from_circuit_ir,
)
from src.utils.TN_gen import D_mps, D_tree  # noqa: E402
from src.utils.qasm3_to_ir import qasm3_to_circuit_ir  # noqa: E402


def build_ghz_circuit(no_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(no_qubits)
    qc.h(0)
    for target_qubit in range(1, no_qubits):
        qc.cx(target_qubit - 1, target_qubit)
    return qc


def build_dynamic_ghz_qasm_text(no_qubits: int) -> str:
    if no_qubits < 1:
        raise ValueError("no_qubits must be at least 1")

    measurement_count = 0 if no_qubits <= 2 else (no_qubits - (3 if no_qubits % 2 else 2)) // 2
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{no_qubits}] q;",
    ]
    if measurement_count > 0:
        lines.append(f"bit[{measurement_count}] c;")

    if no_qubits == 1:
        lines.append("h q[0];")
        return "\n".join(lines)

    if no_qubits == 2:
        lines.extend([
            "h q[0];",
            "cx q[0], q[1];",
        ])
        return "\n".join(lines)

    if no_qubits % 2 == 1:
        lines.extend([
            "h q[0];",
            "cx q[0], q[1];",
            "cx q[1], q[2];",
        ])
        boundary_qubit = 2
        next_qubit = 3
    else:
        lines.extend([
            "h q[0];",
            "cx q[0], q[1];",
        ])
        boundary_qubit = 1
        next_qubit = 2

    classical_bit = 0
    while next_qubit < no_qubits:
        partner_qubit = next_qubit + 1
        lines.extend([
            f"cx q[{boundary_qubit}], q[{next_qubit}];",
            f"measure q[{next_qubit}] -> c[{classical_bit}];",
            f"if (c[{classical_bit}] == 1) x q[{partner_qubit}];",
            f"reset q[{next_qubit}];",
            f"cx q[{partner_qubit}], q[{next_qubit}];",
        ])
        boundary_qubit = partner_qubit
        next_qubit += 2
        classical_bit += 1

    return "\n".join(lines)


def teleportation_qasm_text(theta: float = 0.37, phi: float = 0.21) -> str:
    return f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c;

ry({theta}) q[0];
rz({phi}) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c[1] == 1) x q[2];
if (c[0] == 1) z q[2];
""".strip()


def input_teleportation_state(theta: float = 0.37, phi: float = 0.21) -> DensityMatrix:
    prep = QuantumCircuit(1)
    prep.ry(theta, 0)
    prep.rz(phi, 0)
    return DensityMatrix(Statevector.from_instruction(prep))


def ttn_structure_for_qubits(no_qubits: int) -> list[int]:
    return [1, no_qubits]


def bitstring_from_classical_bits(bits) -> str:
    return "".join(str(int(bit)) for bit in bits)


def make_result_writer(output_jsonl: Path | None) -> Callable[[dict[str, object]], None]:
    if output_jsonl is None:
        return lambda payload: print(json.dumps(payload, sort_keys=True))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _writer(payload: dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line)
        with output_jsonl.open("a", encoding="utf-8") as file_out:
            file_out.write(line + "\n")

    return _writer


def benchmark_ghz(
    qubit_counts: list[int],
    dmax_values: list[int],
    compression_steps_values: list[int],
    no_sweeps: int,
    emit_result: Callable[[dict[str, object]], None],
) -> None:
    print("# GHZ benchmark")
    for no_qubits in qubit_counts:
        exact_state = Statevector.from_instruction(build_ghz_circuit(no_qubits))
        qasm_text = build_dynamic_ghz_qasm_text(no_qubits)

        for network_type in ["tree", "mps"]:
            for dmax in dmax_values:
                for compression_steps in compression_steps_values:
                    if network_type == "tree":
                        structure = ttn_structure_for_qubits(no_qubits)
                        D = D_tree(structure, dmax)
                    else:
                        structure = [1, no_qubits]
                        D = D_mps(no_qubits, dmax)

                    multi_start = time.perf_counter()
                    multi_fidelity, pruning_error, branches = DMRG_dynamic_all_branches_from_qasm3(
                        compression_steps=compression_steps,
                        no_sweeps=no_sweeps,
                        D=D,
                        network_structure=structure,
                        qasm_text=qasm_text,
                        network_type=network_type,
                        return_branches=True,
                        return_state=True,
                        return_classical_bits=True,
                        return_pruning_error=True,
                    )
                    multi_runtime = time.perf_counter() - multi_start

                    branch_report = []
                    for branch in branches:
                        branch_state = Statevector(branch["state"])
                        branch_report.append(
                            {
                                "classical_bits": bitstring_from_classical_bits(branch["classical_bits"]),
                                "probability": float(branch["probability"]),
                                "conditional_probability": float(branch["conditional_probability"]),
                                "branch_fidelity": float(branch["branch_fidelity"]),
                                "exact_state_fidelity": float(state_fidelity(exact_state, branch_state)),
                            }
                        )

                    exact_branch_fidelities = [entry["exact_state_fidelity"] for entry in branch_report]
                    emit_result(
                        {
                            "case": "ghz_dynamic",
                            "network_type": network_type,
                            "no_qubits": no_qubits,
                            "Dmax": dmax,
                            "compression_steps": compression_steps,
                            "no_sweeps": no_sweeps,
                            "multi_path_fidelity": float(multi_fidelity),
                            "multi_path_pruning_error": float(pruning_error),
                            "multi_path_runtime_ms": float(multi_runtime * 1_000.0),
                            "branch_count": len(branch_report),
                            "mean_exact_state_fidelity": float(np.mean(exact_branch_fidelities)) if exact_branch_fidelities else 0.0,
                            "branch_details": branch_report,
                        }
                    )


def benchmark_teleportation(
    dmax_values: list[int],
    compression_steps_values: list[int],
    no_sweeps: int,
    single_path_repeats: int,
    emit_result: Callable[[dict[str, object]], None],
) -> None:
    print("# Teleportation benchmark")
    qasm_text = teleportation_qasm_text()
    input_rho = input_teleportation_state()

    for network_type in ["tree", "mps"]:
        for dmax in dmax_values:
            for compression_steps in compression_steps_values:
                if network_type == "tree":
                    structure = ttn_structure_for_qubits(3)
                    D = D_tree(structure, dmax)
                else:
                    structure = [1, 3]
                    D = D_mps(3, dmax)

                multi_start = time.perf_counter()
                multi_fidelity, pruning_error, branches = DMRG_dynamic_all_branches_from_qasm3(
                    compression_steps=compression_steps,
                    no_sweeps=no_sweeps,
                    D=D,
                    network_structure=structure,
                    qasm_text=qasm_text,
                    network_type=network_type,
                    return_branches=True,
                    return_state=True,
                    return_classical_bits=True,
                    return_pruning_error=True,
                )
                multi_runtime = time.perf_counter() - multi_start

                branch_report = []
                branch_probabilities = []
                for branch in branches:
                    branch_state = Statevector(branch["state"])
                    output_rho = partial_trace(DensityMatrix(branch_state), [0, 1])
                    branch_probability = float(branch["probability"])
                    branch_probabilities.append(branch_probability)
                    branch_report.append(
                        {
                            "classical_bits": bitstring_from_classical_bits(branch["classical_bits"]),
                            "probability": branch_probability,
                            "conditional_probability": float(branch["conditional_probability"]),
                            "branch_fidelity": float(branch["branch_fidelity"]),
                            "teleportation_fidelity": float(state_fidelity(output_rho, input_rho)),
                        }
                    )

                counts = Counter()
                single_path_fidelities = []
                single_path_branch_probs = []
                for repeat in range(single_path_repeats):
                    single_seed = 10_000 + repeat
                    single_fidelity, branch_probability, classical_bits = DMRG_dynamic_single_path_from_qasm3(
                        compression_steps=compression_steps,
                        no_sweeps=no_sweeps,
                        D=D,
                        network_structure=structure,
                        qasm_text=qasm_text,
                        network_type=network_type,
                        seed=single_seed,
                        return_classical_bits=True,
                        return_branch_probability=True,
                    )
                    counts[bitstring_from_classical_bits(classical_bits)] += 1
                    single_path_fidelities.append(float(single_fidelity))
                    single_path_branch_probs.append(float(branch_probability))

                emit_result(
                    {
                        "case": "teleportation",
                        "network_type": network_type,
                        "Dmax": dmax,
                        "compression_steps": compression_steps,
                        "no_sweeps": no_sweeps,
                        "multi_path_fidelity": float(multi_fidelity),
                        "multi_path_pruning_error": float(pruning_error),
                        "multi_path_runtime_ms": float(multi_runtime * 1_000.0),
                        "branch_count": len(branch_report),
                        "branch_probabilities": branch_probabilities,
                        "branch_details": branch_report,
                        "single_path_repeat_count": single_path_repeats,
                        "single_path_mean_fidelity": float(np.mean(single_path_fidelities)),
                        "single_path_std_fidelity": float(np.std(single_path_fidelities)),
                        "single_path_mean_branch_probability": float(np.mean(single_path_branch_probs)),
                        "single_path_frequency": dict(counts),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small correctness benchmarks for TTN/MPS DMRG.")
    parser.add_argument(
        "--ghz-qubits",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="Qubit counts to test for the dynamic GHZ benchmark.",
    )
    parser.add_argument("--dmax", type=int, nargs="+", default=[2, 4, 8], help="Bond dimensions to test.")
    parser.add_argument(
        "--compression-steps",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Compression steps to test.",
    )
    parser.add_argument("--no-sweeps", type=int, default=2, help="Number of DMRG sweeps.")
    parser.add_argument(
        "--single-path-repeats",
        type=int,
        default=32,
        help="Number of single-path samples for the teleportation benchmark.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path where JSONL benchmark rows are written (and still printed to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_jsonl is not None:
        args.output_jsonl.write_text("", encoding="utf-8")

    emit_result = make_result_writer(args.output_jsonl)

    benchmark_ghz(args.ghz_qubits, args.dmax, args.compression_steps, args.no_sweeps, emit_result)
    benchmark_teleportation(args.dmax, args.compression_steps, args.no_sweeps, args.single_path_repeats, emit_result)


if __name__ == "__main__":
    main()