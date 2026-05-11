# main.py

from src.simulation import (
    DMRG_dynamic_multi_branches_from_circuit_ir,
    DMRG_dynamic_single_path_from_circuit_ir,
)
from src.utils.TN_gen import D_tree, D_mps
from src.utils.qasm3_to_ir import qasm3_to_circuit_ir
from tqdm import tqdm
import numpy as np


def _load_qasm_text(source_mode, inline_text, file_path):
    if source_mode == "inline":
        return inline_text
    if source_mode == "file":
        with open(file_path, "r", encoding="utf-8") as file_in:
            return file_in.read()
    raise ValueError("qasm_source_mode must be 'inline' or 'file'")


def _default_dynamic_qasm3():
    """
    Small dynamic circuit with:
    - intermediate measurements,
    - classically conditioned gates,
    - reset.
    """
    return """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];

c[0] = measure q[0];
if (c[0]) {
    x q[2];
}

c[1] = measure q[1];
if (c[0] && c[1]) {
    z q[2];
}

reset q[1];
c[2] = measure q[2];
"""


def _resolve_network_structure(network_type, network_structure, no_qubits):
    if network_type == "tree":
        if network_structure[-1] != no_qubits:
            print(
                "network_structure does not match QASM3 qubit count. "
                f"Using default tree structure [1, {no_qubits}]"
            )
            return [1, no_qubits]
        return network_structure
    return network_structure


def _make_bond_dims(network_type, network_structure, no_qubits, d_max):
    if network_type == "tree":
        return D_tree(network_structure, d_max)
    return D_mps(no_qubits, d_max)


def _print_top_branches(branch_payload, top_k=5):
    if len(branch_payload) == 0:
        print("  No surviving branches.")
        return
    ordered = sorted(branch_payload, key=lambda x: x["probability"], reverse=True)
    print(f"  Top {min(top_k, len(ordered))} branches by probability:")
    for idx, branch in enumerate(ordered[:top_k], start=1):
        classical_bits = branch.get("classical_bits")
        if classical_bits is not None and hasattr(classical_bits, "tolist"):
            classical_bits = classical_bits.tolist()
        print(
            "   "
            f"{idx}. p={branch['probability']:.6f}, "
            f"p_cond={branch['conditional_probability']:.6f}, "
            f"fidelity={branch['branch_fidelity']:.6f}, "
            f"classical_bits={classical_bits}"
        )


def main():
    """Run a dynamic-circuit demo with both single-path and multi-branches modes."""

    # Simulation configuration.
    network_type = "mps"               # "tree" or "mps"
    network_structure = [1, 3]         # used when network_type == "tree"
    chunk_size = 2
    no_sweeps = 2
    dmax_values = [2, 4, 8]

    # Single-path is stochastic at measurement time: run multiple seeds.
    single_path_runs = 4
    single_path_seed_base = 1234

    # Multi-branches controls.
    max_branches = None                # e.g. 8 to cap branch count
    probability_cutoff = 0.0
    max_pruning_error = None

    # QASM input.
    qasm_source_mode = "inline"        # "inline" or "file"
    qasm_file_path = "circuit.qasm"
    qasm_text = _default_dynamic_qasm3()
    qasm_apply_transpile = False       # required with allow_dynamic=True
    qasm_optimization_level = 0

    loaded_qasm_text = _load_qasm_text(qasm_source_mode, qasm_text, qasm_file_path)
    circuit_ir = qasm3_to_circuit_ir(
        loaded_qasm_text,
        apply_transpile=qasm_apply_transpile,
        optimization_level=qasm_optimization_level,
        allow_dynamic=True,
    )

    no_qubits = circuit_ir.no_qubits
    network_structure = _resolve_network_structure(network_type, network_structure, no_qubits)

    print("\nDynamic QASM3 demo")
    print(f"  qubits={circuit_ir.no_qubits}, clbits={circuit_ir.no_clbits}, ops={len(circuit_ir.operations)}")
    print(f"  network_type={network_type}, network_structure={network_structure}")
    print(f"  chunk_size={chunk_size}, no_sweeps={no_sweeps}")

    for d_max in tqdm(dmax_values, desc="d_max sweep"):
        bond_dims = _make_bond_dims(network_type, network_structure, no_qubits, d_max)
        print(f"\n=== d_max={d_max} | bond_dims={bond_dims} ===")

        try:
            # Single-path runs.
            sp_fidelities = []
            sp_branch_probs = []
            sp_peak_elements = []
            for run in range(single_path_runs):
                seed = single_path_seed_base + run
                sp_fidelity, sp_branch_prob, sp_bits, sp_memory = DMRG_dynamic_single_path_from_circuit_ir(
                    chunk_size=chunk_size,
                    no_sweeps=no_sweeps,
                    bond_dims=bond_dims,
                    network_structure=network_structure,
                    circuit_ir=circuit_ir,
                    network_type=network_type,
                    seed=seed,
                    return_branch_probability=True,
                    return_classical_bits=True,
                    return_memory_stats=True,
                )
                sp_fidelities.append(float(sp_fidelity))
                sp_branch_probs.append(float(sp_branch_prob))
                sp_peak_elements.append(int(sp_memory["peak_tensor_elements"]))
                bits = sp_bits.tolist() if hasattr(sp_bits, "tolist") else sp_bits
                print(
                    "  [single-path] "
                    f"run={run + 1}, seed={seed}, fidelity={sp_fidelity:.6f}, "
                    f"branch_prob={sp_branch_prob:.6f}, classical_bits={bits}, "
                    f"peak_tensors={sp_memory['peak_tensor_elements']}"
                )

            print(
                "  [single-path summary] "
                f"fidelity_mean={np.mean(sp_fidelities):.6f}, fidelity_std={np.std(sp_fidelities):.6f}, "
                f"branch_prob_mean={np.mean(sp_branch_probs):.6f}, peak_tensors_mean={np.mean(sp_peak_elements):.1f}"
            )

            # Multi-branches run.
            mp_fidelity, mp_pruning_error, mp_branches, mp_memory = DMRG_dynamic_multi_branches_from_circuit_ir(
                chunk_size=chunk_size,
                no_sweeps=no_sweeps,
                bond_dims=bond_dims,
                network_structure=network_structure,
                circuit_ir=circuit_ir,
                network_type=network_type,
                max_branches=max_branches,
                probability_cutoff=probability_cutoff,
                max_pruning_error=max_pruning_error,
                return_branches=True,
                return_classical_bits=True,
                return_pruning_error=True,
                return_memory_stats=True,
            )

            print(
                "  [multi-branches] "
                f"fidelity={mp_fidelity:.6f}, pruning_error={mp_pruning_error:.6f}, "
                f"n_branches={len(mp_branches)}, peak_tensors={mp_memory['peak_tensor_elements']}"
            )
            _print_top_branches(mp_branches, top_k=5)
        except PermissionError as exc:
            raise RuntimeError(
                "Simulation failed due OS semaphore/process-pool limits "
                "(joblib/loky via cotengra). Run outside restricted sandbox "
                "or disable process-based contraction-path parallelism."
            ) from exc


if __name__ == "__main__":
    main()
