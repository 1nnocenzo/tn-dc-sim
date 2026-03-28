# main.py

from src.simulation import DMRG, DMRG_from_circuit_ir
from src.utils.TN_gen import D_tree, D_mps
from src.utils.graph_gen import * # graph types
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


def main():
    """Configures and runs the simulation experiment."""

    input_mode = "legacy"           # "legacy" (existing random/qaoa flow) or "qasm3" (new IR path)

    no_qubits = 27                  # Total number of qubits (used in legacy mode; in qasm3 mode inferred from circuit)
    qubit_graph = nearest_neighbour_edge_list(no_qubits)  # Qubit connectivity graph
    circuit_type = "random"         # Circuit type: "random" or "qaoa"
    Depth = 5                       # Circuit depth (number of layers)

    network_type = "tree"           # Network type: "tree" (TTN) or "mps"
    qubit_order = "Blind"           # TTN ordering: "Naive" or "Blind"
    network_structure = [1, 3, 9, 27]  # TTN structure (branching hierarchy)

    compression_steps = 2           # Compression steps per depth
    no_sweeps = 2                   # Number of DMRG sweeps
    Dmax = [4, 8, 12]               # List of bond dimensions to test
    runs = 2                        # Number of independent runs per setup

    qasm_source_mode = "inline"     # "inline" or "file"
    qasm_file_path = "circuit.qasm"
    qasm_text = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
rz(0.2) q[2];
"""
    qasm_apply_transpile = True
    qasm_optimization_level = 0

    circuit_ir = None
    if input_mode == "qasm3":
        loaded_qasm_text = _load_qasm_text(qasm_source_mode, qasm_text, qasm_file_path)
        circuit_ir = qasm3_to_circuit_ir(
            loaded_qasm_text,
            apply_transpile=qasm_apply_transpile,
            optimization_level=qasm_optimization_level
        )
        no_qubits = circuit_ir.no_qubits
        if network_type == "tree" and network_structure[-1] != no_qubits:
            print(
                "network_structure does not match QASM3 qubit count. "
                f"Using default tree structure [1, {no_qubits}]"
            )
            network_structure = [1, no_qubits]
    elif input_mode != "legacy":
        raise ValueError("input_mode must be 'legacy' or 'qasm3'")

    Fidelity_list = []

    print(f"Starting simulations in mode: {input_mode}")

    for d_max in tqdm(Dmax, desc="D values"):

        if network_type == "tree":
            D = D_tree(network_structure,d_max)
        else:
            D = D_mps(no_qubits,d_max)
        Fidelity_run = 0
        for i in tqdm(range(runs), desc="Runs", leave=False):
            if input_mode == "legacy":
                edge_list = edge_extract(qubit_graph,qubit_order)
                fidelity_results = DMRG(
                    compression_steps=compression_steps,
                    depth=Depth,
                    no_sweeps=no_sweeps,
                    no_qubits=no_qubits,
                    D=D,
                    network_structure=network_structure,
                    full_edge_list=edge_list,
                    network_type=network_type,
                    circuit_type=circuit_type,
                    run=i
                )
            else:
                np.random.seed(i)
                fidelity_results = DMRG_from_circuit_ir(
                    compression_steps=compression_steps,
                    no_sweeps=no_sweeps,
                    D=D,
                    network_structure=network_structure,
                    circuit_ir=circuit_ir,
                    network_type=network_type
                )
            print(f"Run {i+1} with Dmax={d_max} complete. Final Fidelity: {fidelity_results}")
            Fidelity_run += fidelity_results
        Fidelity_list.append(Fidelity_run/runs)

    print("Bond Dimension, Fidelity")
    data = np.column_stack(( Dmax, Fidelity_list))
    print(data)

if __name__ == "__main__":
    main()
