# Dynamic Ciruit Simulation with Tensor Networks using DMRG

This repository implements simulations of **dynamic circuits** with **Tree Tensor Networks (TTNs)** and **Matrix Product States (MPS)** using the **Density Matrix Renormalization Group (DMRG)** algorithm. 


## Reference

This repository started as a modified version of [AdityaD16/Quantum-Circuit-Simulator-DMRG](https://github.com/AdityaD16/Quantum-Circuit-Simulator-DMRG).


## Installation
Install all the required dependencies with `pip install -r requirements.txt`.

<!-- ## Simulation Parameters

 `main.py` simulates the circuit using TTN or MPS and outputs fidelity results for each bond dimension. The simulation is configured via parameters in `main.py`. Below is a description of each parameter:

| Parameter            | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| `no_qubits`          | Total number of qubits in the circuit. |
| `qubit_graph`        | Connectivity graph of qubits. e.g. nearest neighbour, 3-regular etc. (See `graph_gen.py`) |
| `network_type`       | Tensor network representation: `"tree"` for Tree Tensor Network (TTN) or `"mps"` for Matrix Product State (MPS) |
| `qubit_order`        | Qubit ordering strategy used in TTN construction. `"Naive"` or`"Blind"`(see Sec. III B [arXiv:2504.16718](https://arxiv.org/abs/2504.16718)). |
| `network_structure`  | Defines the hierarchical structure of the TTN, e.g., `[1, 3, 9, 27]` for a ternary tree, `[1, 2, 4, 8, 16]` for a binary tree.  |
| `chunk_size`         | Max number of unitary operations per DMRG chunk |
| `no_sweeps`          | Number of DMRG sweeps|
| `Dmax`               | List of max bond dimensions to use in the tensor network |
| `runs`               | Number of independent simulations to average over | -->

## Main Input Mode
- set `qasm_source_mode` to `"inline"` or `"file"`,
- set `qasm_text` or `qasm_file_path`,
- optionally tune `qasm_apply_transpile` and `qasm_optimization_level`.

## CircuitIR and QASM3 Conversion

The project includes a minimal internal circuit representation (`CircuitIR`) and a converter from OpenQASM 3:

- `src/circuit_ir.py`
- `src/utils/qasm3_to_ir.py`

Example usage:

```python
from src.utils.qasm3_to_ir import qasm3_to_circuit_ir

qasm_text = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

ir = qasm3_to_circuit_ir(
    qasm_text,
    basis_gates=["id", "x", "y", "z", "h", "sx", "rx", "ry", "rz", "cx", "cz"],
    apply_transpile=True,
    optimization_level=0,
)

print(ir)
print(ir.operations)
```

To run the dynamic single-path DMRG simulation directly from OpenQASM 3:

```python
from src.simulation import DMRG_dynamic_single_path_from_qasm3
from src.utils.TN_gen import D_tree

qasm_text = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
rz(0.2) q[2];
"""

network_structure = [1, 3]
bond_dims = D_tree(network_structure, D_max=4)

fidelity = DMRG_dynamic_single_path_from_qasm3(
    chunk_size=1,
    no_sweeps=2,
    bond_dims=bond_dims,
    network_structure=network_structure,
    qasm_text=qasm_text,
    network_type="tree",
    optimization_level=0,
)

print("Final fidelity:", fidelity)
```

To also get the final approximate quantum state vector:

```python
fidelity, state = DMRG_dynamic_single_path_from_qasm3(
    chunk_size=1,
    no_sweeps=2,
    bond_dims=bond_dims,
    network_structure=network_structure,
    qasm_text=qasm_text,
    network_type="tree",
    optimization_level=0,
    return_state=True,
)
print(fidelity, state.shape)
```

It includes:
- QASM3 -> CircuitIR conversion check
- TTN/MPS DMRG runs on a small non-dynamic circuit
- expected failure test on a dynamic instruction (`measure`)
