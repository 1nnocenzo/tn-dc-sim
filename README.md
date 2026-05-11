# Dynamic Circuit Simulation with Tensor Networks (DMRG)

This repository implements dynamic-circuit simulation with tensor networks using a DMRG-style optimizer.

It is built on top of [AdityaD16/Quantum-Circuit-Simulator-DMRG](https://github.com/AdityaD16/Quantum-Circuit-Simulator-DMRG).

Supported simulation modes:
- `single-path`: samples one measurement/reset trajectory.
- `multi-branches`: keeps the full branch ensemble (with optional pruning).

Core implementation file:
- `src/simulation.py`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo script (it executes both single-path and multi-branches on a dynamic OpenQASM 3 circuit):

```bash
python main.py
```

`main.py` lets you configure:
- QASM source (`inline` or `file`)
- network type (`mps` or `tree`)
- `chunk_size`, `no_sweeps`, `dmax` sweep
- multi-branch pruning controls

## What The Simulator Does

For each mode, the simulator:
1. parses/receives a `CircuitIR` with dynamic instructions;
2. buffers unitary gates into chunks of size `chunk_size`;
3. flushes each chunk through a DMRG optimization sweep;
4. handles `measure` and `reset` explicitly on the tensor-network state.

`measure`/`reset` always force a chunk flush before being applied.

## APIs (Current, Canonical Names)

### Single-path

```python
DMRG_dynamic_single_path_from_circuit_ir(
    chunk_size,
    no_sweeps,
    bond_dims,
    network_structure,
    circuit_ir,
    network_type,
    seed=None,
    return_state=False,
    return_bra=False,
    return_classical_bits=False,
    return_branch_probability=False,
    n_samples_final=None,
    return_counts=False,
    return_shots=False,
    return_memory_stats=False,
)
```

```python
DMRG_dynamic_single_path_from_qasm3(
    chunk_size,
    no_sweeps,
    bond_dims,
    network_structure,
    qasm_text,
    network_type,
    seed=None,
    return_state=False,
    return_bra=False,
    return_classical_bits=False,
    return_branch_probability=False,
    basis_gates=None,
    apply_transpile=False,
    optimization_level=0,
    n_samples_final=None,
    return_counts=False,
    return_shots=False,
    return_memory_stats=False,
)
```

### Multi-branches

```python
DMRG_dynamic_multi_branches_from_circuit_ir(
    chunk_size,
    no_sweeps,
    bond_dims,
    network_structure,
    circuit_ir,
    network_type,
    max_branches=None,
    probability_cutoff=0.0,
    max_pruning_error=None,
    return_branches=False,
    return_state=False,
    return_bra=False,
    return_classical_bits=False,
    return_pruning_error=False,
    n_samples_final=None,
    seed=None,
    return_counts=False,
    return_shots=False,
    return_memory_stats=False,
)
```

```python
DMRG_dynamic_multi_branches_from_qasm3(
    chunk_size,
    no_sweeps,
    bond_dims,
    network_structure,
    qasm_text,
    network_type,
    max_branches=None,
    probability_cutoff=0.0,
    max_pruning_error=None,
    return_branches=False,
    return_state=False,
    return_bra=False,
    return_classical_bits=False,
    return_pruning_error=False,
    basis_gates=None,
    apply_transpile=False,
    optimization_level=0,
    n_samples_final=None,
    seed=None,
    return_counts=False,
    return_shots=False,
    return_memory_stats=False,
)
```

## Parameter Reference

### Common parameters

| Parameter | Meaning |
|---|---|
| `chunk_size` | Max number of unitary ops buffered before a DMRG flush (`>= 1`). |
| `no_sweeps` | Number of DMRG sweeps (must be positive and even). |
| `bond_dims` | Bond-dimension vector for the chosen network. |
| `network_structure` | Tree structure (used for `network_type="tree"`). |
| `network_type` | `"mps"` or `"tree"`. |
| `seed` | RNG seed for stochastic steps. |
| `n_samples_final` | Number of final-basis samples when `return_counts`/`return_shots` is requested. |
| `return_counts` | Return histogram of final computational-basis samples. |
| `return_shots` | Return explicit final sampled bitstrings. |
| `return_memory_stats` | Return memory counters (`tensor elements` + byte estimate). |

### Single-path only

| Parameter | Meaning |
|---|---|
| `return_branch_probability` | Return product of sampled measurement/reset outcome probabilities for that trajectory. |
| `return_classical_bits` | Return final classical register values (`np.ndarray`, length = `no_clbits`). |
| `return_state` | Return final state vector (flattened). |
| `return_bra` | Return final bra tensor network object. |

### Multi-branches only

| Parameter | Meaning |
|---|---|
| `max_branches` | Keep at most this many highest-probability branches after each non-unitary expansion and at the end. |
| `probability_cutoff` | Drop branches with absolute branch probability `< cutoff`. |
| `max_pruning_error` | Hard bound on discarded probability mass (`1 - surviving_mass`). |
| `return_pruning_error` | Return pruning error bound (`1 - surviving_mass`). |
| `return_branches` | Return branch payload records. |
| `return_classical_bits` | If `return_branches=True`, include `classical_bits` in each branch record; otherwise return list of branch bit-arrays as a separate output. |
| `return_state` | If `return_branches=True`, include `state` in each branch record; otherwise return list of branch states separately. |
| `return_bra` | If `return_branches=True`, include `bra` in each branch record; otherwise return list of branch bra objects separately. |

## Return Order (Important)

The return type is a scalar if only fidelity is requested; otherwise it is a tuple in fixed append order.

### Single-path output order

1. `fidelity`
2. `branch_probability` (if requested)
3. `classical_bits` (if requested)
4. `state` (if requested)
5. `bra` (if requested)
6. `counts` (if requested)
7. `shots` (if requested)
8. `memory_stats` (if requested)

### Multi-branches output order

1. `fidelity`
2. `pruning_error` (if requested)
3. `branches` (if `return_branches=True`)
4. `classical_bits_list` (if `return_branches=False` and `return_classical_bits=True`)
5. `state_list` (if `return_branches=False` and `return_state=True`)
6. `bra_list` (if `return_branches=False` and `return_bra=True`)
7. `counts` (if requested)
8. `shots` (if requested)
9. `memory_stats` (if requested)

Branch payload record keys:
- always: `probability`, `conditional_probability`, `branch_fidelity`
- optional: `classical_bits`, `state`, `bra`

## Fidelity Definitions Used In Code

- Single-path fidelity: product of per-chunk fidelities along the sampled trajectory.
- Multi-branches fidelity: arithmetic mean of surviving-branch fidelities (not probability-weighted).

If you need a probability-weighted multi-branch fidelity, compute it from `return_branches=True`.

## Network Configuration

### MPS

- Use `network_type="mps"`.
- Constraint: `len(bond_dims) == no_qubits - 1`.
- `network_structure` is ignored by the initializer.

### Tree

- Use `network_type="tree"`.
- Constraint: `network_structure[-1] == no_qubits`.
- `bond_dims` is typically built with `D_tree(network_structure, D_max)`.

Helpers in `src/utils/TN_gen.py`:
- `D_mps(no_qubits, D_max)`
- `D_tree(network_structure, D_max)`

## Dynamic OpenQASM 3 Support and Limits

Conversion path: `src/utils/qasm3_to_ir.py`.

Supported dynamic constructs:
- `measure`
- `reset`
- `if/else` on classical conditions (flattened to per-op conditions)

Supported condition operators in the internal condition IR:
- equality / inequality forms mapped to:
  - `clbit_eq`, `creg_eq`
- boolean composition:
  - `not`, `and`, `or`, `xor`

Current limits:
- only 1- and 2-qubit unitary gates;
- nested conditionals are rejected;
- `measure`/`reset` inside conditional branches are rejected by the converter;
- `switch`, `while`, `for` are not supported;
- with dynamic mode (`allow_dynamic=True`), `apply_transpile=True` is not supported.

## Gate Set Notes

Simulation of unitary ops relies on `single_qubit_gate_from_name` and `two_qubit_gate_from_name` in `src/utils/quantum_gates.py`.

Main supported gates include:
- 1q: `id x y z h s sdg t tdg sx sxdg rx ry rz p u1 u2 u3 u`
- 2q: `cx cz swap rzz`

If your QASM uses other gates, convert/decompose them before simulation.

## Minimal Usage Example

```python
from src.simulation import (
    DMRG_dynamic_single_path_from_qasm3,
    DMRG_dynamic_multi_branches_from_qasm3,
)
from src.utils.TN_gen import D_mps

qasm_text = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c[0] = measure q[0];
if (c[0]) { x q[2]; }
c[1] = measure q[1];
"""

no_qubits = 3
bond_dims = D_mps(no_qubits, D_max=8)

sp = DMRG_dynamic_single_path_from_qasm3(
    chunk_size=2,
    no_sweeps=2,
    bond_dims=bond_dims,
    network_structure=[1, 3],
    qasm_text=qasm_text,
    network_type="mps",
    seed=123,
    return_branch_probability=True,
    return_classical_bits=True,
)

mp = DMRG_dynamic_multi_branches_from_qasm3(
    chunk_size=2,
    no_sweeps=2,
    bond_dims=bond_dims,
    network_structure=[1, 3],
    qasm_text=qasm_text,
    network_type="mps",
    max_branches=None,
    probability_cutoff=0.0,
    return_pruning_error=True,
    return_branches=True,
    return_classical_bits=True,
)
```

## Naming Migration (Old -> Current)

Use these names in all new code:
- `compression_steps` -> `chunk_size`
- `D` -> `bond_dims`
- `DMRG_dynamic_all_branches_from_circuit_ir` -> `DMRG_dynamic_multi_branches_from_circuit_ir`

`DMRG(...)` (legacy static entry-point) is not part of the current dynamic API.
