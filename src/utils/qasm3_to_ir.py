# src/utils/qasm3_to_ir.py

import importlib.util
from src.circuit_ir import CircuitIR
from src.utils.quantum_gates import supported_ir_gates


UNSUPPORTED_DYNAMIC_OPS = {
    "measure",
    "reset",
    "if_else",
    "switch_case",
    "while_loop",
    "for_loop",
}

IGNORED_OPS = {
    "barrier",
}


def _resolve_param_value(param):
    """
    Converts instruction parameters to plain floats when possible.
    """
    try:
        return float(param)
    except Exception as exc:
        raise ValueError(f"Found non-numeric parameter '{param}'. Please bind all parameters before conversion.") from exc


def _extract_qubit_indices(circuit, instruction):
    indices = []
    for qbit in instruction.qubits:
        indices.append(circuit.find_bit(qbit).index)
    return indices


def qasm3_to_circuit_ir(qasm_text, basis_gates=None, apply_transpile=True, optimization_level=0):
    """
    Parses an OpenQASM 3 string and returns a CircuitIR instance.

    Args:
        qasm_text (str): Input OpenQASM 3 program text.
        basis_gates (list | None): Optional gate basis to transpile to.
        apply_transpile (bool): If True, run qiskit transpile before conversion.
        optimization_level (int): Qiskit transpiler optimization level.

    Returns:
        CircuitIR: Converted circuit in internal IR format.
    """
    try:
        from qiskit import qasm3, transpile
    except Exception as exc:
        raise ImportError(
            "Qiskit with OpenQASM 3 importer support is required. "
            "Install with: pip install qiskit qiskit_qasm3_import"
        ) from exc

    if importlib.util.find_spec("qiskit_qasm3_import") is None:
        raise ImportError(
            "Missing optional dependency 'qiskit_qasm3_import' required for OpenQASM 3 parsing. "
            "Install with: pip install qiskit_qasm3_import"
        )

    try:
        qc = qasm3.loads(qasm_text)
    except Exception as exc:
        if exc.__class__.__name__ == "MissingOptionalLibraryError":
            raise ImportError(
                "Missing optional dependency 'qiskit_qasm3_import' required for OpenQASM 3 parsing. "
                "Install with: pip install qiskit_qasm3_import"
            ) from exc
        raise ValueError(f"Unable to parse OpenQASM 3 input: {exc}") from exc

    if apply_transpile:
        if basis_gates is None:
            basis_gates = supported_ir_gates()
        transpile_kwargs = {"optimization_level": optimization_level}
        if basis_gates is not None:
            transpile_kwargs["basis_gates"] = basis_gates
        qc = transpile(qc, **transpile_kwargs)

    ir = CircuitIR(qc.num_qubits, qc.num_clbits)

    for instruction in qc.data:
        op_name = instruction.operation.name

        if op_name in IGNORED_OPS:
            continue

        if op_name in UNSUPPORTED_DYNAMIC_OPS:
            raise ValueError(
                f"Operation '{op_name}' is not supported yet by the non-dynamic IR path."
            )

        if len(instruction.clbits) > 0:
            raise ValueError(
                f"Operation '{op_name}' uses classical bits and is not supported yet by the non-dynamic IR path."
            )

        qubits = _extract_qubit_indices(qc, instruction)

        if len(qubits) == 0:
            raise ValueError(f"Operation '{op_name}' has no qubit target and cannot be simulated in the current path.")

        if len(qubits) > 2:
            raise ValueError(
                f"Operation '{op_name}' acts on {len(qubits)} qubits. "
                "Current path supports only 1- and 2-qubit gates."
            )

        if basis_gates is not None and op_name not in basis_gates:
            raise ValueError(
                f"Operation '{op_name}' is outside the selected basis_gates={basis_gates}."
            )

        params = [_resolve_param_value(p) for p in instruction.operation.params]
        ir.add_operation(op_name, qubits, params=params)

    return ir


def qasm3_file_to_circuit_ir(path, basis_gates=None, apply_transpile=True, optimization_level=0):
    """
    Reads an OpenQASM 3 file and converts it into CircuitIR.
    """
    with open(path, "r", encoding="utf-8") as file_in:
        qasm_text = file_in.read()

    return qasm3_to_circuit_ir(
        qasm_text,
        basis_gates=basis_gates,
        apply_transpile=apply_transpile,
        optimization_level=optimization_level,
    )
