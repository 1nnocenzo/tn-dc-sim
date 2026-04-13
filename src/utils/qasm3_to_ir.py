# src/utils/qasm3_to_ir.py

import importlib.util
import math
import re
from src.circuit_ir import CircuitIR
from src.utils.quantum_gates import supported_ir_gates


UNSUPPORTED_DYNAMIC_OPS = {
    "switch_case",
    "while_loop",
    "for_loop",
}

IGNORED_OPS = {
    "barrier",
}


def _normalize_qasm3_bit_conditions(qasm_text):
    """
    Rewrites single-bit numeric comparisons into boolean comparisons so they are
    accepted by qiskit_qasm3_import.

    Examples:
    - c[0] == 1   -> c[0] == true
    - c[0] == 0   -> c[0] == false
    - b == 1      -> b == true   (only if `b` is declared as scalar `bit b;`)
    """
    scalar_bit_names = set()
    for match in re.finditer(r"\bbit\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", qasm_text):
        scalar_bit_names.add(match.group(1))

    normalized = qasm_text

    # Indexed bit references, e.g. c[0] == 1
    normalized = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*\d+\s*\])\s*==\s*1\b",
        r"\1 == true",
        normalized,
    )
    normalized = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*\d+\s*\])\s*==\s*0\b",
        r"\1 == false",
        normalized,
    )

    # Scalar bit references, only for variables declared as `bit b;`.
    if len(scalar_bit_names) > 0:
        names_pattern = "|".join(sorted((re.escape(name) for name in scalar_bit_names), key=len, reverse=True))
        normalized = re.sub(
            rf"\b({names_pattern})\b\s*==\s*1\b",
            r"\1 == true",
            normalized,
        )
        normalized = re.sub(
            rf"\b({names_pattern})\b\s*==\s*0\b",
            r"\1 == false",
            normalized,
        )

    return normalized


def _resolve_param_value(param):
    """
    Converts instruction parameters to plain floats when possible.
    """
    try:
        return float(param)
    except Exception as exc:
        raise ValueError(f"Found non-numeric parameter '{param}'. Please bind all parameters before conversion.") from exc


def _extract_qubit_indices(circuit, instruction, qubit_map):
    indices = []
    for qbit in instruction.qubits:
        local_index = circuit.find_bit(qbit).index
        indices.append(qubit_map[local_index])
    return indices


def _extract_clbit_indices(circuit, instruction, clbit_map):
    indices = []
    for cbit in instruction.clbits:
        local_index = circuit.find_bit(cbit).index
        indices.append(clbit_map[local_index])
    return indices


def _negate_condition(condition):
    if condition is None:
        return None
    if condition.get("kind") == "not":
        return condition["term"]
    return {"kind": "not", "term": condition}


def _combine_nary_condition(kind, terms):
    flat_terms = []
    for term in terms:
        if term is None:
            continue
        if isinstance(term, dict) and term.get("kind") == kind and isinstance(term.get("terms"), list):
            flat_terms.extend(term["terms"])
        else:
            flat_terms.append(term)

    if len(flat_terms) == 0:
        return None
    if len(flat_terms) == 1:
        return flat_terms[0]
    return {"kind": kind, "terms": flat_terms}


def _combine_conditions_and(lhs, rhs):
    return _combine_nary_condition("and", [lhs, rhs])


def _condition_from_qiskit(condition, circuit, clbit_map):
    if condition is None:
        return None

    if not isinstance(condition, tuple) or len(condition) != 2:
        raise ValueError(f"Unsupported condition format: {condition}")

    lhs, rhs = condition
    lhs_type_name = lhs.__class__.__name__

    if lhs_type_name == "Clbit":
        local_clbit = circuit.find_bit(lhs).index
        global_clbit = clbit_map[local_clbit]
        value = 1 if bool(rhs) else 0
        return {"kind": "clbit_eq", "clbit": global_clbit, "value": value}

    if lhs_type_name == "ClassicalRegister":
        if rhs is None:
            raise ValueError("ClassicalRegister condition value cannot be None")
        register_value = int(rhs)
        register_clbits = []
        for i in range(lhs.size):
            local_clbit = circuit.find_bit(lhs[i]).index
            register_clbits.append(clbit_map[local_clbit])
        return {"kind": "creg_eq", "clbits": register_clbits, "value": register_value}

    raise ValueError(f"Unsupported condition lhs type '{lhs_type_name}'")


def _append_qiskit_instructions_to_ir(circuit, instructions, ir, allow_dynamic, basis_gates, qubit_map, clbit_map, inherited_condition=None):
    for instruction in instructions:
        op = instruction.operation
        op_name = op.name

        if op_name in IGNORED_OPS:
            continue

        if op_name in UNSUPPORTED_DYNAMIC_OPS:
            raise ValueError(
                f"Operation '{op_name}' is not supported yet by this IR converter."
            )

        local_condition = _condition_from_qiskit(getattr(op, "condition", None), circuit, clbit_map)
        op_condition = _combine_conditions_and(inherited_condition, local_condition)

        if op_name == "if_else":
            if not allow_dynamic:
                raise ValueError(
                    "Operation 'if_else' is not supported by the non-dynamic IR path."
                )
            if inherited_condition is not None:
                raise ValueError(
                    "Nested conditionals are not supported by this IR converter."
                )
            branch_condition = _condition_from_qiskit(op.condition, circuit, clbit_map)
            true_condition = _combine_conditions_and(inherited_condition, branch_condition)
            false_condition = _combine_conditions_and(inherited_condition, _negate_condition(branch_condition))

            inner_qubit_map = []
            for qbit in instruction.qubits:
                local_q = circuit.find_bit(qbit).index
                inner_qubit_map.append(qubit_map[local_q])

            inner_clbit_map = []
            for cbit in instruction.clbits:
                local_c = circuit.find_bit(cbit).index
                inner_clbit_map.append(clbit_map[local_c])

            true_block = op.blocks[0] if len(op.blocks) > 0 else None
            false_block = op.blocks[1] if len(op.blocks) > 1 else None

            if true_block is not None:
                _append_qiskit_instructions_to_ir(
                    true_block,
                    true_block.data,
                    ir,
                    allow_dynamic,
                    basis_gates,
                    inner_qubit_map,
                    inner_clbit_map,
                    true_condition,
                )

            if false_block is not None:
                _append_qiskit_instructions_to_ir(
                    false_block,
                    false_block.data,
                    ir,
                    allow_dynamic,
                    basis_gates,
                    inner_qubit_map,
                    inner_clbit_map,
                    false_condition,
                )
            continue

        if op_name == "measure":
            if not allow_dynamic:
                raise ValueError(
                    "Operation 'measure' is not supported by the non-dynamic IR path."
                )
            if inherited_condition is not None:
                raise ValueError(
                    "Only unitary gates are supported inside conditional branches."
                )
            qubits = _extract_qubit_indices(circuit, instruction, qubit_map)
            clbits = _extract_clbit_indices(circuit, instruction, clbit_map)
            if len(qubits) != 1 or len(clbits) != 1:
                raise ValueError("Measure is supported only in the form measure q[i] -> c[j].")
            ir.add_operation(op_name, qubits, condition=op_condition, clbits=clbits)
            continue

        if op_name == "reset":
            if not allow_dynamic:
                raise ValueError(
                    "Operation 'reset' is not supported by the non-dynamic IR path."
                )
            if inherited_condition is not None:
                raise ValueError(
                    "Only unitary gates are supported inside conditional branches."
                )
            qubits = _extract_qubit_indices(circuit, instruction, qubit_map)
            if len(qubits) != 1:
                raise ValueError("Reset is supported only for one qubit at a time.")
            ir.add_operation(op_name, qubits, condition=op_condition)
            continue

        if not allow_dynamic:
            if op_condition is not None:
                raise ValueError(
                    f"Operation '{op_name}' has a classical condition and is not supported by the non-dynamic IR path."
                )
            if len(instruction.clbits) > 0:
                raise ValueError(
                    f"Operation '{op_name}' uses classical bits and is not supported by the non-dynamic IR path."
                )

        qubits = _extract_qubit_indices(circuit, instruction, qubit_map)
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

        params = [_resolve_param_value(p) for p in op.params]
        ir.add_operation(op_name, qubits, params=params, condition=op_condition)


def _is_qiskit_condition_conversion_error(exc):
    message = str(exc)
    return (
        "unhandled binary operator" in message
        or "unhandled unary operator" in message
        or "only '==' is supported in register comparisons" in message
    )


def _ast_operator_symbol(operator):
    if hasattr(operator, "name"):
        return operator.name
    value = getattr(operator, "value", None)
    if isinstance(value, str):
        return value
    return str(operator)


def _ast_numeric_value(expr):
    expr_type = expr.__class__.__name__

    if expr_type == "IntegerLiteral":
        return float(int(expr.value))
    if expr_type == "FloatLiteral":
        return float(expr.value)
    if expr_type == "BooleanLiteral":
        return 1.0 if bool(expr.value) else 0.0
    if expr_type == "Identifier":
        constants = {
            "pi": math.pi,
            "tau": 2.0 * math.pi,
            "euler": math.e,
        }
        if expr.name in constants:
            return float(constants[expr.name])
        raise ValueError(f"Unsupported symbolic parameter '{expr.name}'.")
    if expr_type == "UnaryExpression":
        op = _ast_operator_symbol(expr.op)
        value = _ast_numeric_value(expr.expression)
        if op == "+":
            return +value
        if op == "-":
            return -value
        raise ValueError(f"Unsupported unary numeric operator '{op}'.")
    if expr_type == "BinaryExpression":
        op = _ast_operator_symbol(expr.op)
        lhs = _ast_numeric_value(expr.lhs)
        rhs = _ast_numeric_value(expr.rhs)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            return lhs / rhs
        if op == "**":
            return lhs ** rhs
        raise ValueError(f"Unsupported binary numeric operator '{op}'.")

    raise ValueError(f"Unsupported numeric expression type '{expr_type}'.")


def _ast_integer_value(expr):
    value = _ast_numeric_value(expr)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-12:
        raise ValueError(f"Expression '{expr}' is not an integer value.")
    return rounded


def _resolve_ast_param_value(param_expr):
    return float(_ast_numeric_value(param_expr))


def _resolve_ast_indexed_name_and_offset(expr):
    expr_type = expr.__class__.__name__

    if expr_type == "IndexExpression":
        if expr.collection.__class__.__name__ != "Identifier":
            raise ValueError("Only indexed identifiers are supported in classical/quantum references.")
        if len(expr.index) != 1:
            raise ValueError("Only one-dimensional indexing is supported.")
        return expr.collection.name, _ast_integer_value(expr.index[0])

    if expr_type == "IndexedIdentifier":
        if expr.name.__class__.__name__ != "Identifier":
            raise ValueError("Only indexed identifiers are supported in classical/quantum references.")
        if len(expr.indices) != 1 or len(expr.indices[0]) != 1:
            raise ValueError("Only one-dimensional indexing is supported.")
        return expr.name.name, _ast_integer_value(expr.indices[0][0])

    raise ValueError(f"Unsupported indexed reference type '{expr_type}'.")


def _resolve_ast_qubit_reference(expr, qubit_ranges):
    expr_type = expr.__class__.__name__

    if expr_type == "Identifier":
        name = expr.name
        if name not in qubit_ranges:
            raise ValueError(f"Unknown qubit register '{name}'.")
        return list(qubit_ranges[name])

    name, offset = _resolve_ast_indexed_name_and_offset(expr)
    if name not in qubit_ranges:
        raise ValueError(f"Unknown qubit register '{name}'.")
    reg = qubit_ranges[name]
    if offset < 0 or offset >= len(reg):
        raise ValueError(f"Qubit index {offset} out of range for register '{name}'.")
    return [reg[offset]]


def _resolve_ast_clbit_reference(expr, clbit_ranges):
    expr_type = expr.__class__.__name__

    if expr_type == "Identifier":
        name = expr.name
        if name not in clbit_ranges:
            raise ValueError(f"Unknown classical bit register '{name}'.")
        return list(clbit_ranges[name])

    name, offset = _resolve_ast_indexed_name_and_offset(expr)
    if name not in clbit_ranges:
        raise ValueError(f"Unknown classical bit register '{name}'.")
    reg = clbit_ranges[name]
    if offset < 0 or offset >= len(reg):
        raise ValueError(f"Classical bit index {offset} out of range for register '{name}'.")
    return [reg[offset]]


def _condition_operand_from_ast(expr, clbit_ranges):
    expr_type = expr.__class__.__name__

    if expr_type == "BooleanLiteral":
        return "bool", 1 if bool(expr.value) else 0

    if expr_type in {"Identifier", "IndexExpression", "IndexedIdentifier"}:
        indices = _resolve_ast_clbit_reference(expr, clbit_ranges)
        if len(indices) == 1:
            return "clbit", indices[0]
        return "creg", indices

    return "int", _ast_integer_value(expr)


def _clbit_eq_condition(clbit, value):
    int_value = int(value)
    if int_value not in (0, 1):
        raise ValueError("Classical-bit comparisons only support values 0 or 1.")
    return {"kind": "clbit_eq", "clbit": int(clbit), "value": int_value}


def _condition_from_openqasm_equality(lhs_expr, rhs_expr, clbit_ranges):
    lhs_kind, lhs_value = _condition_operand_from_ast(lhs_expr, clbit_ranges)
    rhs_kind, rhs_value = _condition_operand_from_ast(rhs_expr, clbit_ranges)

    if lhs_kind == "clbit" and rhs_kind in {"bool", "int"}:
        return _clbit_eq_condition(lhs_value, rhs_value)
    if rhs_kind == "clbit" and lhs_kind in {"bool", "int"}:
        return _clbit_eq_condition(rhs_value, lhs_value)

    if lhs_kind == "creg" and rhs_kind in {"bool", "int"}:
        return {"kind": "creg_eq", "clbits": list(lhs_value), "value": int(rhs_value)}
    if rhs_kind == "creg" and lhs_kind in {"bool", "int"}:
        return {"kind": "creg_eq", "clbits": list(rhs_value), "value": int(lhs_value)}

    if lhs_kind == "clbit" and rhs_kind == "clbit":
        lhs_true = _clbit_eq_condition(lhs_value, 1)
        rhs_true = _clbit_eq_condition(rhs_value, 1)
        return _negate_condition(_combine_nary_condition("xor", [lhs_true, rhs_true]))

    if lhs_kind in {"bool", "int"} and rhs_kind in {"bool", "int"}:
        return {"kind": "literal", "value": int(lhs_value) == int(rhs_value)}

    raise ValueError(
        f"Unsupported equality operands '{lhs_kind}' and '{rhs_kind}' in dynamic condition."
    )


def _condition_from_openqasm_expr(expr, clbit_ranges):
    expr_type = expr.__class__.__name__

    if expr_type == "BooleanLiteral":
        return {"kind": "literal", "value": bool(expr.value)}

    if expr_type in {"Identifier", "IndexExpression", "IndexedIdentifier"}:
        operand_kind, operand_value = _condition_operand_from_ast(expr, clbit_ranges)
        if operand_kind == "clbit":
            return _clbit_eq_condition(operand_value, 1)
        if operand_kind == "creg":
            return _negate_condition({"kind": "creg_eq", "clbits": list(operand_value), "value": 0})
        return {"kind": "literal", "value": bool(int(operand_value))}

    if expr_type == "UnaryExpression":
        op = _ast_operator_symbol(expr.op)
        if op == "!":
            return _negate_condition(_condition_from_openqasm_expr(expr.expression, clbit_ranges))
        raise ValueError(f"Unsupported unary boolean operator '{op}'.")

    if expr_type == "BinaryExpression":
        op = _ast_operator_symbol(expr.op)

        if op in {"&&", "||", "^"}:
            lhs_condition = _condition_from_openqasm_expr(expr.lhs, clbit_ranges)
            rhs_condition = _condition_from_openqasm_expr(expr.rhs, clbit_ranges)
            if op == "&&":
                return _combine_nary_condition("and", [lhs_condition, rhs_condition])
            if op == "||":
                return _combine_nary_condition("or", [lhs_condition, rhs_condition])
            return _combine_nary_condition("xor", [lhs_condition, rhs_condition])

        if op == "==":
            return _condition_from_openqasm_equality(expr.lhs, expr.rhs, clbit_ranges)

        if op == "!=":
            return _negate_condition(_condition_from_openqasm_equality(expr.lhs, expr.rhs, clbit_ranges))

        raise ValueError(f"Unsupported binary boolean operator '{op}'.")

    raise ValueError(f"Unsupported condition expression type '{expr_type}'.")


def _extract_register_maps_from_openqasm_ast(program):
    qubit_ranges = {}
    clbit_ranges = {}
    no_qubits = 0
    no_clbits = 0

    for statement in program.statements:
        stmt_type = statement.__class__.__name__

        if stmt_type == "QubitDeclaration":
            name = statement.qubit.name
            if name in qubit_ranges:
                raise ValueError(f"Qubit register '{name}' redeclared.")
            size = 1 if statement.size is None else _ast_integer_value(statement.size)
            assert size >= 1, "Qubit register size must be positive."
            qubit_ranges[name] = list(range(no_qubits, no_qubits + size))
            no_qubits += size
            continue

        if stmt_type == "ClassicalDeclaration" and statement.type.__class__.__name__ == "BitType":
            name = statement.identifier.name
            if name in clbit_ranges:
                raise ValueError(f"Classical bit register '{name}' redeclared.")
            size = 1 if statement.type.size is None else _ast_integer_value(statement.type.size)
            assert size >= 1, "Classical bit register size must be positive."
            clbit_ranges[name] = list(range(no_clbits, no_clbits + size))
            no_clbits += size
            continue

    return qubit_ranges, clbit_ranges, no_qubits, no_clbits


def _append_openqasm_statements_to_ir(statements, ir, allow_dynamic, basis_gates, qubit_ranges, clbit_ranges, inherited_condition=None):
    for statement in statements:
        stmt_type = statement.__class__.__name__

        if stmt_type in {"Include", "QubitDeclaration", "ClassicalDeclaration"}:
            continue

        if stmt_type == "QuantumBarrier":
            continue

        if stmt_type == "BranchingStatement":
            if not allow_dynamic:
                raise ValueError("Operation 'if' is not supported by the non-dynamic IR path.")
            if inherited_condition is not None:
                raise ValueError("Nested conditionals are not supported by this IR converter.")

            branch_condition = _condition_from_openqasm_expr(statement.condition, clbit_ranges)
            true_condition = _combine_conditions_and(inherited_condition, branch_condition)
            false_condition = _combine_conditions_and(inherited_condition, _negate_condition(branch_condition))

            _append_openqasm_statements_to_ir(
                statement.if_block,
                ir,
                allow_dynamic,
                basis_gates,
                qubit_ranges,
                clbit_ranges,
                inherited_condition=true_condition,
            )
            if len(statement.else_block) > 0:
                _append_openqasm_statements_to_ir(
                    statement.else_block,
                    ir,
                    allow_dynamic,
                    basis_gates,
                    qubit_ranges,
                    clbit_ranges,
                    inherited_condition=false_condition,
                )
            continue

        if stmt_type == "QuantumMeasurementStatement":
            if not allow_dynamic:
                raise ValueError("Operation 'measure' is not supported by the non-dynamic IR path.")
            if inherited_condition is not None:
                raise ValueError("Only unitary gates are supported inside conditional branches.")
            if statement.target is None:
                raise ValueError("Measure without a classical target is not supported in the dynamic path.")

            qubits = _resolve_ast_qubit_reference(statement.measure.qubit, qubit_ranges)
            clbits = _resolve_ast_clbit_reference(statement.target, clbit_ranges)
            if len(qubits) != len(clbits):
                raise ValueError("Measurement source and target widths do not match.")

            for qubit, clbit in zip(qubits, clbits):
                ir.add_operation("measure", [qubit], condition=inherited_condition, clbits=[clbit])
            continue

        if stmt_type == "QuantumReset":
            if not allow_dynamic:
                raise ValueError("Operation 'reset' is not supported by the non-dynamic IR path.")
            if inherited_condition is not None:
                raise ValueError("Only unitary gates are supported inside conditional branches.")
            qubits = _resolve_ast_qubit_reference(statement.qubits, qubit_ranges)
            for qubit in qubits:
                ir.add_operation("reset", [qubit], condition=inherited_condition)
            continue

        if stmt_type == "QuantumGate":
            op_name = statement.name.name.lower()
            if op_name in IGNORED_OPS:
                continue

            if len(statement.modifiers) > 0:
                raise ValueError(f"Gate modifiers are not supported by this IR converter: {statement}")

            qubits = []
            for qubit_ref in statement.qubits:
                resolved = _resolve_ast_qubit_reference(qubit_ref, qubit_ranges)
                if len(resolved) != 1:
                    raise ValueError(
                        "Gate arguments must reference individual qubits in this simplified dynamic path."
                    )
                qubits.append(resolved[0])

            if len(qubits) == 0:
                raise ValueError(f"Operation '{op_name}' has no qubit target and cannot be simulated in the current path.")
            if len(qubits) > 2:
                raise ValueError(
                    f"Operation '{op_name}' acts on {len(qubits)} qubits. "
                    "Current path supports only 1- and 2-qubit gates."
                )

            if not allow_dynamic and inherited_condition is not None:
                raise ValueError(
                    f"Operation '{op_name}' has a classical condition and is not supported by the non-dynamic IR path."
                )

            if basis_gates is not None and op_name not in basis_gates:
                raise ValueError(
                    f"Operation '{op_name}' is outside the selected basis_gates={basis_gates}."
                )

            params = [_resolve_ast_param_value(param_expr) for param_expr in statement.arguments]
            ir.add_operation(op_name, qubits, params=params, condition=inherited_condition)
            continue

        if stmt_type in {"SwitchStatement", "WhileLoop", "ForInLoop"}:
            raise ValueError(f"Operation '{stmt_type}' is not supported yet by this IR converter.")

        raise ValueError(f"Unsupported OpenQASM 3 statement type '{stmt_type}'.")


def _qasm3_to_circuit_ir_openqasm_ast(qasm_text, basis_gates, allow_dynamic):
    try:
        import openqasm3
    except Exception as exc:
        raise ImportError(
            "The 'openqasm3' package is required for dynamic condition fallback parsing. "
            "Install with: pip install openqasm3"
        ) from exc

    try:
        program = openqasm3.parse(qasm_text)
    except Exception as exc:
        raise ValueError(f"Unable to parse OpenQASM 3 input: {exc}") from exc

    qubit_ranges, clbit_ranges, no_qubits, no_clbits = _extract_register_maps_from_openqasm_ast(program)
    ir = CircuitIR(no_qubits, no_clbits)
    _append_openqasm_statements_to_ir(
        program.statements,
        ir,
        allow_dynamic=allow_dynamic,
        basis_gates=basis_gates,
        qubit_ranges=qubit_ranges,
        clbit_ranges=clbit_ranges,
        inherited_condition=None,
    )
    return ir


def qasm3_to_circuit_ir(qasm_text, basis_gates=None, apply_transpile=True, optimization_level=0, allow_dynamic=False):
    """
    Parses an OpenQASM 3 string and returns a CircuitIR instance.

    Args:
        qasm_text (str): Input OpenQASM 3 program text.
        basis_gates (list | None): Optional gate basis to transpile to.
        apply_transpile (bool): If True, run qiskit transpile before conversion.
        optimization_level (int): Qiskit transpiler optimization level.
        allow_dynamic (bool): If True, keep measure/reset/if instructions in the IR.

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

    qasm_text = _normalize_qasm3_bit_conditions(qasm_text)

    if apply_transpile and allow_dynamic:
        raise ValueError("apply_transpile=True is not supported yet when allow_dynamic=True.")

    try:
        qc = qasm3.loads(qasm_text)
    except Exception as exc:
        if exc.__class__.__name__ == "MissingOptionalLibraryError":
            raise ImportError(
                "Missing optional dependency 'qiskit_qasm3_import' required for OpenQASM 3 parsing. "
                "Install with: pip install qiskit_qasm3_import"
            ) from exc
        if allow_dynamic and _is_qiskit_condition_conversion_error(exc):
            return _qasm3_to_circuit_ir_openqasm_ast(
                qasm_text,
                basis_gates=basis_gates,
                allow_dynamic=allow_dynamic,
            )
        raise ValueError(f"Unable to parse OpenQASM 3 input: {exc}") from exc

    if apply_transpile:
        if basis_gates is None:
            basis_gates = supported_ir_gates()
        transpile_kwargs = {"optimization_level": optimization_level}
        if basis_gates is not None:
            transpile_kwargs["basis_gates"] = basis_gates
        qc = transpile(qc, **transpile_kwargs)

    ir = CircuitIR(qc.num_qubits, qc.num_clbits)
    qubit_map = list(range(qc.num_qubits))
    clbit_map = list(range(qc.num_clbits))

    _append_qiskit_instructions_to_ir(
        qc,
        qc.data,
        ir,
        allow_dynamic=allow_dynamic,
        basis_gates=basis_gates,
        qubit_map=qubit_map,
        clbit_map=clbit_map,
        inherited_condition=None,
    )

    return ir


def qasm3_file_to_circuit_ir(path, basis_gates=None, apply_transpile=True, optimization_level=0, allow_dynamic=False):
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
        allow_dynamic=allow_dynamic,
    )
