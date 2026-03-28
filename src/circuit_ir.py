# src/circuit_ir.py

class CircuitOperation:
    """
    Represents one circuit instruction in the internal IR format.
    """
    def __init__(self, name, qubits, params=None, condition=None):
        self.name = name
        self.qubits = list(qubits)
        self.params = [] if params is None else list(params)
        self.condition = condition

    def __repr__(self):
        return f"CircuitOperation(name={self.name}, qubits={self.qubits}, params={self.params}, condition={self.condition})"


class CircuitIR:
    """
    Internal, backend-agnostic representation of a quantum circuit.
    """
    def __init__(self, no_qubits, no_clbits=0):
        self.no_qubits = no_qubits
        self.no_clbits = no_clbits
        self.operations = []

    def add_operation(self, name, qubits, params=None, condition=None):
        self.operations.append(CircuitOperation(name, qubits, params, condition))

    def __len__(self):
        return len(self.operations)

    def __repr__(self):
        return f"CircuitIR(no_qubits={self.no_qubits}, no_clbits={self.no_clbits}, no_operations={len(self.operations)})"
