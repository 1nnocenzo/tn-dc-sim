# src/utils/quantum_gates.py

import numpy as np
from scipy.linalg import expm

def hadamard_gate():
    """Returns the Hadamard gate as a 2x2 tensor."""
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

def cnot_gate():
    """Returns the CNOT gate as a 2x2x2x2 tensor."""
    return np.array([[[[1, 0], [0, 0]], [[0, 0], [0, 1]]], [[[0, 0], [0, 1]], [[1, 0], [0, 0]]]])

def random_gate():
    """Returns a random 2-qubit unitary gate as a 2x2x2x2 tensor."""
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    Q, _ = np.linalg.qr(A)
    return Q.reshape(2, 2, 2, 2)

def random_isometry(rows, cols):
    """Returns a random isometry matrix reshaped as a tensor."""
    A = np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
    Q, _ = np.linalg.qr(A)
    if cols > rows:
        Q_t, _ = np.linalg.qr(A.T)
        Q = Q_t.T
    return Q

def ZZ_QAOA(gamma):
    """Returns the ZZ interaction gate for QAOA with parameter gamma."""
    A = np.diag([np.exp(-0.5j * gamma), np.exp(0.5j * gamma), np.exp(0.5j * gamma), np.exp(-0.5j * gamma)])
    return A.reshape(2, 2, 2, 2)

def X_QAOA(beta):
    """Returns the X rotation gate for QAOA with parameter beta."""
    X_gate = np.array([[0, 1], [1, 0]])
    return expm(-0.5j * beta * X_gate)


def id_gate():
    return np.identity(2)


def x_gate():
    return np.array([[0, 1], [1, 0]])


def y_gate():
    return np.array([[0, -1j], [1j, 0]])


def z_gate():
    return np.array([[1, 0], [0, -1]])


def s_gate():
    return np.array([[1, 0], [0, 1j]])


def sdg_gate():
    return np.array([[1, 0], [0, -1j]])


def t_gate():
    return np.array([[1, 0], [0, np.exp(0.25j * np.pi)]])


def tdg_gate():
    return np.array([[1, 0], [0, np.exp(-0.25j * np.pi)]])


def sx_gate():
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


def sxdg_gate():
    return np.conjugate(np.transpose(sx_gate()))


def rx_gate(theta):
    return expm(-0.5j * theta * x_gate())


def ry_gate(theta):
    return expm(-0.5j * theta * y_gate())


def rz_gate(theta):
    return expm(-0.5j * theta * z_gate())


def p_gate(lam):
    return np.array([[1, 0], [0, np.exp(1j * lam)]])


def u_gate(theta, phi, lam):
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    return np.array(
        [
            [ct, -np.exp(1j * lam) * st],
            [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
        ]
    )


def cz_gate():
    return np.diag([1, 1, 1, -1]).reshape(2, 2, 2, 2)


def swap_gate():
    A = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    return A.reshape(2, 2, 2, 2)


def RZZ_gate(theta):
    return np.diag(
        [
            np.exp(-0.5j * theta),
            np.exp(0.5j * theta),
            np.exp(0.5j * theta),
            np.exp(-0.5j * theta),
        ]
    ).reshape(2, 2, 2, 2)


def supported_single_qubit_gates():
    return ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1", "u2", "u3"]


def supported_two_qubit_gates():
    return ["cx", "cz", "swap"]
    #return ["cx", "cz", "swap", "rzz"]


def supported_ir_gates():
    return supported_single_qubit_gates() + supported_two_qubit_gates()


def _single_qubit_matrix_to_tensor(U):
    """
    Converts a 1-qubit operator matrix U[out, in] to tensor T[in, out].
    """
    assert U.shape == (2, 2), "Single-qubit operator must be 2x2"
    return np.transpose(U, (1, 0))


def _two_qubit_matrix_to_tensor(U):
    """
    Converts a 2-qubit operator matrix U[out1,out2, in1,in2] to tensor T[in1,in2,out1,out2].
    """
    assert U.shape == (4, 4), "Two-qubit operator must be 4x4"
    return np.transpose(U.reshape(2, 2, 2, 2), (2, 3, 0, 1))


def single_qubit_gate_from_name(name, params=None):
    if params is None:
        params = []
    gate_name = name.lower()

    if gate_name == "id":
        U = id_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "x":
        U = x_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "y":
        U = y_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "z":
        U = z_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "h":
        U = hadamard_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "s":
        U = s_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "sdg":
        U = sdg_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "t":
        U = t_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "tdg":
        U = tdg_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "sx":
        U = sx_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "sxdg":
        U = sxdg_gate()
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "rx":
        assert len(params) == 1, "rx expects one parameter"
        U = rx_gate(params[0])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "ry":
        assert len(params) == 1, "ry expects one parameter"
        U = ry_gate(params[0])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "rz":
        assert len(params) == 1, "rz expects one parameter"
        U = rz_gate(params[0])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "p" or gate_name == "u1":
        assert len(params) == 1, f"{gate_name} expects one parameter"
        U = p_gate(params[0])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "u2":
        assert len(params) == 2, "u2 expects two parameters"
        U = u_gate(np.pi / 2, params[0], params[1])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "u3" or gate_name == "u":
        assert len(params) == 3, f"{gate_name} expects three parameters"
        U = u_gate(params[0], params[1], params[2])
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "__proj0__":
        U = np.array([[1, 0], [0, 0]], dtype=complex)
        return _single_qubit_matrix_to_tensor(U)
    if gate_name == "__proj1__":
        U = np.array([[0, 0], [0, 1]], dtype=complex)
        return _single_qubit_matrix_to_tensor(U)

    raise ValueError(f"Unsupported single-qubit gate '{name}'")


def two_qubit_gate_from_name(name, params=None):
    if params is None:
        params = []
    gate_name = name.lower()

    if gate_name == "cx":
        U = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        return _two_qubit_matrix_to_tensor(U)
    if gate_name == "cz":
        U = np.diag([1, 1, 1, -1]).astype(complex)
        return _two_qubit_matrix_to_tensor(U)
    if gate_name == "swap":
        U = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )
        return _two_qubit_matrix_to_tensor(U)
    if gate_name == "rzz":
        assert len(params) == 1, "rzz expects one parameter"
        theta = params[0]
        U = np.diag(
            [
                np.exp(-0.5j * theta),
                np.exp(0.5j * theta),
                np.exp(0.5j * theta),
                np.exp(-0.5j * theta),
            ]
        ).astype(complex)
        return _two_qubit_matrix_to_tensor(U)

    raise ValueError(f"Unsupported two-qubit gate '{name}'")
