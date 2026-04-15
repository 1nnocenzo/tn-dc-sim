# src/simulation.py

import numpy as np
import opt_einsum as oe
import time
import gc
import copy
from collections import deque

from src.network import Network
from src.circuit_ir import CircuitIR
from src.utils.TN_gen import circuit_from_edge_list, circuit_from_ir, Tree, MPS, full_network_from_edge_list
from src.utils.qasm3_to_ir import qasm3_to_circuit_ir
from src.utils.quantum_gates import (
    hadamard_gate,
    X_QAOA,
    ZZ_QAOA
)

_MEASUREMENT_PROB_EPS = 1e-12
_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize
_MPS_ZERO_BRA_PERTURB_EPS = 1e-10

def partition_into_k_parts(lst, k):
    """
    Utility to partition a list into k roughly equal-sized sublists.

    Args:
        lst (list): The list to partition.
        k (int): The number of partitions.

    Returns:
        list: A list of k sublists.
    """
    avg = len(lst) // k
    remainder = len(lst) % k
    result = []
    start = 0
    for i in range(k):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

def find_path(network,start,target):
    """
    Finds the shortest path between two nodes in the network using Breadth-First Search.

    Args:
        network (Network): The tensor network object.
        start_node (str): The name of the starting node.
        target_node (str): The name of the target node.

    Returns:
        list: A list of node names representing the path, or None if no path exists.
    """

    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        if node == target:
            return path  

        for neighbor in network.nodes[node].neighbours:
            if neighbor not in path:  
                queue.append((neighbor, path + [neighbor]))

    return None 

def pre_order_traversal(network):
    """
    Performs a pre-order traversal of a tree-like tensor network, starting from the root.
    Assumes a specific structure where the first neighbor is the parent.

    Args:
        network (Network): The tensor network to traverse.

    Returns:
        list: A list of node names in pre-order.
    """
    order = []
    _,root =  list(network.nodes.items())[0]
    def dfs(node):
        if node is None:
            return
        order.append(node.name)
        if node.name[3:] == "00":
            for child in node.neighbours:
                dfs(network.nodes[child])
        else:
            for child in node.neighbours[1:]:
                dfs(network.nodes[child])
    dfs(root)
    return order

def _trailing_integer_suffix(name):
    idx = len(name) - 1
    while idx >= 0 and name[idx].isdigit():
        idx -= 1
    if idx == len(name) - 1:
        raise ValueError(f"Node name '{name}' has no trailing integer suffix.")
    return int(name[idx + 1 :])

def _mps_linear_traversal(network):
    """
    Returns MPS nodes ordered from left to right.
    Assumes node names include a trailing site index (e.g. Bra0, Bra1, ...).
    """
    return sorted(list(network.nodes.keys()), key=_trailing_integer_suffix)

def all_paths(network):
    """
    Generates all sequential paths needed for one half of a DMRG sweep.

    Args:
        network (Network): The tensor network.

    Returns:
        list: A list of paths, where each path is a list of node names.
    """
    path = []
    
    L = pre_order_traversal(network)
    # print(L)
    for i in range(len(L)-1):
        start = L[i]
        target = L[i+1]
        path.append(find_path(network,start,target))
    return path

def qr_series(network,path):
    """
    Performs a series of QR decompositions along a path to re-orthogonalize
    the tensor network, moving the center of orthogonality.

    Args:
        network (Network): The tensor network to modify.
        path (list): The path of nodes along which to perform the QR series.
    """
    tensor = []
    for i in range(len(path)-1):

        name1 = path[i]
        node1 = network.nodes[name1]
        name2 = path[i+1]
        node2 = network.nodes[name2]

        common_element = list(set(node1.subscript) & set(node2.subscript))
        common = common_element[0]

        idx1 = node1.subscript.index(common)
        idx2 = node2.subscript.index(common)

        A = node1.tensor
        A = np.moveaxis(A,idx1,-1)
        shape = A.shape
        A = A.reshape(-1,A.shape[-1])

        Q,R = np.linalg.qr(A, mode="reduced")

        # Keep the original bond dimension even when QR is rank-reduced (m < n).
        m = A.shape[0]
        n = A.shape[1]
        k = Q.shape[1]
        if k != n:
            Q_full = np.zeros((m, n), dtype=Q.dtype)
            Q_full[:, :k] = Q
            Q = Q_full
            R_full = np.zeros((n, n), dtype=R.dtype)
            R_full[:k, :] = R
            R = R_full

        Q = Q.reshape(shape)
        Q = np.moveaxis(Q,-1,idx1)



        # ein_in = node1.subscript.copy()
        # ein_in[idx1] = 'a'
        # ein_str = ''.join(node1.subscript) +","+''.join(ein_in) +f"->{common}a"

        
        # I = oe.contract(ein_str, np.conjugate(Q),Q)

        # assert (np.allclose(np.identity(node1.dim[idx1]),I,1e-12)) == True, "Q not isometry!!"
        

        # A_ = oe.contract(f"{common}a," +''.join(node1.subscript)+ "->" + ''.join(ein_in),R,Q)
 
        # assert (np.allclose(A_,node1.tensor,1e-14)) == True, "R wrong!!"

        ein_out = node2.subscript.copy()
        for c in range(5):
            com =  oe.get_symbol(c) 
            if com not in ein_out:
                break
            if c==5-1:
                assert 1!=0, "Increase c in qr_series"
        ein_out[idx2] = com


        M = oe.contract(f"{com}{common}," +''.join(node2.subscript)+ "->" + ''.join(ein_out),R,node2.tensor)

        network.replace_tensor(name1,Q)
        network.replace_tensor(name2,M)

def sweep(ns,full_network,bra, network_type="tree"):
    """
    Core iterative optimization algorithm (a DMRG-like sweep). It finds the
    best tensor network approximation (`bra`) for the state produced by the `full_network`.

    Args:
        num_sweeps (int): The number of full forward and backward sweeps. Must be even.
        full_network (Network): The complete network (ket, circuit, bra) for calculating fidelity.
        bra_network (Network): The variational tensor network (`bra`) to be optimized.

    Returns:
        np.ndarray: An array containing the fidelity at each step of the sweep.
    """
    partial_fidelity = np.zeros((ns,bra.number_nodes))
    norm_floor = 1e-30
    if network_type == "mps":
        L = _mps_linear_traversal(bra)
        col_index = {node_name: idx for idx, node_name in enumerate(L)}
        for i in range(ns):
            order = L if i % 2 == 0 else L[::-1]
            for k, j in enumerate(order):
                F = full_network.contract_all_but_one(j)
                f = oe.contract("...,...->", F, np.conjugate(F))
                f_real = float(np.real(f))
                if not np.isfinite(f_real):
                    f_real = norm_floor
                denom = np.sqrt(max(f_real, norm_floor))
                A = np.conjugate(F) / denom
                bra.replace_tensor(j, A)
                full_network.replace_tensor(j, A)
                if k != len(order) - 1:
                    next_node = order[k + 1]
                    assert next_node in bra.nodes[j].neighbours, "Invalid MPS sweep order"
                    qr_series(bra, [j, next_node])
                    full_network.replace_tensor(j, bra.nodes[j].tensor)
                    full_network.replace_tensor(next_node, bra.nodes[next_node].tensor)
                partial_fidelity[i, col_index[j]] = f.real
        return partial_fidelity

    m = -1
    path = all_paths(bra)
    L = pre_order_traversal(bra)
    for i in range(ns):
        f = 0
        if i%2 == 0:
            m +=1
            counter = 1
            p = path
        else:
            m = -1
            counter = -1
            p = path[::-1]
        for k in range(len(L)):
            j = L[m]
            F = full_network.contract_all_but_one(j)
            f = oe.contract("...,...->", F, np.conjugate(F))
            f_real = float(np.real(f))
            if not np.isfinite(f_real):
                f_real = norm_floor
            denom = np.sqrt(max(f_real, norm_floor))
            A = np.conjugate(F) / denom
            bra.replace_tensor(j,A)
            full_network.replace_tensor(j,A)
            if k!=len(L)-1:
                p_ = p[k][::counter]
                assert j== p_[0], "Wrong path"
                qr_series(bra,p_)
                for names in p_:
                    full_network.replace_tensor(names,bra.nodes[names].tensor)
            partial_fidelity[i,m] = f.real
            m +=counter
    return partial_fidelity


def _ordered_open_legs_for_physical_qubits(network, no_qubits):
    """
    Reconstructs the physical open-leg order used by full_network_from_edge_list.
    """
    leaf_queue = list(network.leaf_nodes)
    open_legs_map = {name: list(network.nodes[name].open_legs) for name in leaf_queue}
    ordered_legs = []

    for _ in range(no_qubits):
        while len(leaf_queue) > 0 and len(open_legs_map[leaf_queue[0]]) == 0:
            leaf_queue.pop(0)
        assert len(leaf_queue) > 0, "Not enough open legs to map all qubits."
        leaf_name = leaf_queue[0]
        ordered_legs.append(open_legs_map[leaf_name].pop(0))
        if len(open_legs_map[leaf_name]) == 0:
            leaf_queue.pop(0)

    return ordered_legs


def _contract_network_with_output_order(network, output_legs):
    einsum_str_parts = []
    tensor_list = []

    for node in network.nodes.values():
        einsum_str_parts.append("".join(node.subscript))
        tensor_list.append(node.tensor)

    einsum_str = f"{','.join(einsum_str_parts)}->{''.join(output_legs)}"
    return oe.contract(einsum_str, *tensor_list, optimize="auto", memory_limit=16e9)


def _state_vector_from_bra(bra, no_qubits):
    """
    Returns the ket state vector approximated by the final bra network.
    """
    output_legs = _ordered_open_legs_for_physical_qubits(bra, no_qubits)
    bra_tensor = _contract_network_with_output_order(bra, output_legs)
    ket_tensor = np.conjugate(bra_tensor)
    return ket_tensor.reshape(-1)


def _sample_measurements_from_state(state, no_qubits, n_samples_final, seed=None, return_shots=False):
    """
    Samples computational-basis measurements from a state vector.
    """
    assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be a positive integer"

    state = np.asarray(state).reshape(-1)
    expected_dim = 2 ** no_qubits
    assert state.size == expected_dim, "State size does not match no_qubits"

    probs = np.abs(state) ** 2
    norm = np.sum(probs)
    assert norm > 0, "State has zero norm, cannot sample measurements"
    probs = probs / norm

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(expected_dim, size=n_samples_final, p=probs)

    unique_indices, unique_counts = np.unique(sampled_indices, return_counts=True)
    counts = {
        format(int(idx), f"0{no_qubits}b"): int(cnt)
        for idx, cnt in zip(unique_indices, unique_counts)
    }

    if return_shots:
        shots = [format(int(idx), f"0{no_qubits}b") for idx in sampled_indices]
        return counts, shots
    return counts, None


def _apply_mps_bra_symmetry_breaking_perturbation(bra):
    # A strictly-product bra can trap one-site MPS sweeps in symmetry-protected
    # fixed points (e.g., GHZ plateau at fidelity 0.5). A tiny deterministic
    # perturbation breaks that symmetry while remaining numerically negligible.
    for node_name in bra.nodes:
        tensor = bra.nodes[node_name].tensor.astype(complex, copy=True)
        perturb = (np.arange(tensor.size, dtype=float).reshape(tensor.shape) + 1.0)
        tensor = tensor + _MPS_ZERO_BRA_PERTURB_EPS * (perturb + 1j * perturb)
        norm = float(np.linalg.norm(tensor))
        if norm > 0.0:
            tensor = tensor / norm
        bra.replace_tensor(node_name, tensor)


def _initialize_ket_bra(network_type, no_qubits, D, network_structure, bra_initial="random"):
    if network_type == "tree":
        assert no_qubits == network_structure[-1], "Number of qubits mismatch"
        ket = Tree(D, "Ket", 0, "zero", network_structure)
        bra = Tree(D, "Bra", ket.rank_all + 1, bra_initial, network_structure)
    elif network_type == "mps":
        assert no_qubits == len(D) + 1, "Number of qubits mismatch"
        ket = MPS(D, "Ket", 0, "zero")
        bra = MPS(D, "Bra", ket.rank_all + 1, bra_initial)
        if bra_initial == "zero":
            _apply_mps_bra_symmetry_breaking_perturbation(bra)
    else:
        raise ValueError("network_type must be 'tree' or 'mps'")
    return ket, bra


def _update_ket_with_bra_conjugate(ket, bra):
    for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
        assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
        ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))


def _update_bra_with_ket_conjugate(bra, ket):
    for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
        assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
        bra.replace_tensor(bra_name, np.conjugate(ket_node.tensor))


def _evaluate_dynamic_condition(condition, classical_bits):
    if condition is None:
        return True

    condition_kind = condition.get("kind")

    if condition_kind == "literal":
        return bool(condition["value"])

    if condition_kind == "clbit_eq":
        idx = condition["clbit"]
        value = condition["value"]
        return int(classical_bits[idx]) == int(value)

    if condition_kind == "creg_eq":
        register_bits = condition["clbits"]
        target_value = int(condition["value"])
        register_value = 0
        for bit_pos, bit_index in enumerate(register_bits):
            register_value |= (int(classical_bits[bit_index]) & 1) << bit_pos
        return register_value == target_value

    if condition_kind == "not":
        return not _evaluate_dynamic_condition(condition["term"], classical_bits)

    if condition_kind == "and":
        for term in condition["terms"]:
            if not _evaluate_dynamic_condition(term, classical_bits):
                return False
        return True

    if condition_kind == "or":
        for term in condition["terms"]:
            if _evaluate_dynamic_condition(term, classical_bits):
                return True
        return False

    if condition_kind == "xor":
        if "terms" in condition:
            terms = condition["terms"]
        elif "lhs" in condition and "rhs" in condition:
            terms = [condition["lhs"], condition["rhs"]]
        else:
            raise ValueError("xor condition expects either 'terms' or both 'lhs' and 'rhs'")

        result = False
        for term in terms:
            result ^= _evaluate_dynamic_condition(term, classical_bits)
        return result

    raise ValueError(f"Unsupported condition kind '{condition_kind}'")


def _build_chunk_ir_from_operations(no_qubits, operations):
    chunk_ir = CircuitIR(no_qubits, 0)
    for op in operations:
        chunk_ir.add_operation(op.name, op.qubits, op.params, op.condition, op.clbits)
    return chunk_ir


def _apply_ir_chunk_with_dmrg(ket, bra, chunk_ir, no_qubits, no_sweeps, network_type):
    if network_type == "mps":
        _apply_mps_bra_symmetry_breaking_perturbation(bra)
    circ = circuit_from_ir(bra.rank_all + 1, chunk_ir)
    N = full_network_from_edge_list(ket, circ, bra, no_qubits)
    f = sweep(no_sweeps, N, bra, network_type=network_type)
    F_ = np.max(f[-1, :])
    del circ
    del N
    gc.collect()
    return F_


def _node_axis_for_leg(network, leg_symbol):
    for node_name, node in network.nodes.items():
        if leg_symbol in node.subscript:
            return node_name, node.subscript.index(leg_symbol)
    raise ValueError("Unable to locate target physical leg in the network.")


def _apply_single_qubit_operator_to_network(network, no_qubits, target_qubit, operator):
    ordered_legs = _ordered_open_legs_for_physical_qubits(network, no_qubits)
    target_leg = ordered_legs[target_qubit]
    node_name, axis = _node_axis_for_leg(network, target_leg)
    node = network.nodes[node_name]

    updated_tensor = np.tensordot(operator, node.tensor, axes=([1], [axis]))
    updated_tensor = np.moveaxis(updated_tensor, 0, axis)
    network.replace_tensor(node_name, updated_tensor)


def _rescale_network_state(network, scale):
    first_node_name = list(network.nodes.keys())[0]
    first_node = network.nodes[first_node_name]
    network.replace_tensor(first_node_name, first_node.tensor * scale)


def _measurement_probability_from_tn(ket, bra, no_qubits, target_qubit, outcome):
    assert outcome == 0 or outcome == 1, "outcome must be 0 or 1"

    projector_name = "__proj0__" if outcome == 0 else "__proj1__"
    proj_ir = CircuitIR(no_qubits, 0)
    proj_ir.add_operation(projector_name, [target_qubit])

    circ = circuit_from_ir(bra.rank_all + 1, proj_ir)
    N = full_network_from_edge_list(ket, circ, bra, no_qubits)
    value = N.contract_all()
    p = float(np.real(np.asarray(value).reshape(-1)[0]))
    p = min(max(p, 0.0), 1.0)

    del circ
    del N
    gc.collect()
    return p


def _sample_binary_outcome(p0, rng):
    p0 = min(max(float(p0), 0.0), 1.0)
    p1 = 1.0 - p0
    if p0 <= _MEASUREMENT_PROB_EPS:
        p0, p1 = 0.0, 1.0
    elif p1 <= _MEASUREMENT_PROB_EPS:
        p0, p1 = 1.0, 0.0
    if p0 <= 0.0:
        return 1, p1
    if p1 <= 0.0:
        return 0, p0
    outcome = 0 if rng.random() < p0 else 1
    p_outcome = p0 if outcome == 0 else p1
    return outcome, p_outcome


def _collapse_qubit_single_path(ket, bra, no_qubits, target_qubit, outcome, outcome_probability):
    projector = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex) if outcome == 0 else np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    _apply_single_qubit_operator_to_network(ket, no_qubits, target_qubit, projector)
    if outcome_probability > 0.0:
        _rescale_network_state(ket, 1.0 / np.sqrt(outcome_probability))
    _update_bra_with_ket_conjugate(bra, ket)


def _flush_dynamic_branch_unitary_buffer(branch, no_qubits, no_sweeps, network_type):
    if len(branch["unitary_buffer"]) == 0:
        return
    chunk_ir = _build_chunk_ir_from_operations(no_qubits, branch["unitary_buffer"])
    F_ = _apply_ir_chunk_with_dmrg(
        branch["ket"], branch["bra"], chunk_ir, no_qubits, no_sweeps, network_type
    )
    branch["fidelity_terms"].append(F_)
    _update_ket_with_bra_conjugate(branch["ket"], branch["bra"])
    branch["unitary_buffer"] = []


def _branch_final_fidelity(branch):
    if len(branch["fidelity_terms"]) == 0:
        return 1.0
    return float(np.prod(np.array(branch["fidelity_terms"], dtype=float)))


def _prune_dynamic_branches(branches, max_branches=None, probability_cutoff=0.0):
    """
    Prunes a branch ensemble and returns (kept_branches, dropped_probability_mass).

    `probability_cutoff` and branch weights are interpreted in absolute probability
    mass (not renormalized branch probabilities).
    """
    assert probability_cutoff >= 0.0, "probability_cutoff must be >= 0"
    if max_branches is not None:
        assert max_branches >= 1, "max_branches must be >= 1 when provided"

    positive_branches = [b for b in branches if float(b["weight"]) > 0.0]
    if len(positive_branches) == 0:
        return [], 0.0

    total_mass_before = float(sum(float(b["weight"]) for b in positive_branches))
    kept = positive_branches

    if probability_cutoff > 0.0:
        cutoff_kept = [b for b in kept if float(b["weight"]) >= probability_cutoff]
        if len(cutoff_kept) > 0:
            kept = cutoff_kept
        else:
            # Keep the heaviest branch to avoid an empty ensemble.
            kept = [max(kept, key=lambda x: float(x["weight"]))]

    if max_branches is not None and len(kept) > max_branches:
        kept = sorted(kept, key=lambda x: float(x["weight"]), reverse=True)[:max_branches]

    total_mass_after = float(sum(float(b["weight"]) for b in kept))
    dropped_mass = max(0.0, total_mass_before - total_mass_after)
    return kept, dropped_mass


def _sample_measurements_from_branch_ensemble(branches, no_qubits, n_samples_final, seed=None, return_shots=False):
    """
    Samples measurements from a mixed state represented as a weighted branch ensemble.
    """
    assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be a positive integer"

    surviving_mass = float(sum(float(b["weight"]) for b in branches))
    assert surviving_mass > 0.0, "Branch ensemble has zero surviving probability mass"

    dim = 2 ** no_qubits
    probs = np.zeros(dim, dtype=float)

    for branch in branches:
        w = float(branch["weight"]) / surviving_mass
        if w <= 0.0:
            continue
        state = _state_vector_from_bra(branch["bra"], no_qubits).reshape(-1)
        assert state.size == dim, "State size does not match no_qubits"
        branch_probs = np.abs(state) ** 2
        norm = float(np.sum(branch_probs))
        assert norm > 0.0, "Encountered zero-norm branch while sampling"
        probs += w * (branch_probs / norm)

    probs_sum = float(np.sum(probs))
    assert probs_sum > 0.0, "Mixed-state probability vector has zero norm"
    probs = probs / probs_sum

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(dim, size=n_samples_final, p=probs)

    unique_indices, unique_counts = np.unique(sampled_indices, return_counts=True)
    counts = {
        format(int(idx), f"0{no_qubits}b"): int(cnt)
        for idx, cnt in zip(unique_indices, unique_counts)
    }

    if return_shots:
        shots = [format(int(idx), f"0{no_qubits}b") for idx in sampled_indices]
        return counts, shots
    return counts, None


def _single_path_memory_footprint(ket, bra):
    return int(ket.memory_footprint() + bra.memory_footprint())


def _branch_ensemble_memory_footprint(branches):
    return int(
        sum(int(b["ket"].memory_footprint()) + int(b["bra"].memory_footprint()) for b in branches)
    )


def _memory_stats_payload(final_elements, peak_elements):
    final_elements = int(final_elements)
    peak_elements = int(peak_elements)
    return {
        "final_tensor_elements": final_elements,
        "peak_tensor_elements": peak_elements,
        "final_bytes_estimate": int(final_elements * _COMPLEX128_BYTES),
        "peak_bytes_estimate": int(peak_elements * _COMPLEX128_BYTES),
    }


def DMRG_dynamic_all_branches_from_circuit_ir(
    compression_steps,
    no_sweeps,
    D,
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
):
    """
    Dynamic simulation that keeps all post-measurement/reset branches.

    Branch pruning controls:
    - max_branches: keeps at most this many highest-probability branches after each
      non-unitary expansion.
    - probability_cutoff: drops branches with absolute probability mass below cutoff.
    - max_pruning_error: optional hard cap on discarded total probability mass.
    - return_shots: returns sampled bitstrings (Qiskit-style shots list).
    - return_memory_stats: returns simulation memory stats (tensor elements + byte estimate).

    Returns:
        float | tuple:
            First output is the weighted average branch fidelity over the surviving
            ensemble (conditioned on surviving branches).
            Optional outputs include pruning error, per-branch data, and final samples.
    """
    assert no_sweeps % 2 == 0 and no_sweeps > 0, "Number of sweeps must be a positive even integer"
    assert compression_steps >= 1, "compression_steps must be >= 1"
    assert isinstance(circuit_ir, CircuitIR), "circuit_ir must be a CircuitIR instance"
    assert probability_cutoff >= 0.0, "probability_cutoff must be >= 0"
    if max_branches is not None:
        assert max_branches >= 1, "max_branches must be >= 1 when provided"
    if max_pruning_error is not None:
        assert 0.0 <= max_pruning_error <= 1.0, "max_pruning_error must be in [0, 1]"
    if return_counts or return_shots:
        assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be set when requesting final sampling"

    no_qubits = circuit_ir.no_qubits
    no_clbits = circuit_ir.no_clbits

    ket, bra = _initialize_ket_bra(
        network_type, no_qubits, D, network_structure, bra_initial="zero"
    )
    _update_ket_with_bra_conjugate(ket, bra)

    branches = [
        {
            "ket": ket,
            "bra": bra,
            "classical_bits": np.zeros(no_clbits, dtype=int),
            "weight": 1.0,
            "fidelity_terms": [],
            "unitary_buffer": [],
        }
    ]
    peak_memory_footprint = 0
    if return_memory_stats:
        peak_memory_footprint = _branch_ensemble_memory_footprint(branches)

    def enforce_pruning_error_bound(current_branches):
        if max_pruning_error is None:
            return
        surviving_mass = float(sum(float(b["weight"]) for b in current_branches))
        pruning_error = max(0.0, 1.0 - surviving_mass)
        assert pruning_error <= max_pruning_error + 1e-12, (
            f"Pruning error {pruning_error} exceeded max_pruning_error={max_pruning_error}"
        )

    for op in circuit_ir.operations:
        next_branches = []
        op_name = op.name.lower()

        for branch in branches:
            classical_bits = branch["classical_bits"]
            if not _evaluate_dynamic_condition(op.condition, classical_bits):
                next_branches.append(branch)
                continue

            if op_name == "measure":
                _flush_dynamic_branch_unitary_buffer(branch, no_qubits, no_sweeps, network_type)
                assert len(op.qubits) == 1 and len(op.clbits) == 1, "measure expects one qubit and one clbit"
                target_qubit = op.qubits[0]
                target_clbit = op.clbits[0]

                _update_ket_with_bra_conjugate(branch["ket"], branch["bra"])
                p0 = _measurement_probability_from_tn(branch["ket"], branch["bra"], no_qubits, target_qubit, 0)
                p0 = min(max(float(p0), 0.0), 1.0)
                p1 = 1.0 - p0
                if p0 <= _MEASUREMENT_PROB_EPS:
                    p0, p1 = 0.0, 1.0
                elif p1 <= _MEASUREMENT_PROB_EPS:
                    p0, p1 = 1.0, 0.0

                if p0 > 0.0:
                    ket0 = copy.deepcopy(branch["ket"])
                    bra0 = copy.deepcopy(branch["bra"])
                    _collapse_qubit_single_path(ket0, bra0, no_qubits, target_qubit, 0, p0)
                    bits0 = classical_bits.copy()
                    bits0[target_clbit] = 0
                    next_branches.append(
                        {
                            "ket": ket0,
                            "bra": bra0,
                            "classical_bits": bits0,
                            "weight": float(branch["weight"]) * p0,
                            "fidelity_terms": list(branch["fidelity_terms"]),
                            "unitary_buffer": [],
                        }
                    )

                if p1 > 0.0:
                    ket1 = copy.deepcopy(branch["ket"])
                    bra1 = copy.deepcopy(branch["bra"])
                    _collapse_qubit_single_path(ket1, bra1, no_qubits, target_qubit, 1, p1)
                    bits1 = classical_bits.copy()
                    bits1[target_clbit] = 1
                    next_branches.append(
                        {
                            "ket": ket1,
                            "bra": bra1,
                            "classical_bits": bits1,
                            "weight": float(branch["weight"]) * p1,
                            "fidelity_terms": list(branch["fidelity_terms"]),
                            "unitary_buffer": [],
                        }
                    )
                continue

            if op_name == "reset":
                _flush_dynamic_branch_unitary_buffer(branch, no_qubits, no_sweeps, network_type)
                assert len(op.qubits) == 1, "reset expects one qubit"
                target_qubit = op.qubits[0]

                _update_ket_with_bra_conjugate(branch["ket"], branch["bra"])
                p0 = _measurement_probability_from_tn(branch["ket"], branch["bra"], no_qubits, target_qubit, 0)
                p0 = min(max(float(p0), 0.0), 1.0)
                p1 = 1.0 - p0
                if p0 <= _MEASUREMENT_PROB_EPS:
                    p0, p1 = 0.0, 1.0
                elif p1 <= _MEASUREMENT_PROB_EPS:
                    p0, p1 = 1.0, 0.0

                if p0 > 0.0:
                    ket0 = copy.deepcopy(branch["ket"])
                    bra0 = copy.deepcopy(branch["bra"])
                    _collapse_qubit_single_path(ket0, bra0, no_qubits, target_qubit, 0, p0)
                    next_branches.append(
                        {
                            "ket": ket0,
                            "bra": bra0,
                            "classical_bits": classical_bits.copy(),
                            "weight": float(branch["weight"]) * p0,
                            "fidelity_terms": list(branch["fidelity_terms"]),
                            "unitary_buffer": [],
                        }
                    )

                if p1 > 0.0:
                    ket1 = copy.deepcopy(branch["ket"])
                    bra1 = copy.deepcopy(branch["bra"])
                    _collapse_qubit_single_path(ket1, bra1, no_qubits, target_qubit, 1, p1)
                    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
                    _apply_single_qubit_operator_to_network(ket1, no_qubits, target_qubit, x_gate)
                    _update_bra_with_ket_conjugate(bra1, ket1)
                    next_branches.append(
                        {
                            "ket": ket1,
                            "bra": bra1,
                            "classical_bits": classical_bits.copy(),
                            "weight": float(branch["weight"]) * p1,
                            "fidelity_terms": list(branch["fidelity_terms"]),
                            "unitary_buffer": [],
                        }
                    )
                continue

            assert len(op.qubits) > 0 and len(op.qubits) <= 2, "Dynamic path supports only 1- and 2-qubit unitary ops"
            branch["unitary_buffer"].append(op)
            if len(branch["unitary_buffer"]) >= compression_steps:
                _flush_dynamic_branch_unitary_buffer(branch, no_qubits, no_sweeps, network_type)
            next_branches.append(branch)

        branches = next_branches
        if return_memory_stats:
            peak_memory_footprint = max(
                peak_memory_footprint, _branch_ensemble_memory_footprint(branches)
            )

        if op_name == "measure" or op_name == "reset":
            branches, _ = _prune_dynamic_branches(
                branches, max_branches=max_branches, probability_cutoff=probability_cutoff
            )
            enforce_pruning_error_bound(branches)
            if return_memory_stats:
                peak_memory_footprint = max(
                    peak_memory_footprint, _branch_ensemble_memory_footprint(branches)
                )

    for branch in branches:
        _flush_dynamic_branch_unitary_buffer(branch, no_qubits, no_sweeps, network_type)

    branches, _ = _prune_dynamic_branches(
        branches, max_branches=max_branches, probability_cutoff=probability_cutoff
    )
    enforce_pruning_error_bound(branches)
    if return_memory_stats:
        peak_memory_footprint = max(
            peak_memory_footprint, _branch_ensemble_memory_footprint(branches)
        )

    surviving_mass = float(sum(float(b["weight"]) for b in branches))
    pruning_error_bound = max(0.0, 1.0 - surviving_mass)

    if len(branches) == 0:
        final_fidelity = 0.0
    elif surviving_mass <= 0.0:
        final_fidelity = 0.0
    else:
        final_fidelity = float(
            sum(
                (float(b["weight"]) / surviving_mass) * _branch_final_fidelity(b)
                for b in branches
            )
        )

    counts = None
    shots = None
    if return_counts or return_shots:
        counts, shots = _sample_measurements_from_branch_ensemble(
            branches, no_qubits, n_samples_final, seed=seed, return_shots=return_shots
        )

    memory_stats = None
    if return_memory_stats:
        final_memory_footprint = _branch_ensemble_memory_footprint(branches)
        memory_stats = _memory_stats_payload(final_memory_footprint, peak_memory_footprint)

    need_branch_payload = return_branches or return_state or return_bra or return_classical_bits
    branch_payload = []
    if need_branch_payload:
        for branch in branches:
            probability = float(branch["weight"])
            conditional_probability = (
                (probability / surviving_mass) if surviving_mass > 0.0 else 0.0
            )
            record = {
                "probability": probability,
                "conditional_probability": conditional_probability,
                "branch_fidelity": _branch_final_fidelity(branch),
            }
            if return_classical_bits:
                record["classical_bits"] = branch["classical_bits"].copy()
            if return_state:
                record["state"] = _state_vector_from_bra(branch["bra"], no_qubits)
            if return_bra:
                record["bra"] = branch["bra"]
            branch_payload.append(record)

    outputs = [final_fidelity]
    if return_pruning_error:
        outputs.append(pruning_error_bound)
    if return_branches:
        outputs.append(branch_payload)
    else:
        if return_classical_bits:
            outputs.append([entry["classical_bits"] for entry in branch_payload])
        if return_state:
            outputs.append([entry["state"] for entry in branch_payload])
        if return_bra:
            outputs.append([entry["bra"] for entry in branch_payload])
    if return_counts:
        outputs.append(counts)
    if return_shots:
        outputs.append(shots)
    if return_memory_stats:
        outputs.append(memory_stats)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def DMRG_dynamic_all_branches_from_qasm3(
    compression_steps,
    no_sweeps,
    D,
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
):
    """
    Parses dynamic OpenQASM 3 into CircuitIR and runs all-branches dynamic TTN/MPS simulation.
    """
    circuit_ir = qasm3_to_circuit_ir(
        qasm_text,
        basis_gates=basis_gates,
        apply_transpile=apply_transpile,
        optimization_level=optimization_level,
        allow_dynamic=True,
    )
    return DMRG_dynamic_all_branches_from_circuit_ir(
        compression_steps=compression_steps,
        no_sweeps=no_sweeps,
        D=D,
        network_structure=network_structure,
        circuit_ir=circuit_ir,
        network_type=network_type,
        max_branches=max_branches,
        probability_cutoff=probability_cutoff,
        max_pruning_error=max_pruning_error,
        return_branches=return_branches,
        return_state=return_state,
        return_bra=return_bra,
        return_classical_bits=return_classical_bits,
        return_pruning_error=return_pruning_error,
        n_samples_final=n_samples_final,
        seed=seed,
        return_counts=return_counts,
        return_shots=return_shots,
        return_memory_stats=return_memory_stats,
    )


def DMRG_dynamic_single_path_from_circuit_ir(compression_steps, no_sweeps, D, network_structure, circuit_ir, network_type, seed=None, return_state=False, return_bra=False, return_classical_bits=False, return_branch_probability=False, n_samples_final=None, return_counts=False, return_shots=False, return_memory_stats=False):
    """
    Single-path dynamic simulation using TTN/MPS + DMRG chunks.

    `compression_steps` is interpreted as the maximum number of unitary
    operations accumulated in a chunk before running a DMRG sweep.
    Non-unitary ops (measure/reset) always flush the current unitary chunk.
    `return_shots` returns sampled bitstrings, while
    `return_memory_stats` returns tensor-memory stats.
    """
    assert no_sweeps % 2 == 0 and no_sweeps > 0, "Number of sweeps must be a positive even integer"
    assert compression_steps >= 1, "compression_steps must be >= 1"
    assert isinstance(circuit_ir, CircuitIR), "circuit_ir must be a CircuitIR instance"
    if return_counts or return_shots:
        assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be set when requesting final sampling"

    no_qubits = circuit_ir.no_qubits
    no_clbits = circuit_ir.no_clbits

    ket, bra = _initialize_ket_bra(network_type, no_qubits, D, network_structure, bra_initial="zero")
    _update_ket_with_bra_conjugate(ket, bra)

    classical_bits = np.zeros(no_clbits, dtype=int)
    rng = np.random.default_rng(seed)
    branch_probability = 1.0
    fidelity_terms = []
    unitary_buffer = []
    peak_memory_footprint = 0
    if return_memory_stats:
        peak_memory_footprint = _single_path_memory_footprint(ket, bra)

    def flush_unitary_chunk():
        nonlocal unitary_buffer
        if len(unitary_buffer) == 0:
            return
        chunk_ir = _build_chunk_ir_from_operations(no_qubits, unitary_buffer)
        F_ = _apply_ir_chunk_with_dmrg(ket, bra, chunk_ir, no_qubits, no_sweeps, network_type)
        fidelity_terms.append(F_)
        _update_ket_with_bra_conjugate(ket, bra)
        unitary_buffer = []

    for op in circuit_ir.operations:
        if not _evaluate_dynamic_condition(op.condition, classical_bits):
            continue

        op_name = op.name.lower()

        if op_name == "measure":
            flush_unitary_chunk()
            assert len(op.qubits) == 1 and len(op.clbits) == 1, "measure expects one qubit and one clbit"
            target_qubit = op.qubits[0]
            target_clbit = op.clbits[0]

            _update_ket_with_bra_conjugate(ket, bra)
            p0 = _measurement_probability_from_tn(ket, bra, no_qubits, target_qubit, 0)
            outcome, p_outcome = _sample_binary_outcome(p0, rng)
            branch_probability *= p_outcome

            _collapse_qubit_single_path(ket, bra, no_qubits, target_qubit, outcome, p_outcome)
            classical_bits[target_clbit] = outcome
            if return_memory_stats:
                peak_memory_footprint = max(
                    peak_memory_footprint, _single_path_memory_footprint(ket, bra)
                )
            continue

        if op_name == "reset":
            flush_unitary_chunk()
            assert len(op.qubits) == 1, "reset expects one qubit"
            target_qubit = op.qubits[0]

            _update_ket_with_bra_conjugate(ket, bra)
            p0 = _measurement_probability_from_tn(ket, bra, no_qubits, target_qubit, 0)
            outcome, p_outcome = _sample_binary_outcome(p0, rng)
            branch_probability *= p_outcome

            _collapse_qubit_single_path(ket, bra, no_qubits, target_qubit, outcome, p_outcome)

            if outcome == 1:
                x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
                _apply_single_qubit_operator_to_network(ket, no_qubits, target_qubit, x_gate)
                _update_bra_with_ket_conjugate(bra, ket)
            if return_memory_stats:
                peak_memory_footprint = max(
                    peak_memory_footprint, _single_path_memory_footprint(ket, bra)
                )
            continue

        assert len(op.qubits) > 0 and len(op.qubits) <= 2, "Dynamic path supports only 1- and 2-qubit unitary ops"
        unitary_buffer.append(op)
        if len(unitary_buffer) >= compression_steps:
            flush_unitary_chunk()
            if return_memory_stats:
                peak_memory_footprint = max(
                    peak_memory_footprint, _single_path_memory_footprint(ket, bra)
                )

    flush_unitary_chunk()

    if len(fidelity_terms) == 0:
        final_fidelity = 1.0
    else:
        final_fidelity = np.cumprod(np.array(fidelity_terms))[-1]

    need_state = return_state or return_counts or return_shots
    final_state = None
    if need_state:
        final_state = _state_vector_from_bra(bra, no_qubits)

    counts = None
    shots = None
    if return_counts or return_shots:
        counts, shots = _sample_measurements_from_state(
            final_state, no_qubits, n_samples_final, seed=seed, return_shots=return_shots
        )

    memory_stats = None
    if return_memory_stats:
        final_memory_footprint = _single_path_memory_footprint(ket, bra)
        peak_memory_footprint = max(peak_memory_footprint, final_memory_footprint)
        memory_stats = _memory_stats_payload(final_memory_footprint, peak_memory_footprint)

    outputs = [final_fidelity]
    if return_branch_probability:
        outputs.append(branch_probability)
    if return_classical_bits:
        outputs.append(classical_bits.copy())
    if return_state:
        outputs.append(final_state)
    if return_bra:
        outputs.append(bra)
    if return_counts:
        outputs.append(counts)
    if return_shots:
        outputs.append(shots)
    if return_memory_stats:
        outputs.append(memory_stats)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def DMRG_dynamic_single_path_from_qasm3(compression_steps, no_sweeps, D, network_structure, qasm_text, network_type, seed=None, return_state=False, return_bra=False, return_classical_bits=False, return_branch_probability=False, basis_gates=None, apply_transpile=False, optimization_level=0, n_samples_final=None, return_counts=False, return_shots=False, return_memory_stats=False):
    """
    Parses dynamic OpenQASM 3 into CircuitIR and runs single-path dynamic TTN/MPS simulation.
    """
    circuit_ir = qasm3_to_circuit_ir(
        qasm_text,
        basis_gates=basis_gates,
        apply_transpile=apply_transpile,
        optimization_level=optimization_level,
        allow_dynamic=True,
    )
    return DMRG_dynamic_single_path_from_circuit_ir(
        compression_steps=compression_steps,
        no_sweeps=no_sweeps,
        D=D,
        network_structure=network_structure,
        circuit_ir=circuit_ir,
        network_type=network_type,
        seed=seed,
        return_state=return_state,
        return_bra=return_bra,
        return_classical_bits=return_classical_bits,
        return_branch_probability=return_branch_probability,
        n_samples_final=n_samples_final,
        return_counts=return_counts,
        return_shots=return_shots,
        return_memory_stats=return_memory_stats,
    )


def DMRG_from_circuit_ir(compression_steps, no_sweeps, D, network_structure, circuit_ir, network_type, return_state=False, return_bra=False, n_samples_final=None, seed=None, return_counts=False, return_shots=False):
    """
    Runs DMRG compression for a generic non-dynamic CircuitIR.

    Args:
        compression_steps (int): Maximum number of unitary operations per chunk.
        no_sweeps (int): Number of sweeps for each chunk.
        D (list): Bond dimensions for TTN/MPS.
        network_structure (list): TTN hierarchy, ignored for MPS except for validation.
        circuit_ir (CircuitIR): Input circuit in internal representation.
        network_type (str): 'tree' or 'mps'.

    Returns:
        float | tuple: Final cumulative fidelity, optionally with final state vector and/or bra network.
    """
    assert no_sweeps % 2 == 0, "Number of sweeps must be even"
    assert compression_steps >= 1, "compression_steps must be >= 1"
    assert isinstance(circuit_ir, CircuitIR), "circuit_ir must be a CircuitIR instance"
    assert circuit_ir.no_clbits == 0, "Non-dynamic path does not support classical bits"
    if return_counts or return_shots:
        assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be set when requesting final sampling"

    no_qubits = circuit_ir.no_qubits
    ket, bra = _initialize_ket_bra(network_type, no_qubits, D, network_structure, bra_initial="zero")
    _update_ket_with_bra_conjugate(ket, bra)

    fidelity_terms = []
    unitary_buffer = []

    def flush_unitary_chunk():
        nonlocal unitary_buffer
        if len(unitary_buffer) == 0:
            return
        chunk_ir = _build_chunk_ir_from_operations(no_qubits, unitary_buffer)
        F_ = _apply_ir_chunk_with_dmrg(ket, bra, chunk_ir, no_qubits, no_sweeps, network_type)
        fidelity_terms.append(F_)
        _update_ket_with_bra_conjugate(ket, bra)
        unitary_buffer = []

    for op in circuit_ir.operations:
        assert op.condition is None, "Non-dynamic path does not support classically conditioned operations"
        assert len(op.clbits) == 0, "Non-dynamic path does not support classical-bit operands in operations"
        unitary_buffer.append(op)
        if len(unitary_buffer) >= compression_steps:
            flush_unitary_chunk()

    flush_unitary_chunk()

    if len(fidelity_terms) == 0:
        final_fidelity = 1.0
    else:
        final_fidelity = np.cumprod(np.array(fidelity_terms))[-1]

    need_state = return_state or return_counts or return_shots
    final_state = None
    if need_state:
        final_state = _state_vector_from_bra(bra, no_qubits)

    counts = None
    shots = None
    if return_counts or return_shots:
        counts, shots = _sample_measurements_from_state(
            final_state, no_qubits, n_samples_final, seed=seed, return_shots=return_shots
        )

    outputs = [final_fidelity]
    if return_state:
        outputs.append(final_state)
    if return_bra:
        outputs.append(bra)
    if return_counts:
        outputs.append(counts)
    if return_shots:
        outputs.append(shots)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def DMRG_from_qasm3(compression_steps, no_sweeps, D, network_structure, qasm_text, network_type, basis_gates=None, apply_transpile=True, optimization_level=0, return_state=False, return_bra=False, n_samples_final=None, seed=None, return_counts=False, return_shots=False):
    """
    Parses OpenQASM 3 into CircuitIR and runs DMRG_from_circuit_ir.
    """
    circuit_ir = qasm3_to_circuit_ir(
        qasm_text,
        basis_gates=basis_gates,
        apply_transpile=apply_transpile,
        optimization_level=optimization_level,
    )
    return DMRG_from_circuit_ir(
        compression_steps=compression_steps,
        no_sweeps=no_sweeps,
        D=D,
        network_structure=network_structure,
        circuit_ir=circuit_ir,
        network_type=network_type,
        return_state=return_state,
        return_bra=return_bra,
        n_samples_final=n_samples_final,
        seed=seed,
        return_counts=return_counts,
        return_shots=return_shots,
    )

def DMRG(compression_steps,depth, no_sweeps,no_qubits,D,network_structure,full_edge_list, network_type,circuit_type,run, return_state=False, return_bra=False, n_samples_final=None, seed=None, return_counts=False, return_shots=False):
    """
    Orchestrates a full quantum circuit simulation using partitioned
    circuits and DMRG-style compression.

    Args:
        compression_steps (int): How many partitions to divide the circuit into at each depth.
        depth (int): The number of layers in the quantum algorithm (e.g., p in QAOA).
        no_sweeps (int): The number of sweeps for the `sweep` optimizer.
        no_qubits (int): The total number of qubits.
        D (list): A list of bond dimensions for the tensor network.
        network_structure (list): The hierarchical structure for a Tree Tensor Network.
        full_edge_list (list): The list of all edges defining two-qubit gate locations.
        network_type (str): The type of tensor network ('tree' or 'mps').
        circuit_type (str): The type of circuit ('qaoa' or 'random').
        run (int): The current run number, used for seeding random number generators.

    Returns:
        float: The final cumulative fidelity of the simulation.
    """
    assert no_sweeps%2 ==0, "Number of sweeps must be even"
    if return_counts or return_shots:
        assert n_samples_final is not None and n_samples_final > 0, "n_samples_final must be set when requesting final sampling"

    print(f" Total Compression Step = {compression_steps*depth} and network = {network_type} and nodes: {network_structure}" )
    
    Fid = []
    partial_edge_list = partition_into_k_parts(full_edge_list, compression_steps)
 
    for d in range(depth):
        assert circuit_type == "qaoa" or circuit_type == "random", "Wrong circuit type"
        single_qubit_ket = np.identity(2)
        if circuit_type == "random":
            two_qubit_gate = None
            single_qubit_bra = np.identity(2)
        else: 
            
            if d==0:
                single_qubit_ket =  hadamard_gate()
            np.random.seed(d + run*10)
            two_qubit_gate = ZZ_QAOA((np.random.uniform(0, 2*np.pi)))
            if compression_steps == 1:
                np.random.seed(d+1+run*10)
                single_qubit_bra = X_QAOA((np.random.uniform(0, np.pi)))
            else:
                single_qubit_bra =np.identity(2)
                
        
        if network_type == "tree":
            assert no_qubits == network_structure[-1], "Number of qubits mismatch"
            if d==0:
                ket = Tree(D,"Ket",0,"zero",network_structure)
                bra = Tree(D,"Bra",ket.rank_all+1,"random",network_structure)
            else:
                for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                    assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                    ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))
            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[0],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)
            print("Number of 2 qubit gates:",len(partial_edge_list[0]))
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            print("Memory = ", bra.memory_footprint())
            start_time = time.time()
            f = sweep(no_sweeps,N,bra, network_type=network_type)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

        elif network_type == 'mps':

            assert no_qubits == len(D)+1, "Number of qubits mismatch"
            if d == 0:
                ket = MPS(D,"Ket",0,"zero")
                bra = MPS(D,"Bra",ket.rank_all+1,"random")
            else:
                for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                    assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                    ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))
            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[0],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)

            print("Number of 2 qubit gates:",len(partial_edge_list[0]))
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            print("Memory = ", bra.memory_footprint())
            start_time = time.time()
            f = sweep(no_sweeps,N,bra, network_type=network_type)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

        print(f"Compression step = {1} and Depth = {d+1}; Fidelity = {np.cumprod(np.array(Fid))[-1]}; sweep time = {time.time() - start_time}")

        for i in range(compression_steps-1):
            if circuit_type == "random":
                two_qubit_gate = None
                single_qubit_ket = np.identity(2)
                single_qubit_bra = np.identity(2)
            else:
                single_qubit_ket =  np.identity(2)
                if i == compression_steps-2:
                    np.random.seed(d+1+run*10)
                    single_qubit_bra = X_QAOA((np.random.uniform(0, np.pi)))
                else:
                    single_qubit_bra =np.identity(2)
            del circ
            del N
            gc.collect()
            
            for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))

            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[i+1],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            
            print("Number of 2 qubit gates:",len(partial_edge_list[i+1]))

            start_time = time.time()
            f = sweep(no_sweeps,N,bra, network_type=network_type)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

            print(f"Compression step = {i+2}  and Depth = {d+1}; Fidelity = {np.cumprod(np.array(Fid))[-1]}; sweep time = {time.time() - start_time}")
        
        del circ
        del N
        gc.collect()


    final_fidelity = np.cumprod(np.array(Fid))[-1]

    need_state = return_state or return_counts or return_shots
    final_state = None
    if need_state:
        final_state = _state_vector_from_bra(bra, no_qubits)

    counts = None
    shots = None
    if return_counts or return_shots:
        counts, shots = _sample_measurements_from_state(
            final_state, no_qubits, n_samples_final, seed=seed, return_shots=return_shots
        )

    outputs = [final_fidelity]
    if return_state:
        outputs.append(final_state)
    if return_bra:
        outputs.append(bra)
    if return_counts:
        outputs.append(counts)
    if return_shots:
        outputs.append(shots)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)
