"""
Quantum routing engine for Quantum Relief Router.

This module converts a set of candidate emergency routes into a
binary optimization problem and solves it with QAOA.

The optimizer selects one route while penalizing selections that
violate the one-route constraint.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

try:
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator

    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config import CONFIG


def _validate_dependencies() -> None:
    """
    Verify that the required quantum dependencies exist.
    """

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit is not installed. "
            "Install qiskit before using quantum routing."
        )

    if not QISKIT_AER_AVAILABLE:
        raise RuntimeError(
            "Qiskit Aer is not installed. "
            "Install qiskit-aer before using quantum routing."
        )

    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is not installed. "
            "Install scipy before using quantum routing."
        )


def _normalize_route_costs(
    routes: list[dict[str, Any]],
) -> np.ndarray:
    """
    Normalize route distances to values between 0 and 1.

    Lower distance produces a lower optimization cost.
    """

    distances = np.array(
        [
            float(route["distance"])
            for route in routes
        ],
        dtype=float,
    )

    minimum = float(np.min(distances))
    maximum = float(np.max(distances))

    if math.isclose(minimum, maximum):
        return np.zeros(len(distances))

    return (distances - minimum) / (
        maximum - minimum
    )


def build_qubo(
    routes: list[dict[str, Any]],
    penalty: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Build the QUBO matrix for route selection.

    Objective:

        minimize
            route_cost
            + penalty * (sum(x_i) - 1)^2

    This forces the optimizer toward selecting exactly one route.

    Returns:
        QUBO matrix and penalty value.
    """

    if not routes:
        raise ValueError(
            "At least one candidate route is required."
        )

    if penalty is None:
        penalty = CONFIG.quantum.constraint_penalty

    num_routes = len(routes)

    normalized_costs = _normalize_route_costs(
        routes
    )

    qubo = np.zeros(
        (num_routes, num_routes),
        dtype=float,
    )

    # Linear route costs.
    for i in range(num_routes):
        qubo[i, i] += normalized_costs[i]

    # Expand:
    #
    # penalty * (sum(x_i) - 1)^2
    #
    # Since x_i^2 = x_i for binary variables:
    #
    # penalty * [
    #     sum(x_i)
    #     + 2 sum(x_i x_j)
    #     - 2 sum(x_i)
    #     + 1
    # ]
    #
    # Therefore:
    #
    # diagonal contribution = -penalty
    # pair contribution      = 2 * penalty

    for i in range(num_routes):
        qubo[i, i] -= penalty

    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            qubo[i, j] += 2.0 * penalty

    return qubo, penalty


def qubo_energy(
    bitstring: np.ndarray,
    qubo: np.ndarray,
) -> float:
    """
    Calculate the QUBO energy for a binary state.
    """

    return float(
        bitstring @ qubo @ bitstring
    )


def enumerate_optimal_state(
    qubo: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Find the exact optimal binary state by enumeration.

    This is used as a reference solution and fallback.

    It is intentionally limited to the configured qubit count.
    """

    num_variables = qubo.shape[0]

    if num_variables > CONFIG.quantum.max_qubits:
        raise ValueError(
            f"QUBO contains {num_variables} variables, "
            f"which exceeds the configured "
            f"{CONFIG.quantum.max_qubits}-qubit limit."
        )

    best_state = None
    best_energy = float("inf")

    for state_number in range(
        2**num_variables
    ):
        bits = np.array(
            [
                (state_number >> i) & 1
                for i in range(num_variables)
            ],
            dtype=float,
        )

        energy = qubo_energy(
            bits,
            qubo,
        )

        if energy < best_energy:
            best_energy = energy
            best_state = bits

    if best_state is None:
        raise RuntimeError(
            "Unable to determine an optimal state."
        )

    return best_state, best_energy


def _qubo_to_ising(
    qubo: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Convert a QUBO matrix into Ising coefficients.

    Binary variables:

        x_i = (1 - z_i) / 2

    Returns:
        Linear Z coefficients and constant offset.
    """

    n = qubo.shape[0]

    linear = np.zeros(n)
    constant = 0.0

    # Diagonal terms.
    for i in range(n):
        qii = qubo[i, i]

        constant += qii / 2.0
        linear[i] -= qii / 2.0

    # Upper-triangular quadratic terms.
    for i in range(n):
        for j in range(i + 1, n):
            qij = qubo[i, j]

            constant += qij / 4.0

            linear[i] -= qij / 4.0
            linear[j] -= qij / 4.0

    return linear, constant


def _create_qaoa_circuit(
    qubo: np.ndarray,
    reps: int,
) -> QAOAAnsatz:
    """
    Construct a QAOA ansatz from the QUBO matrix.
    """

    from qiskit.quantum_info import SparsePauliOp

    linear, _ = _qubo_to_ising(qubo)

    num_qubits = qubo.shape[0]

    paulis = []
    coefficients = []

    for i in range(num_qubits):
        label = ["I"] * num_qubits

        label[num_qubits - i - 1] = "Z"

        paulis.append(
            "".join(label)
        )

        coefficients.append(
            float(linear[i])
        )

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):

            coefficient = (
                qubo[i, j] / 4.0
            )

            if math.isclose(
                coefficient,
                0.0,
            ):
                continue

            label = ["I"] * num_qubits

            label[num_qubits - i - 1] = "Z"
            label[num_qubits - j - 1] = "Z"

            paulis.append(
                "".join(label)
            )

            coefficients.append(
                float(coefficient)
            )

    cost_operator = SparsePauliOp.from_list(
        [
            (pauli, coefficient)
            for pauli, coefficient in zip(
                paulis,
                coefficients,
            )
        ]
    )

    return QAOAAnsatz(
        cost_operator,
        reps=reps,
    )


def _sample_qaoa_state(
    circuit,
    parameters: np.ndarray,
    shots: int,
) -> dict[str, int]:
    """
    Execute a QAOA circuit and return measurement counts.
    """

    simulator = AerSimulator()

    bound_circuit = circuit.assign_parameters(
        parameters
    )

    bound_circuit.measure_all()

    result = simulator.run(
        bound_circuit,
        shots=shots,
    ).result()

    return dict(
        result.get_counts()
    )


def _bitstring_to_array(
    bitstring: str,
    num_qubits: int,
) -> np.ndarray:
    """
    Convert a Qiskit measurement bitstring into
    a binary NumPy array.
    """

    cleaned = bitstring.replace(
        " ",
        "",
    )

    cleaned = cleaned.zfill(
        num_qubits
    )

    # Qiskit strings are displayed in reverse
    # qubit order relative to the variable index.
    cleaned = cleaned[::-1]

    return np.array(
        [
            int(bit)
            for bit in cleaned
        ],
        dtype=float,
    )


def _select_best_measurement(
    counts: dict[str, int],
    qubo: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Select the lowest-energy state measured by QAOA.

    Returns:
        Best binary state
        Best energy
        Probability of that state
    """

    shots = sum(counts.values())

    best_state = None
    best_energy = float("inf")
    best_count = 0

    for bitstring, count in counts.items():

        state = _bitstring_to_array(
            bitstring,
            qubo.shape[0],
        )

        energy = qubo_energy(
            state,
            qubo,
        )

        if energy < best_energy:
            best_energy = energy
            best_state = state
            best_count = count

    if best_state is None:
        raise RuntimeError(
            "QAOA produced no measurable states."
        )

    probability = (
        best_count / shots
        if shots > 0
        else 0.0
    )

    return (
        best_state,
        best_energy,
        probability,
    )


def solve_with_qaoa(
    routes: list[dict[str, Any]],
    shots: int | None = None,
    reps: int | None = None,
) -> dict[str, Any]:
    """
    Solve route selection with QAOA.

    Returns a dictionary containing:
    - selected route
    - probabilities
    - QAOA energy
    - runtime
    - qubit count
    - shot count
    - measurement counts
    - classical reference solution
    """

    _validate_dependencies()

    if not routes:
        raise ValueError(
            "No candidate routes were supplied."
        )

    if len(routes) > CONFIG.quantum.max_qubits:
        routes = routes[
            : CONFIG.quantum.max_qubits
        ]

    shots = (
        shots
        if shots is not None
        else CONFIG.quantum.default_shots
    )

    reps = (
        reps
        if reps is not None
        else CONFIG.quantum.qaoa_reps
    )

    shots = max(
        CONFIG.quantum.min_shots,
        min(
            shots,
            CONFIG.quantum.max_shots,
        ),
    )

    start_time = time.perf_counter()

    qubo, penalty = build_qubo(
        routes
    )

    num_qubits = len(routes)

    circuit = _create_qaoa_circuit(
        qubo,
        reps,
    )

    parameter_count = circuit.num_parameters

    if parameter_count == 0:
        parameters = np.array([])
    else:
        parameters = np.full(
            parameter_count,
            0.5,
            dtype=float,
        )

    def objective(
        values: np.ndarray,
    ) -> float:
        """
        Evaluate QAOA parameters using a statevector.
        """

        bound = circuit.assign_parameters(
            values
        )

        statevector = Statevector.from_instruction(
            bound
        )

        probabilities = (
            statevector.probabilities()
        )

        expectation = 0.0

        for state_number, probability in enumerate(
            probabilities
        ):
            if probability <= 1e-12:
                continue

            bits = np.array(
                [
                    (state_number >> i) & 1
                    for i in range(num_qubits)
                ],
                dtype=float,
            )

            expectation += (
                probability
                * qubo_energy(
                    bits,
                    qubo,
                )
            )

        return float(expectation)

    optimization_result = minimize(
        objective,
        parameters,
        method="COBYLA",
        options={
            "maxiter": CONFIG.quantum.maxiter,
        },
    )

    optimized_parameters = (
        optimization_result.x
    )

    counts = _sample_qaoa_state(
        circuit,
        optimized_parameters,
        shots,
    )

    best_state, best_energy, best_probability = (
        _select_best_measurement(
            counts,
            qubo,
        )
    )

    selected_indices = [
        index
        for index, value in enumerate(
            best_state
        )
        if value > 0.5
    ]

    if not selected_indices:
        # Fall back to the lowest-distance route.
        selected_index = int(
            np.argmin(
                [
                    route["distance"]
                    for route in routes
                ]
            )
        )
    else:
        selected_index = selected_indices[0]

    probabilities = {}

    total_shots = sum(
        counts.values()
    )

    for index, route in enumerate(routes):

        probability = 0.0

        for bitstring, count in counts.items():

            state = _bitstring_to_array(
                bitstring,
                num_qubits,
            )

            if (
                index < len(state)
                and state[index] > 0.5
            ):
                probability += count

        if total_shots > 0:
            probability /= total_shots

        probabilities[index] = probability

    exact_state, exact_energy = (
        enumerate_optimal_state(qubo)
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "selected_route_index": selected_index,
        "selected_route": routes[selected_index],
        "probabilities": probabilities,
        "counts": counts,
        "qubo": qubo,
        "penalty": penalty,
        "qaoa_energy": best_energy,
        "exact_energy": exact_energy,
        "exact_state": exact_state,
        "best_probability": best_probability,
        "runtime": elapsed,
        "qubits": num_qubits,
        "shots": shots,
        "reps": reps,
        "optimizer_success": bool(
            optimization_result.success
        ),
        "optimizer_message": str(
            optimization_result.message
        ),
    }


def solve_route_selection(
    routes: list[dict[str, Any]],
    shots: int | None = None,
    reps: int | None = None,
) -> dict[str, Any]:
    """
    Public entry point for quantum route selection.

    If the quantum stack is unavailable, the function returns
    a deterministic fallback result based on the shortest route.
    """

    if not routes:
        raise ValueError(
            "No candidate routes available."
        )

    if not QISKIT_AVAILABLE or not QISKIT_AER_AVAILABLE:
        best_index = int(
            np.argmin(
                [
                    route["distance"]
                    for route in routes
                ]
            )
        )

        return {
            "selected_route_index": best_index,
            "selected_route": routes[best_index],
            "probabilities": {
                index: (
                    1.0
                    if index == best_index
                    else 0.0
                )
                for index in range(
                    len(routes)
                )
            },
            "counts": {},
            "qubo": None,
            "penalty": None,
            "qaoa_energy": None,
            "exact_energy": None,
            "exact_state": None,
            "best_probability": 1.0,
            "runtime": 0.0,
            "qubits": min(
                len(routes),
                CONFIG.quantum.max_qubits,
            ),
            "shots": shots
            or CONFIG.quantum.default_shots,
            "reps": reps
            or CONFIG.quantum.qaoa_reps,
            "optimizer_success": False,
            "optimizer_message": (
                "Qiskit unavailable. "
                "Used classical fallback."
            ),
            "quantum_available": False,
        }

    result = solve_with_qaoa(
        routes=routes,
        shots=shots,
        reps=reps,
    )

    result["quantum_available"] = True

    return result
