"""
Telemetry and performance tracking for Quantum Relief Router.

Tracks:
- Routing runtime
- QAOA runtime
- Classical runtime
- Graph size
- Qubit count
- Shot count
- Route distance
- Emergency scenario information

No personally identifying information is collected.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Iterator

import pandas as pd

from config import CONFIG


@dataclass
class TelemetryRecord:
    """
    Single application performance record.
    """

    operation: str
    runtime_seconds: float

    nodes: int = 0
    edges: int = 0

    qubits: int = 0
    shots: int = 0
    qaoa_reps: int = 0

    route_distance_m: float = 0.0

    blocked_roads: int = 0

    scenario: str = ""

    success: bool = True

    details: str = ""


class Telemetry:
    """
    Stores telemetry records for one application session.
    """

    def __init__(
        self,
        enabled: bool | None = None,
    ) -> None:

        if enabled is None:
            enabled = CONFIG.telemetry.enabled

        self.enabled = enabled

        self.records: list[
            TelemetryRecord
        ] = []

    def record(
        self,
        operation: str,
        runtime_seconds: float,
        nodes: int = 0,
        edges: int = 0,
        qubits: int = 0,
        shots: int = 0,
        qaoa_reps: int = 0,
        route_distance_m: float = 0.0,
        blocked_roads: int = 0,
        scenario: str = "",
        success: bool = True,
        details: str = "",
    ) -> None:
        """
        Add a telemetry record.
        """

        if not self.enabled:
            return

        record = TelemetryRecord(
            operation=operation,
            runtime_seconds=round(
                runtime_seconds,
                CONFIG.telemetry.timing_precision,
            ),
            nodes=nodes,
            edges=edges,
            qubits=qubits,
            shots=shots,
            qaoa_reps=qaoa_reps,
            route_distance_m=route_distance_m,
            blocked_roads=blocked_roads,
            scenario=scenario,
            success=success,
            details=details,
        )

        self.records.append(
            record
        )

    @contextmanager
    def timer(
        self,
        operation: str,
        **metadata: Any,
    ) -> Iterator[dict[str, float]]:
        """
        Context manager for measuring operation runtime.

        Example:

            with telemetry.timer("QAOA") as timer:
                run_qaoa()

        The resulting runtime is stored automatically.
        """

        start = time.perf_counter()

        result = {
            "runtime_seconds": 0.0
        }

        try:
            yield result

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            result[
                "runtime_seconds"
            ] = elapsed

            self.record(
                operation=operation,
                runtime_seconds=elapsed,
                success=False,
                details=str(exc),
                **metadata,
            )

            raise

        else:

            elapsed = (
                time.perf_counter()
                - start
            )

            result[
                "runtime_seconds"
            ] = elapsed

            self.record(
                operation=operation,
                runtime_seconds=elapsed,
                success=True,
                **metadata,
            )

    def clear(self) -> None:
        """
        Delete all telemetry records.
        """

        self.records.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert telemetry records into a DataFrame.
        """

        if not self.records:
            return pd.DataFrame(
                columns=[
                    "operation",
                    "runtime_seconds",
                    "nodes",
                    "edges",
                    "qubits",
                    "shots",
                    "qaoa_reps",
                    "route_distance_m",
                    "blocked_roads",
                    "scenario",
                    "success",
                    "details",
                ]
            )

        return pd.DataFrame(
            [
                asdict(record)
                for record in self.records
            ]
        )

    def latest(
        self,
    ) -> TelemetryRecord | None:
        """
        Return the most recent telemetry record.
        """

        if not self.records:
            return None

        return self.records[-1]

    def summary(self) -> dict[str, Any]:
        """
        Return a compact telemetry summary.
        """

        if not self.records:
            return {
                "records": 0,
                "total_runtime_seconds": 0.0,
                "successful_operations": 0,
                "failed_operations": 0,
            }

        total_runtime = sum(
            record.runtime_seconds
            for record in self.records
        )

        successful = sum(
            record.success
            for record in self.records
        )

        return {
            "records": len(self.records),
            "total_runtime_seconds": round(
                total_runtime,
                CONFIG.telemetry.timing_precision,
            ),
            "successful_operations": successful,
            "failed_operations": (
                len(self.records)
                - successful
            ),
        }


def measure_runtime(
    function,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """
    Execute a function and return:

        (function_result, runtime_seconds)
    """

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    return result, runtime


def create_route_telemetry(
    operation: str,
    runtime: float,
    graph,
    route_distance: float = 0.0,
    qubits: int = 0,
    shots: int = 0,
    qaoa_reps: int = 0,
    blocked_roads: int = 0,
    scenario: str = "",
    success: bool = True,
    details: str = "",
) -> TelemetryRecord:
    """
    Create a telemetry record directly from routing data.
    """

    return TelemetryRecord(
        operation=operation,
        runtime_seconds=round(
            runtime,
            CONFIG.telemetry.timing_precision,
        ),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        qubits=qubits,
        shots=shots,
        qaoa_reps=qaoa_reps,
        route_distance_m=route_distance,
        blocked_roads=blocked_roads,
        scenario=scenario,
        success=success,
        details=details,
    )


def compare_runtime(
    classical_runtime: float,
    quantum_runtime: float,
) -> dict[str, float]:
    """
    Compare classical and quantum runtime.
    """

    difference = (
        quantum_runtime
        - classical_runtime
    )

    if classical_runtime > 0:
        ratio = (
            quantum_runtime
            / classical_runtime
        )
    else:
        ratio = 0.0

    return {
        "classical_runtime_seconds": (
            classical_runtime
        ),
        "quantum_runtime_seconds": (
            quantum_runtime
        ),
        "difference_seconds": difference,
        "quantum_to_classical_ratio": ratio,
    }
