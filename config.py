"""
Configuration for Quantum Relief Router.

Centralizes application settings so the rest of the project
does not contain scattered magic numbers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantumConfig:
    """Quantum optimization settings."""

    max_qubits: int = 12
    default_qubits: int = 8

    qaoa_reps: int = 2
    maxiter: int = 100

    default_shots: int = 2000
    min_shots: int = 500
    max_shots: int = 5000

    # Number of candidate routes considered by the quantum optimizer.
    max_candidates: int = 8

    # Penalty used in the QUBO formulation for constraint violations.
    constraint_penalty: float = 1000.0


@dataclass(frozen=True)
class MapConfig:
    """Mapping and routing settings."""

    default_zoom: int = 13

    # Maximum number of road-network nodes retained after downloading
    # OpenStreetMap data.
    max_nodes: int = 250

    # Maximum number of candidate routes sent to the optimizer.
    max_routes: int = 8

    # Search radius around the requested emergency location.
    default_radius_m: int = 3000

    # OpenStreetMap services.
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"

    overpass_url: str = "https://overpass-api.de/api/interpreter"

    user_agent: str = "QuantumReliefRouter/1.0"


@dataclass(frozen=True)
class SimulationConfig:
    """Emergency simulation settings."""

    default_blocked_roads: int = 2
    max_blocked_roads: int = 10

    random_seed: int = 42

    # Fraction of roads eligible for simulated closure.
    max_closure_fraction: float = 0.15


@dataclass(frozen=True)
class TelemetryConfig:
    """Application performance settings."""

    enabled: bool = True

    # Number of decimal places used when displaying timings.
    timing_precision: int = 3


@dataclass(frozen=True)
class AppConfig:
    """Global application configuration."""

    app_name: str = "Quantum Relief Router"

    quantum: QuantumConfig = QuantumConfig()
    mapping: MapConfig = MapConfig()
    simulation: SimulationConfig = SimulationConfig()
    telemetry: TelemetryConfig = TelemetryConfig()


CONFIG = AppConfig()
