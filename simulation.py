"""
Emergency scenario and road-closure simulation for Quantum Relief Router.

This module provides:
- Random road closures
- Controlled road closures
- Emergency scenarios
- Graph recovery
- Scenario summaries
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import networkx as nx

from config import CONFIG


@dataclass
class EmergencyScenario:
    """
    Describes a simulated emergency situation.
    """

    name: str
    description: str
    severity: str
    blocked_roads: int


SCENARIOS = {
    "Earthquake": EmergencyScenario(
        name="Earthquake",
        description=(
            "Simulates road closures caused by structural damage "
            "and debris."
        ),
        severity="High",
        blocked_roads=4,
    ),
    "Flood": EmergencyScenario(
        name="Flood",
        description=(
            "Simulates flooded roads and inaccessible low-lying areas."
        ),
        severity="High",
        blocked_roads=5,
    ),
    "Wildfire": EmergencyScenario(
        name="Wildfire",
        description=(
            "Simulates road closures caused by wildfire evacuation zones."
        ),
        severity="Critical",
        blocked_roads=6,
    ),
    "Hurricane": EmergencyScenario(
        name="Hurricane",
        description=(
            "Simulates widespread transportation disruptions."
        ),
        severity="Critical",
        blocked_roads=7,
    ),
    "Tornado": EmergencyScenario(
        name="Tornado",
        description=(
            "Simulates localized road closures from storm damage."
        ),
        severity="High",
        blocked_roads=3,
    ),
    "Medical Emergency": EmergencyScenario(
        name="Medical Emergency",
        description=(
            "Simulates a smaller emergency requiring rapid routing."
        ),
        severity="Medium",
        blocked_roads=2,
    ),
}


def get_scenario(
    scenario_name: str,
) -> EmergencyScenario:
    """
    Retrieve a predefined emergency scenario.
    """

    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown emergency scenario: {scenario_name}"
        )

    return SCENARIOS[scenario_name]


def list_scenarios() -> list[str]:
    """
    Return available scenario names.
    """

    return list(SCENARIOS.keys())


def _edge_score(
    graph: nx.Graph,
    edge: tuple[Any, Any],
) -> float:
    """
    Assign a score to an edge for closure selection.

    Higher scores make an edge more likely to be selected.

    Edges with higher traffic-like connectivity are given more weight,
    since closing them creates a meaningful routing challenge.
    """

    node_a, node_b = edge

    degree_a = graph.degree(node_a)
    degree_b = graph.degree(node_b)

    edge_data = graph.get_edge_data(
        node_a,
        node_b,
    ) or {}

    distance = float(
        edge_data.get(
            "distance",
            1.0,
        )
    )

    connectivity_score = (
        degree_a + degree_b
    )

    distance_score = min(
        distance / 100.0,
        10.0,
    )

    return (
        connectivity_score
        + distance_score
    )


def select_roads_to_block(
    graph: nx.Graph,
    number_to_block: int,
    seed: int | None = None,
) -> list[tuple[Any, Any]]:
    """
    Select roads for simulated closure.

    The selection is weighted toward roads with higher graph
    connectivity so the simulation creates meaningful route changes.
    """

    if graph.number_of_edges() == 0:
        return []

    number_to_block = max(
        0,
        min(
            number_to_block,
            graph.number_of_edges(),
        ),
    )

    if number_to_block == 0:
        return []

    random_generator = random.Random(
        seed
        if seed is not None
        else CONFIG.simulation.random_seed
    )

    candidates = []

    for node_a, node_b, data in graph.edges(
        data=True
    ):
        if data.get("blocked", False):
            continue

        score = _edge_score(
            graph,
            (node_a, node_b),
        )

        candidates.append(
            (
                node_a,
                node_b,
                score,
            )
        )

    if not candidates:
        return []

    selected: list[tuple[Any, Any]] = []

    # Weighted sampling without replacement.
    remaining = candidates.copy()

    for _ in range(
        min(
            number_to_block,
            len(remaining),
        )
    ):
        total_weight = sum(
            max(item[2], 0.01)
            for item in remaining
        )

        random_value = (
            random_generator.random()
            * total_weight
        )

        cumulative = 0.0

        selected_index = 0

        for index, item in enumerate(
            remaining
        ):
            cumulative += max(
                item[2],
                0.01,
            )

            if cumulative >= random_value:
                selected_index = index
                break

        node_a, node_b, _ = remaining.pop(
            selected_index
        )

        selected.append(
            (
                node_a,
                node_b,
            )
        )

    return selected


def block_roads(
    graph: nx.Graph,
    edges: list[tuple[Any, Any]],
) -> nx.Graph:
    """
    Mark selected roads as blocked.

    Returns a copy of the original graph.
    """

    updated_graph = graph.copy()

    for node_a, node_b in edges:

        if not updated_graph.has_edge(
            node_a,
            node_b,
        ):
            continue

        updated_graph[node_a][node_b][
            "blocked"
        ] = True

    return updated_graph


def unblock_all_roads(
    graph: nx.Graph,
) -> nx.Graph:
    """
    Remove all simulated road closures.
    """

    updated_graph = graph.copy()

    for node_a, node_b, data in (
        updated_graph.edges(data=True)
    ):
        data["blocked"] = False

    return updated_graph


def apply_scenario(
    graph: nx.Graph,
    scenario_name: str,
    number_to_block: int | None = None,
    seed: int | None = None,
) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """
    Apply an emergency scenario to a road graph.

    Returns:
        Updated graph and list of blocked edges.
    """

    scenario = get_scenario(
        scenario_name
    )

    if number_to_block is None:
        number_to_block = (
            scenario.blocked_roads
        )

    number_to_block = min(
        number_to_block,
        CONFIG.simulation.max_blocked_roads,
    )

    blocked_edges = select_roads_to_block(
        graph=graph,
        number_to_block=number_to_block,
        seed=seed,
    )

    updated_graph = block_roads(
        graph,
        blocked_edges,
    )

    return (
        updated_graph,
        blocked_edges,
    )


def manually_block_roads(
    graph: nx.Graph,
    edges: list[tuple[Any, Any]],
) -> nx.Graph:
    """
    Apply user-selected road closures.
    """

    return block_roads(
        graph,
        edges,
    )


def count_blocked_roads(
    graph: nx.Graph,
) -> int:
    """
    Count currently blocked roads.
    """

    return sum(
        1
        for _, _, data in graph.edges(
            data=True
        )
        if data.get("blocked", False)
    )


def calculate_closure_percentage(
    graph: nx.Graph,
) -> float:
    """
    Calculate the percentage of roads currently blocked.
    """

    total_edges = graph.number_of_edges()

    if total_edges == 0:
        return 0.0

    blocked = count_blocked_roads(
        graph
    )

    return (
        blocked
        / total_edges
        * 100.0
    )


def scenario_summary(
    graph: nx.Graph,
    scenario_name: str,
) -> dict[str, Any]:
    """
    Create a summary of the current emergency scenario.
    """

    scenario = get_scenario(
        scenario_name
    )

    blocked = count_blocked_roads(
        graph
    )

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "severity": scenario.severity,
        "blocked_roads": blocked,
        "total_roads": graph.number_of_edges(),
        "closure_percentage": calculate_closure_percentage(
            graph
        ),
    }


def validate_network_connectivity(
    graph: nx.Graph,
) -> dict[str, Any]:
    """
    Analyze connectivity after road closures.
    """

    if graph.number_of_nodes() == 0:
        return {
            "connected": False,
            "components": 0,
            "largest_component": 0,
        }

    active_graph = graph.copy()

    blocked_edges = [
        (node_a, node_b)
        for node_a, node_b, data in (
            active_graph.edges(
                data=True
            )
        )
        if data.get("blocked", False)
    ]

    active_graph.remove_edges_from(
        blocked_edges
    )

    components = list(
        nx.connected_components(
            active_graph
        )
    )

    largest_component = max(
        (
            len(component)
            for component in components
        ),
        default=0,
    )

    return {
        "connected": (
            len(components) <= 1
        ),
        "components": len(components),
        "largest_component": largest_component,
        "total_nodes": active_graph.number_of_nodes(),
    }


def reset_simulation(
    graph: nx.Graph,
) -> nx.Graph:
    """
    Restore the network to its original unblocked state.
    """

    return unblock_all_roads(
        graph
    )
