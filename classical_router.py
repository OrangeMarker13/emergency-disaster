"""
Classical routing engine for Quantum Relief Router.

Provides:
- Dijkstra shortest-path routing
- A* shortest-path routing
- Route distance calculation
- Multiple candidate route generation
- Classical baseline metrics
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any

import networkx as nx


def _heuristic(
    graph: nx.Graph,
    node_a: Any,
    node_b: Any,
) -> float:
    """
    Geographic heuristic for A* routing.

    Returns an estimated distance in meters.
    """

    a = graph.nodes[node_a]
    b = graph.nodes[node_b]

    lat1 = math.radians(a["latitude"])
    lat2 = math.radians(b["latitude"])

    lon1 = math.radians(a["longitude"])
    lon2 = math.radians(b["longitude"])

    earth_radius = 6_371_000.0

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return (
        earth_radius
        * 2
        * math.atan2(
            math.sqrt(value),
            math.sqrt(1 - value),
        )
    )


def _edge_is_blocked(
    graph: nx.Graph,
    node_a: Any,
    node_b: Any,
) -> bool:
    """
    Determine whether an edge is currently blocked.
    """

    edge_data = graph.get_edge_data(node_a, node_b)

    if edge_data is None:
        return True

    return bool(edge_data.get("blocked", False))


def dijkstra_route(
    graph: nx.Graph,
    start: Any,
    target: Any,
) -> dict[str, Any]:
    """
    Find the shortest available route using Dijkstra's algorithm.
    """

    start_time = time.perf_counter()

    if start not in graph:
        raise ValueError("Start node does not exist.")

    if target not in graph:
        raise ValueError("Target node does not exist.")

    distances: dict[Any, float] = {
        start: 0.0
    }

    previous: dict[Any, Any] = {}

    queue: list[tuple[float, Any]] = [
        (0.0, start)
    ]

    visited: set[Any] = set()

    while queue:

        current_distance, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)

        if current == target:
            break

        for neighbor in graph.neighbors(current):

            if _edge_is_blocked(
                graph,
                current,
                neighbor,
            ):
                continue

            edge_data = graph.get_edge_data(
                current,
                neighbor,
            )

            weight = float(
                edge_data.get("distance", 1.0)
            )

            new_distance = (
                current_distance + weight
            )

            if new_distance < distances.get(
                neighbor,
                float("inf"),
            ):
                distances[neighbor] = new_distance
                previous[neighbor] = current

                heapq.heappush(
                    queue,
                    (
                        new_distance,
                        neighbor,
                    ),
                )

    if target not in distances:
        raise nx.NetworkXNoPath(
            "No available route exists between the selected nodes."
        )

    path = _reconstruct_path(
        previous,
        start,
        target,
    )

    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": "Dijkstra",
        "path": path,
        "distance": distances[target],
        "runtime": elapsed,
        "reachable": True,
    }


def astar_route(
    graph: nx.Graph,
    start: Any,
    target: Any,
) -> dict[str, Any]:
    """
    Find the shortest available route using A*.
    """

    start_time = time.perf_counter()

    if start not in graph:
        raise ValueError("Start node does not exist.")

    if target not in graph:
        raise ValueError("Target node does not exist.")

    open_set: list[
        tuple[float, float, Any]
    ] = []

    heapq.heappush(
        open_set,
        (
            _heuristic(graph, start, target),
            0.0,
            start,
        ),
    )

    g_score: dict[Any, float] = {
        start: 0.0
    }

    previous: dict[Any, Any] = {}

    visited: set[Any] = set()

    while open_set:

        _, current_cost, current = heapq.heappop(
            open_set
        )

        if current in visited:
            continue

        visited.add(current)

        if current == target:
            break

        for neighbor in graph.neighbors(current):

            if _edge_is_blocked(
                graph,
                current,
                neighbor,
            ):
                continue

            edge_data = graph.get_edge_data(
                current,
                neighbor,
            )

            weight = float(
                edge_data.get("distance", 1.0)
            )

            tentative_g = (
                current_cost + weight
            )

            if tentative_g < g_score.get(
                neighbor,
                float("inf"),
            ):
                g_score[neighbor] = tentative_g
                previous[neighbor] = current

                f_score = (
                    tentative_g
                    + _heuristic(
                        graph,
                        neighbor,
                        target,
                    )
                )

                heapq.heappush(
                    open_set,
                    (
                        f_score,
                        tentative_g,
                        neighbor,
                    ),
                )

    if target not in g_score:
        raise nx.NetworkXNoPath(
            "No available route exists between the selected nodes."
        )

    path = _reconstruct_path(
        previous,
        start,
        target,
    )

    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": "A*",
        "path": path,
        "distance": g_score[target],
        "runtime": elapsed,
        "reachable": True,
    }


def _reconstruct_path(
    previous: dict[Any, Any],
    start: Any,
    target: Any,
) -> list[Any]:
    """
    Reconstruct a path from predecessor information.
    """

    path = [target]
    current = target

    while current != start:

        if current not in previous:
            raise nx.NetworkXNoPath(
                "Unable to reconstruct route."
            )

        current = previous[current]
        path.append(current)

    path.reverse()

    return path


def route_distance(
    graph: nx.Graph,
    path: list[Any],
) -> float:
    """
    Calculate total distance of a route.
    """

    if len(path) < 2:
        return 0.0

    total = 0.0

    for start, end in zip(
        path[:-1],
        path[1:],
    ):
        edge_data = graph.get_edge_data(
            start,
            end,
        )

        if edge_data is None:
            raise ValueError(
                "Route contains an invalid edge."
            )

        total += float(
            edge_data.get(
                "distance",
                0.0,
            )
        )

    return total


def generate_candidate_routes(
    graph: nx.Graph,
    start: Any,
    target: Any,
    max_routes: int = 8,
) -> list[dict[str, Any]]:
    """
    Generate multiple reasonable candidate routes.

    Uses NetworkX shortest-simple-path generation
    while respecting blocked roads.
    """

    start_time = time.perf_counter()

    if start not in graph:
        raise ValueError("Start node does not exist.")

    if target not in graph:
        raise ValueError("Target node does not exist.")

    available_graph = graph.copy()

    blocked_edges = [
        (
            u,
            v,
        )
        for u, v, data in available_graph.edges(
            data=True
        )
        if data.get("blocked", False)
    ]

    available_graph.remove_edges_from(
        blocked_edges
    )

    candidates: list[dict[str, Any]] = []

    try:
        paths = nx.shortest_simple_paths(
            available_graph,
            start,
            target,
            weight="distance",
        )

        for index, path in enumerate(paths):

            if index >= max_routes:
                break

            distance = route_distance(
                available_graph,
                path,
            )

            candidates.append(
                {
                    "id": index,
                    "path": path,
                    "distance": distance,
                    "rank": index + 1,
                }
            )

    except nx.NetworkXNoPath:
        return []

    elapsed = time.perf_counter() - start_time

    for candidate in candidates:
        candidate["generation_runtime"] = elapsed

    return candidates


def run_classical_baseline(
    graph: nx.Graph,
    start: Any,
    target: Any,
) -> dict[str, Any]:
    """
    Run both classical routing algorithms and return
    their results for comparison with the quantum optimizer.
    """

    results: dict[str, Any] = {}

    try:
        results["dijkstra"] = dijkstra_route(
            graph,
            start,
            target,
        )
    except nx.NetworkXNoPath:
        results["dijkstra"] = {
            "algorithm": "Dijkstra",
            "reachable": False,
            "path": [],
            "distance": float("inf"),
            "runtime": 0.0,
        }

    try:
        results["astar"] = astar_route(
            graph,
            start,
            target,
        )
    except nx.NetworkXNoPath:
        results["astar"] = {
            "algorithm": "A*",
            "reachable": False,
            "path": [],
            "distance": float("inf"),
            "runtime": 0.0,
        }

    return results
