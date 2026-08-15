"""
Visualization utilities for Quantum Relief Router.

Provides:
- Interactive PyDeck road-network maps
- Route overlays
- Emergency location markers
- Blocked-road visualization
- Quantum probability charts
- Classical vs. quantum performance data
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk


def _edge_rows(
    graph,
) -> list[dict[str, Any]]:
    """
    Convert graph edges into rows suitable for PyDeck.
    """

    rows = []

    for node_a, node_b, data in graph.edges(
        data=True
    ):
        start = graph.nodes[node_a]
        end = graph.nodes[node_b]

        rows.append(
            {
                "start_lat": start["latitude"],
                "start_lon": start["longitude"],
                "end_lat": end["latitude"],
                "end_lon": end["longitude"],
                "road_name": data.get(
                    "road_name",
                    "Unnamed road",
                ),
                "distance": float(
                    data.get(
                        "distance",
                        0.0,
                    )
                ),
                "blocked": bool(
                    data.get(
                        "blocked",
                        False,
                    )
                ),
            }
        )

    return rows


def _route_rows(
    graph,
    path: list[Any],
) -> list[dict[str, Any]]:
    """
    Convert a route path into PyDeck line segments.
    """

    rows = []

    if len(path) < 2:
        return rows

    for node_a, node_b in zip(
        path[:-1],
        path[1:],
    ):
        start = graph.nodes[node_a]
        end = graph.nodes[node_b]

        edge_data = graph.get_edge_data(
            node_a,
            node_b,
        ) or {}

        rows.append(
            {
                "start_lat": start["latitude"],
                "start_lon": start["longitude"],
                "end_lat": end["latitude"],
                "end_lon": end["longitude"],
                "distance": float(
                    edge_data.get(
                        "distance",
                        0.0,
                    )
                ),
            }
        )

    return rows


def create_route_map(
    graph,
    emergency_location: tuple[float, float] | None = None,
    start_node: Any | None = None,
    target_node: Any | None = None,
    selected_route: list[Any] | None = None,
    alternative_routes: list[list[Any]] | None = None,
    height: int = 600,
) -> pdk.Deck:
    """
    Build the main interactive routing map.

    Args:
        graph: NetworkX road graph.
        emergency_location: (latitude, longitude).
        start_node: Starting graph node.
        target_node: Destination graph node.
        selected_route: Quantum-selected route.
        alternative_routes: Other candidate routes.
        height: Map height in pixels.
    """

    edge_data = _edge_rows(graph)

    edge_df = pd.DataFrame(
        edge_data
    )

    layers = []

    if not edge_df.empty:

        available_edges = edge_df[
            ~edge_df["blocked"]
        ].copy()

        blocked_edges = edge_df[
            edge_df["blocked"]
        ].copy()

        if not available_edges.empty:

            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=available_edges,
                    get_source_position=[
                        "start_lon",
                        "start_lat",
                    ],
                    get_target_position=[
                        "end_lon",
                        "end_lat",
                    ],
                    get_width=3,
                    pickable=True,
                )
            )

        if not blocked_edges.empty:

            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=blocked_edges,
                    get_source_position=[
                        "start_lon",
                        "start_lat",
                    ],
                    get_target_position=[
                        "end_lon",
                        "end_lat",
                    ],
                    get_width=6,
                    pickable=True,
                )
            )

    if alternative_routes:

        alternative_rows = []

        for route in alternative_routes:

            alternative_rows.extend(
                _route_rows(
                    graph,
                    route,
                )
            )

        if alternative_rows:

            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=pd.DataFrame(
                        alternative_rows
                    ),
                    get_source_position=[
                        "start_lon",
                        "start_lat",
                    ],
                    get_target_position=[
                        "end_lon",
                        "end_lat",
                    ],
                    get_width=5,
                    pickable=True,
                )
            )

    if selected_route:

        selected_rows = _route_rows(
            graph,
            selected_route,
        )

        if selected_rows:

            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=pd.DataFrame(
                        selected_rows
                    ),
                    get_source_position=[
                        "start_lon",
                        "start_lat",
                    ],
                    get_target_position=[
                        "end_lon",
                        "end_lat",
                    ],
                    get_width=10,
                    pickable=True,
                )
            )

    marker_rows = []

    if emergency_location:

        marker_rows.append(
            {
                "latitude": emergency_location[0],
                "longitude": emergency_location[1],
                "type": "Emergency",
            }
        )

    if start_node is not None:

        start_data = graph.nodes[start_node]

        marker_rows.append(
            {
                "latitude": start_data["latitude"],
                "longitude": start_data["longitude"],
                "type": "Start",
            }
        )

    if target_node is not None:

        target_data = graph.nodes[target_node]

        marker_rows.append(
            {
                "latitude": target_data["latitude"],
                "longitude": target_data["longitude"],
                "type": "Destination",
            }
        )

    if marker_rows:

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(
                    marker_rows
                ),
                get_position=[
                    "longitude",
                    "latitude",
                ],
                get_radius=100,
                pickable=True,
            )
        )

    if emergency_location:

        center_lat = emergency_location[0]
        center_lon = emergency_location[1]

    elif marker_rows:

        center_lat = marker_rows[0][
            "latitude"
        ]
        center_lon = marker_rows[0][
            "longitude"
        ]

    elif graph.number_of_nodes() > 0:

        first_node = next(
            iter(graph.nodes)
        )

        center_lat = graph.nodes[
            first_node
        ]["latitude"]

        center_lon = graph.nodes[
            first_node
        ]["longitude"]

    else:

        center_lat = 0.0
        center_lon = 0.0

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        pitch=0,
        bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>Road:</b> {road_name}<br/>"
                "<b>Distance:</b> {distance} m"
            )
        },
        map_style=None,
    )


def create_probability_dataframe(
    probabilities: dict[int, float],
    routes: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert quantum measurement probabilities into a DataFrame.
    """

    rows = []

    for index, route in enumerate(routes):

        probability = float(
            probabilities.get(
                index,
                0.0,
            )
        )

        rows.append(
            {
                "Route": f"Route {index + 1}",
                "Distance (m)": float(
                    route.get(
                        "distance",
                        0.0,
                    )
                ),
                "Probability": probability,
            }
        )

    return pd.DataFrame(rows)


def create_route_comparison_dataframe(
    classical_results: dict[str, Any],
    quantum_result: dict[str, Any],
) -> pd.DataFrame:
    """
    Create a DataFrame comparing routing approaches.
    """

    rows = []

    dijkstra = classical_results.get(
        "dijkstra"
    )

    if dijkstra:

        rows.append(
            {
                "Method": "Dijkstra",
                "Distance (m)": (
                    dijkstra["distance"]
                    if dijkstra["reachable"]
                    else None
                ),
                "Runtime (s)": dijkstra[
                    "runtime"
                ],
                "Status": (
                    "Route found"
                    if dijkstra["reachable"]
                    else "No route"
                ),
            }
        )

    astar = classical_results.get(
        "astar"
    )

    if astar:

        rows.append(
            {
                "Method": "A*",
                "Distance (m)": (
                    astar["distance"]
                    if astar["reachable"]
                    else None
                ),
                "Runtime (s)": astar[
                    "runtime"
                ],
                "Status": (
                    "Route found"
                    if astar["reachable"]
                    else "No route"
                ),
            }
        )

    if quantum_result:

        selected_route = quantum_result.get(
            "selected_route"
        )

        rows.append(
            {
                "Method": "QAOA",
                "Distance (m)": (
                    selected_route.get(
                        "distance"
                    )
                    if selected_route
                    else None
                ),
                "Runtime (s)": quantum_result.get(
                    "runtime",
                    0.0,
                ),
                "Status": (
                    "Quantum optimization"
                    if quantum_result.get(
                        "quantum_available",
                        False,
                    )
                    else "Classical fallback"
                ),
            }
        )

    return pd.DataFrame(rows)


def create_probability_chart_data(
    probabilities: dict[int, float],
) -> pd.DataFrame:
    """
    Create compact probability chart data.
    """

    return pd.DataFrame(
        {
            "Route": [
                f"Route {index + 1}"
                for index in probabilities
            ],
            "Probability": [
                float(value)
                for value in probabilities.values()
            ],
        }
    )


def create_telemetry_dataframe(
    telemetry_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert telemetry records into a DataFrame.
    """

    if not telemetry_records:
        return pd.DataFrame()

    return pd.DataFrame(
        telemetry_records
    )
