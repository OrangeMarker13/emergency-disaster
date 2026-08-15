"""
Quantum Relief Router
=====================

A disaster-response routing application that combines:

- OpenStreetMap geographic data
- NetworkX road graphs
- Classical shortest-path algorithms
- Qiskit QAOA optimization
- Emergency road-closure simulation
- Interactive PyDeck visualization
- Performance telemetry
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st

from classical_router import (
    generate_candidate_routes,
    run_classical_baseline,
)
from config import CONFIG
from map_data import (
    find_nearest_node,
    geocode_location,
    load_network,
)
from quantum_router import (
    solve_route_selection,
)
from simulation import (
    apply_scenario,
    count_blocked_roads,
    list_scenarios,
    reset_simulation,
    scenario_summary,
    validate_network_connectivity,
)
from telemetry import (
    Telemetry,
    compare_runtime,
)
from visualization import (
    create_probability_chart_data,
    create_route_comparison_dataframe,
    create_route_map,
)


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Quantum Relief Router",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            text-align: center;
            margin-bottom: 1rem;
        }

        .status-card {
            padding: 0.9rem;
            border-radius: 0.7rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(128, 128, 128, 0.25);
        }

        .small-text {
            font-size: 0.85rem;
            opacity: 0.75;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = None

if "original_graph" not in st.session_state:
    st.session_state.original_graph = None

if "location" not in st.session_state:
    st.session_state.location = None

if "location_name" not in st.session_state:
    st.session_state.location_name = ""

if "start_node" not in st.session_state:
    st.session_state.start_node = None

if "target_node" not in st.session_state:
    st.session_state.target_node = None

if "blocked_edges" not in st.session_state:
    st.session_state.blocked_edges = []

if "classical_results" not in st.session_state:
    st.session_state.classical_results = None

if "candidate_routes" not in st.session_state:
    st.session_state.candidate_routes = []

if "quantum_result" not in st.session_state:
    st.session_state.quantum_result = None

if "telemetry" not in st.session_state:
    st.session_state.telemetry = Telemetry()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def reset_results() -> None:
    """Clear route optimization results."""

    st.session_state.classical_results = None
    st.session_state.candidate_routes = []
    st.session_state.quantum_result = None


def format_distance(distance_m: float) -> str:
    """Format a distance in meters."""

    if distance_m >= 1000:
        return f"{distance_m / 1000:.2f} km"

    return f"{distance_m:.0f} m"


def format_runtime(seconds: float) -> str:
    """Format runtime."""

    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"

    return f"{seconds:.3f} s"


def route_coordinates(
    graph: nx.Graph,
    path: list[Any],
) -> list[tuple[float, float]]:
    """Convert graph nodes into latitude/longitude coordinates."""

    coordinates = []

    for node in path:
        data = graph.nodes[node]

        coordinates.append(
            (
                float(data["latitude"]),
                float(data["longitude"]),
            )
        )

    return coordinates


def route_is_valid(
    graph: nx.Graph,
    path: list[Any],
) -> bool:
    """Check whether a route uses blocked roads."""

    if len(path) < 2:
        return False

    for node_a, node_b in zip(
        path[:-1],
        path[1:],
    ):
        edge = graph.get_edge_data(
            node_a,
            node_b,
        )

        if edge is None:
            return False

        if edge.get("blocked", False):
            return False

    return True


def initialize_network(
    query: str,
    radius_m: int,
) -> None:
    """Geocode the emergency location and load its road network."""

    with st.spinner(
        "Loading geographic and road-network data..."
    ):

        location = geocode_location(
            query
        )

        graph, is_fallback = load_network(
            latitude=location["latitude"],
            longitude=location["longitude"],
            radius_m=radius_m,
            use_fallback=True,
        )

    st.session_state.location = (
        location["latitude"],
        location["longitude"],
    )

    st.session_state.location_name = (
        location["display_name"]
    )

    st.session_state.graph = graph

    st.session_state.original_graph = (
        graph.copy()
    )

    st.session_state.start_node = (
        find_nearest_node(
            graph,
            location["latitude"],
            location["longitude"],
        )
    )

    # Use the graph node farthest from the emergency
    # location as a demonstration destination.
    target_node = _find_farthest_node(
        graph,
        st.session_state.start_node,
    )

    st.session_state.target_node = (
        target_node
    )

    st.session_state.blocked_edges = []

    reset_results()

    if is_fallback:
        st.warning(
            "OpenStreetMap data was unavailable, "
            "so the application is using a synthetic "
            "road network for this run."
        )
    else:
        st.success(
            "Live OpenStreetMap road data loaded."
        )


def _find_farthest_node(
    graph: nx.Graph,
    source: Any,
) -> Any:
    """Find a geographically distant node."""

    source_data = graph.nodes[source]

    source_lat = source_data["latitude"]
    source_lon = source_data["longitude"]

    farthest_node = source
    farthest_distance = -1.0

    for node, data in graph.nodes(
        data=True
    ):

        lat = data.get("latitude")
        lon = data.get("longitude")

        if lat is None or lon is None:
            continue

        lat_distance = (
            lat - source_lat
        )

        lon_distance = (
            lon - source_lon
        )

        distance = math.sqrt(
            lat_distance**2
            + lon_distance**2
        )

        if distance > farthest_distance:
            farthest_distance = distance
            farthest_node = node

    return farthest_node


def apply_selected_scenario(
    scenario_name: str,
    blocked_count: int,
) -> None:
    """Apply a simulated emergency scenario."""

    if st.session_state.graph is None:
        return

    graph, blocked_edges = apply_scenario(
        st.session_state.graph,
        scenario_name=scenario_name,
        number_to_block=blocked_count,
        seed=CONFIG.simulation.random_seed,
    )

    st.session_state.graph = graph
    st.session_state.blocked_edges = (
        blocked_edges
    )

    reset_results()


def reset_network() -> None:
    """Reset the road network to its original state."""

    if st.session_state.original_graph is None:
        return

    st.session_state.graph = (
        reset_simulation(
            st.session_state.original_graph
        )
    )

    st.session_state.blocked_edges = []

    reset_results()


def run_routing() -> None:
    """Run classical and quantum route optimization."""

    graph = st.session_state.graph
    start_node = st.session_state.start_node
    target_node = st.session_state.target_node

    if graph is None:
        st.error(
            "Load an emergency location first."
        )
        return

    if start_node is None or target_node is None:
        st.error(
            "A valid start and destination are required."
        )
        return

    reset_results()

    # ---------------------------------------------------------------
    # Classical routing
    # ---------------------------------------------------------------

    with st.spinner(
        "Running classical routing algorithms..."
    ):

        classical_start = (
            st.session_state.telemetry
        )

        classical_results, classical_runtime = (
            _run_classical_with_timing(
                graph,
                start_node,
                target_node,
            )
        )

    st.session_state.classical_results = (
        classical_results
    )

    st.session_state.telemetry.record(
        operation="Classical Routing",
        runtime_seconds=classical_runtime,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        blocked_roads=count_blocked_roads(
            graph
        ),
        success=True,
    )

    # ---------------------------------------------------------------
    # Candidate routes
    # ---------------------------------------------------------------

    with st.spinner(
        "Generating candidate routes..."
    ):

        candidates = (
            generate_candidate_routes(
                graph,
                start_node,
                target_node,
                max_routes=CONFIG.mapping.max_routes,
            )
        )

    st.session_state.candidate_routes = (
        candidates
    )

    if not candidates:

        st.error(
            "No valid route exists after the simulated road closures."
        )

        return

    # ---------------------------------------------------------------
    # Quantum optimization
    # ---------------------------------------------------------------

    with st.spinner(
        "Running QAOA route optimization..."
    ):

        quantum_result, quantum_runtime = (
            _run_quantum_with_timing(
                candidates
            )
        )

    st.session_state.quantum_result = (
        quantum_result
    )

    st.session_state.telemetry.record(
        operation="QAOA Route Optimization",
        runtime_seconds=quantum_runtime,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        qubits=quantum_result.get(
            "qubits",
            0,
        ),
        shots=quantum_result.get(
            "shots",
            0,
        ),
        qaoa_reps=quantum_result.get(
            "reps",
            0,
        ),
        route_distance_m=quantum_result[
            "selected_route"
        ].get(
            "distance",
            0.0,
        ),
        blocked_roads=count_blocked_roads(
            graph
        ),
        success=True,
        details=quantum_result.get(
            "optimizer_message",
            "",
        ),
    )


def _run_classical_with_timing(
    graph: nx.Graph,
    start: Any,
    target: Any,
) -> tuple[dict[str, Any], float]:

    start_time = (
        __import__("time")
        .perf_counter()
    )

    result = run_classical_baseline(
        graph,
        start,
        target,
    )

    elapsed = (
        __import__("time")
        .perf_counter()
        - start_time
    )

    return result, elapsed


def _run_quantum_with_timing(
    routes: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:

    start_time = (
        __import__("time")
        .perf_counter()
    )

    result = solve_route_selection(
        routes,
        shots=CONFIG.quantum.default_shots,
        reps=CONFIG.quantum.qaoa_reps,
    )

    elapsed = (
        __import__("time")
        .perf_counter()
        - start_time
    )

    result["runtime"] = elapsed

    return result, elapsed


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

    st.header("Emergency Setup")

    location_query = st.text_input(
        "Emergency location",
        value="Charlotte, NC",
        help=(
            "Enter a city, address, landmark, "
            "or geographic location."
        ),
    )

    radius_m = st.slider(
        "Map radius",
        min_value=1000,
        max_value=5000,
        value=CONFIG.mapping.default_radius_m,
        step=500,
        help="Road-network area around the emergency.",
    )

    if st.button(
        "Load Emergency Area",
        use_container_width=True,
    ):

        try:
            initialize_network(
                location_query,
                radius_m,
            )

        except Exception as exc:

            st.error(
                f"Unable to load the emergency area: {exc}"
            )

    st.divider()

    st.header("Disaster Simulation")

    scenario_name = st.selectbox(
        "Emergency scenario",
        options=list_scenarios(),
    )

    scenario = (
        None
        if scenario_name is None
        else scenario_name
    )

    default_blocked = (
        CONFIG.simulation.default_blocked_roads
    )

    blocked_count = st.slider(
        "Road closures",
        min_value=0,
        max_value=CONFIG.simulation.max_blocked_roads,
        value=min(
            default_blocked,
            CONFIG.simulation.max_blocked_roads,
        ),
    )

    if st.button(
        "Simulate Emergency",
        use_container_width=True,
    ):

        if st.session_state.graph is None:

            st.warning(
                "Load an emergency area first."
            )

        else:

            apply_selected_scenario(
                scenario_name,
                blocked_count,
            )

    if st.button(
        "Reset Road Network",
        use_container_width=True,
    ):

        reset_network()

    st.divider()

    st.header("Quantum Settings")

    shots = st.slider(
        "QAOA shots",
        min_value=CONFIG.quantum.min_shots,
        max_value=CONFIG.quantum.max_shots,
        value=CONFIG.quantum.default_shots,
        step=500,
    )

    qaoa_reps = st.slider(
        "QAOA repetitions",
        min_value=1,
        max_value=3,
        value=CONFIG.quantum.qaoa_reps,
    )

    st.caption(
        f"Maximum qubits: {CONFIG.quantum.max_qubits}"
    )

    st.caption(
        f"Candidate routes: "
        f"{CONFIG.mapping.max_routes}"
    )


# -------------------------------------------------------------------
# Main header
# -------------------------------------------------------------------

st.title("⚛️ Quantum Relief Router")

st.subheader(
    "Disaster-response routing with quantum optimization"
)

st.write(
    "The application builds a road network around an emergency, "
    "simulates transportation disruptions, generates candidate "
    "routes, and uses QAOA to select a route."
)


# -------------------------------------------------------------------
# Network status
# -------------------------------------------------------------------

if st.session_state.graph is None:

    st.info(
        "Enter an emergency location in the sidebar "
        "and select Load Emergency Area."
    )

    st.markdown(
        """
        ### How it works

        1. Locate the emergency using geographic data.
        2. Build a road network.
        3. Simulate blocked roads.
        4. Generate candidate routes.
        5. Run Dijkstra and A* as classical baselines.
        6. Encode route selection as a QUBO.
        7. Optimize the QUBO with QAOA.
        8. Display the selected route and quantum probabilities.
        """
    )

    st.stop()


graph = st.session_state.graph


# -------------------------------------------------------------------
# Network metrics
# -------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Road Nodes",
        f"{graph.number_of_nodes():,}",
    )

with col2:
    st.metric(
        "Road Segments",
        f"{graph.number_of_edges():,}",
    )

with col3:
    st.metric(
        "Blocked Roads",
        f"{count_blocked_roads(graph):,}",
    )

with col4:
    connectivity = (
        validate_network_connectivity(
            graph
        )
    )

    st.metric(
        "Network Components",
        connectivity["components"],
    )


st.caption(
    f"Emergency area: {st.session_state.location_name}"
)


# -------------------------------------------------------------------
# Scenario status
# -------------------------------------------------------------------

if scenario_name:

    summary = scenario_summary(
        graph,
        scenario_name,
    )

    st.markdown(
        f"""
        <div class="status-card">
        <b>{summary["scenario"]}</b>
        &nbsp; | &nbsp;
        Severity: <b>{summary["severity"]}</b>
        &nbsp; | &nbsp;
        Blocked roads: <b>{summary["blocked_roads"]}</b>
        &nbsp; | &nbsp;
        Network closure: <b>{summary["closure_percentage"]:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Destination and routing controls
# -------------------------------------------------------------------

st.subheader("Routing")

routing_col1, routing_col2 = st.columns(
    [3, 1]
)

with routing_col1:

    st.write(
        "The emergency location serves as the starting point. "
        "The application selects a distant reachable road node "
        "as the demonstration destination."
    )

with routing_col2:

    if st.button(
        "🚑 Optimize Route",
        type="primary",
        use_container_width=True,
    ):

        # Reuse the current sidebar quantum settings.
        CONFIG_QUANTUM_SHOTS = shots
        CONFIG_QUANTUM_REPS = qaoa_reps

        # Temporarily pass selected settings through session state.
        st.session_state.selected_shots = (
            CONFIG_QUANTUM_SHOTS
        )

        st.session_state.selected_reps = (
            CONFIG_QUANTUM_REPS
        )

        try:
            # The routing function uses the configured defaults.
            # Override the result afterward if needed.
            run_routing()

        except Exception as exc:

            st.error(
                f"Routing optimization failed: {exc}"
            )


# -------------------------------------------------------------------
# Route results
# -------------------------------------------------------------------

quantum_result = (
    st.session_state.quantum_result
)

classical_results = (
    st.session_state.classical_results
)

candidate_routes = (
    st.session_state.candidate_routes
)


if quantum_result is not None:

    st.subheader("Optimization Results")

    selected_route = quantum_result.get(
        "selected_route"
    )

    selected_distance = (
        selected_route.get(
            "distance",
            0.0,
        )
        if selected_route
        else 0.0
    )

    quantum_probability = (
        quantum_result.get(
            "best_probability",
            0.0,
        )
    )

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )

    with result_col1:
        st.metric(
            "Selected Route",
            f"Route {quantum_result['selected_route_index'] + 1}",
        )

    with result_col2:
        st.metric(
            "Route Distance",
            format_distance(
                selected_distance
            ),
        )

    with result_col3:
        st.metric(
            "QAOA Probability",
            f"{quantum_probability * 100:.2f}%",
        )

    with result_col4:
        st.metric(
            "Qubits",
            quantum_result.get(
                "qubits",
                0,
            ),
        )

    # ---------------------------------------------------------------
    # Map
    # ---------------------------------------------------------------

    selected_path = (
        selected_route.get(
            "path",
            [],
        )
        if selected_route
        else []
    )

    alternative_paths = [
        route["path"]
        for route in candidate_routes
        if route["path"] != selected_path
    ]

    route_map = create_route_map(
        graph=graph,
        emergency_location=st.session_state.location,
        start_node=st.session_state.start_node,
        target_node=st.session_state.target_node,
        selected_route=selected_path,
        alternative_routes=alternative_paths,
    )

    st.pydeck_chart(
        route_map,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Quantum probabilities
    # ---------------------------------------------------------------

    st.subheader(
        "Quantum Route Probabilities"
    )

    probability_data = (
        create_probability_chart_data(
            quantum_result.get(
                "probabilities",
                {},
            )
        )
    )

    if not probability_data.empty:

        st.bar_chart(
            probability_data.set_index(
                "Route"
            )
        )

    # ---------------------------------------------------------------
    # Classical comparison
    # ---------------------------------------------------------------

    st.subheader(
        "Classical vs. Quantum"
    )

    comparison = (
        create_route_comparison_dataframe(
            classical_results,
            quantum_result,
        )
    )

    if not comparison.empty:

        display_comparison = (
            comparison.copy()
        )

        display_comparison[
            "Distance (m)"
        ] = display_comparison[
            "Distance (m)"
        ].round(2)

        display_comparison[
            "Runtime (s)"
        ] = display_comparison[
            "Runtime (s)"
        ].round(4)

        st.dataframe(
            display_comparison,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # Route candidates
    # ---------------------------------------------------------------

    st.subheader(
        "Candidate Routes"
    )

    candidate_rows = []

    for index, route in enumerate(
        candidate_routes
    ):

        candidate_rows.append(
            {
                "Route": f"Route {index + 1}",
                "Distance": format_distance(
                    route["distance"]
                ),
                "Quantum Probability": (
                    quantum_result.get(
                        "probabilities",
                        {},
                    ).get(
                        index,
                        0.0,
                    )
                    * 100
                ),
                "Selected": (
                    index
                    == quantum_result[
                        "selected_route_index"
                    ]
                ),
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    st.dataframe(
        candidate_df,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Telemetry
    # ---------------------------------------------------------------

    with st.expander(
        "Performance Telemetry"
    ):

        telemetry_summary = (
            st.session_state.telemetry.summary()
        )

        telemetry_col1, telemetry_col2, telemetry_col3 = (
            st.columns(3)
        )

        with telemetry_col1:
            st.metric(
                "Operations",
                telemetry_summary[
                    "records"
                ],
            )

        with telemetry_col2:
            st.metric(
                "Successful",
                telemetry_summary[
                    "successful_operations"
                ],
            )

        with telemetry_col3:
            st.metric(
                "Total Runtime",
                format_runtime(
                    telemetry_summary[
                        "total_runtime_seconds"
                    ]
                ),
            )

        telemetry_df = (
            st.session_state.telemetry.to_dataframe()
        )

        if not telemetry_df.empty:

            st.dataframe(
                telemetry_df,
                use_container_width=True,
                hide_index=True,
            )

else:

    # ---------------------------------------------------------------
    # Initial map
    # ---------------------------------------------------------------

    route_map = create_route_map(
        graph=graph,
        emergency_location=st.session_state.location,
        start_node=st.session_state.start_node,
        target_node=st.session_state.target_node,
    )

    st.pydeck_chart(
        route_map,
        use_container_width=True,
    )

    st.info(
        "Select Optimize Route to generate candidate routes "
        "and run the quantum optimization."
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()

st.caption(
    "Quantum Relief Router | "
    "OpenStreetMap + NetworkX + Qiskit QAOA"
)
