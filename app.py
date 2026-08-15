import json
import math
import time
import urllib.parse
import urllib.request
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import pydeck as pdk
import streamlit as st

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Relief Router",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1220;
            color: #f1f5f9;
        }

        [data-testid="stSidebar"] {
            background-color: #111827;
        }

        .metric-card {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 12px;
            padding: 18px;
            min-height: 115px;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.82rem;
            margin-bottom: 7px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.55rem;
            font-weight: 700;
        }

        .metric-detail {
            color: #64748b;
            font-size: 0.75rem;
            margin-top: 5px;
        }

        .status-good {
            color: #4ade80;
            font-weight: 700;
        }

        .status-warning {
            color: #facc15;
            font-weight: 700;
        }

        .status-neutral {
            color: #60a5fa;
            font-weight: 700;
        }

        h1, h2, h3 {
            color: #f8fafc;
        }

        .route-box {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 12px;
            padding: 16px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

NOMINATIM_USER_AGENT = "QuantumReliefRouter-CongressionalAppChallenge/1.0"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

MAX_QAOA_QUBITS = 12
QAOA_REPS = 2
QAOA_SHOTS = 512
QAOA_MAXITER = 45

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_VIEW = {
    "latitude": 35.2271,
    "longitude": -80.8431,
    "zoom": 10,
    "pitch": 45,
}

if "view_state" not in st.session_state:
    st.session_state.view_state = DEFAULT_VIEW

if "result" not in st.session_state:
    st.session_state.result = None

if "last_location" not in st.session_state:
    st.session_state.last_location = None


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location(query: str) -> Optional[Dict]:
    geolocator = Nominatim(
        user_agent=NOMINATIM_USER_AGENT,
        timeout=15,
    )

    location = geolocator.geocode(
        query,
        exactly_one=True,
        addressdetails=True,
    )

    if location is None:
        return None

    return {
        "latitude": float(location.latitude),
        "longitude": float(location.longitude),
        "display_name": location.address,
    }


# ============================================================
# OSM / OVERPASS
# ============================================================

def overpass_request(query: str) -> Optional[Dict]:
    encoded = urllib.parse.urlencode({"data": query}).encode("utf-8")

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            request = urllib.request.Request(
                endpoint,
                data=encoded,
                headers={
                    "User-Agent": NOMINATIM_USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=35) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)

        except Exception:
            continue

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_network(
    latitude: float,
    longitude: float,
    radius_m: int = 7000,
) -> Optional[Dict]:

    query = f"""
    [out:json][timeout:30];

    way(
        around:{radius_m},
        {latitude},
        {longitude}
    )
    ["highway"]
    ["highway"!~"footway|path|cycleway|steps|pedestrian|construction|proposed|service"];

    out body;
    >;
    out skel qt;
    """

    data = overpass_request(query)

    if not data or "elements" not in data:
        return None

    return data


def build_osm_graph(data: Dict) -> nx.Graph:
    graph = nx.Graph()

    nodes = {}
    ways = []

    for element in data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (
                float(element["lat"]),
                float(element["lon"]),
            )

        elif element["type"] == "way":
            ways.append(element)

    for way in ways:
        refs = way.get("nodes", [])

        for a, b in zip(refs[:-1], refs[1:]):
            if a not in nodes or b not in nodes:
                continue

            lat1, lon1 = nodes[a]
            lat2, lon2 = nodes[b]

            distance_km = geodesic(
                (lat1, lon1),
                (lat2, lon2),
            ).km

            if distance_km <= 2.0:
                graph.add_edge(
                    a,
                    b,
                    distance_km=distance_km,
                    lat1=lat1,
                    lon1=lon1,
                    lat2=lat2,
                    lon2=lon2,
                )

    return graph


# ============================================================
# SYNTHETIC FALLBACK
# ============================================================

def build_fallback_graph(
    latitude: float,
    longitude: float,
) -> nx.Graph:

    graph = nx.Graph()

    # Approximately 0.02 degrees of latitude/longitude around
    # the search center.
    offsets = [
        (-0.035, -0.035),
        (-0.035, 0.0),
        (-0.035, 0.035),
        (0.0, -0.035),
        (0.0, 0.0),
        (0.0, 0.035),
        (0.035, -0.035),
        (0.035, 0.0),
        (0.035, 0.035),
    ]

    for i, (dlat, dlon) in enumerate(offsets):
        graph.add_node(
            i,
            latitude=latitude + dlat,
            longitude=longitude + dlon,
        )

    for i in graph.nodes:
        lat1 = graph.nodes[i]["latitude"]
        lon1 = graph.nodes[i]["longitude"]

        for j in graph.nodes:
            if j <= i:
                continue

            lat2 = graph.nodes[j]["latitude"]
            lon2 = graph.nodes[j]["longitude"]

            distance = geodesic(
                (lat1, lon1),
                (lat2, lon2),
            ).km

            if distance < 5.5:
                graph.add_edge(
                    i,
                    j,
                    distance_km=distance,
                )

    return graph


# ============================================================
# NETWORK REDUCTION
# ============================================================

def select_spatial_nodes(
    graph: nx.Graph,
    center_lat: float,
    center_lon: float,
    count: int,
) -> List[int]:

    if len(graph) == 0:
        return []

    nodes = []

    for node, data in graph.nodes(data=True):
        if "latitude" in data:
            lat = data["latitude"]
            lon = data["longitude"]
        else:
            lat = data.get("lat")
            lon = data.get("lon")

        if lat is None or lon is None:
            continue

        nodes.append(
            (
                node,
                float(lat),
                float(lon),
            )
        )

    if len(nodes) <= count:
        return [x[0] for x in nodes]

    # Farthest-point sampling gives geographic coverage instead
    # of selecting many adjacent road vertices.
    selected = []

    center_distances = [
        (
            node,
            geodesic(
                (center_lat, center_lon),
                (lat, lon),
            ).km,
        )
        for node, lat, lon in nodes
    ]

    first = max(
        center_distances,
        key=lambda x: x[1],
    )[0]

    selected.append(first)

    while len(selected) < count:
        best_node = None
        best_distance = -1

        for node, lat, lon in nodes:
            if node in selected:
                continue

            min_distance = min(
                geodesic(
                    (lat, lon),
                    (
                        graph.nodes[s].get("latitude"),
                        graph.nodes[s].get("longitude"),
                    ),
                ).km
                for s in selected
            )

            if min_distance > best_distance:
                best_distance = min_distance
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)

    return selected


def build_node_matrix(
    graph: nx.Graph,
    selected_nodes: List,
) -> np.ndarray:

    n = len(selected_nodes)
    matrix = np.full((n, n), np.inf)

    for i in range(n):
        matrix[i, i] = 0.0

    for i, source in enumerate(selected_nodes):
        for j, target in enumerate(selected_nodes):
            if i == j:
                continue

            try:
                length = nx.shortest_path_length(
                    graph,
                    source,
                    target,
                    weight="distance_km",
                )

                matrix[i, j] = float(length)

            except nx.NetworkXNoPath:
                pass

    return matrix


# ============================================================
# CANDIDATE ROUTING GRAPH
# ============================================================

def make_candidate_graph(
    distance_matrix: np.ndarray,
    max_edges: int = MAX_QAOA_QUBITS,
) -> nx.Graph:

    n = len(distance_matrix)

    candidate = nx.Graph()

    for i in range(n):
        candidate.add_node(i)

    finite_edges = []

    for i, j in combinations(range(n), 2):
        d = distance_matrix[i, j]

        if np.isfinite(d):
            finite_edges.append(
                (float(d), i, j)
            )

    finite_edges.sort()

    # Start with an MST so the candidate graph stays connected.
    base = nx.Graph()
    base.add_nodes_from(range(n))

    for distance, i, j in finite_edges:
        if nx.utils.union_find.UnionFind(). __class__:
            pass

    # Kruskal MST without depending on private NetworkX internals.
    components = [{i} for i in range(n)]

    def find_component(x):
        for component in components:
            if x in component:
                return component
        return None

    for distance, i, j in finite_edges:
        ci = find_component(i)
        cj = find_component(j)

        if ci is not cj:
            base.add_edge(
                i,
                j,
                distance_km=distance,
            )

            merged = ci | cj
            components.remove(ci)
            components.remove(cj)
            components.append(merged)

        if base.number_of_edges() >= n - 1:
            break

    # Add the cheapest remaining edges until we hit the quantum
    # variable budget.
    for distance, i, j in finite_edges:
        if base.has_edge(i, j):
            continue

        if base.number_of_edges() >= max_edges:
            break

        base.add_edge(
            i,
            j,
            distance_km=distance,
        )

    return base


# ============================================================
# CLASSICAL ROUTING
# ============================================================

def route_distance(
    route: List[int],
    matrix: np.ndarray,
) -> float:

    total = 0.0

    for a, b in zip(route[:-1], route[1:]):
        value = matrix[a, b]

        if not np.isfinite(value):
            return float("inf")

        total += value

    return total


def nearest_neighbor_route(
    matrix: np.ndarray,
) -> List[int]:

    n = len(matrix)
    route = [0]
    remaining = set(range(1, n))

    while remaining:
        current = route[-1]

        reachable = [
            x
            for x in remaining
            if np.isfinite(matrix[current, x])
        ]

        if not reachable:
            return []

        next_node = min(
            reachable,
            key=lambda x: matrix[current, x],
        )

        route.append(next_node)
        remaining.remove(next_node)

    return route


def two_opt(
    route: List[int],
    matrix: np.ndarray,
) -> List[int]:

    if len(route) < 4:
        return route

    best = route[:]
    best_distance = route_distance(best, matrix)

    improved = True

    while improved:
        improved = False

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = (
                    best[:i]
                    + best[i:j + 1][::-1]
                    + best[j + 1:]
                )

                distance = route_distance(
                    candidate,
                    matrix,
                )

                if distance < best_distance:
                    best = candidate
                    best_distance = distance
                    improved = True

    return best


# ============================================================
# QUBO
# ============================================================

def build_path_qubo(
    candidate_graph: nx.Graph,
    start: int,
    end: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, float]:

    edges = list(candidate_graph.edges())

    if len(edges) > MAX_QAOA_QUBITS:
        edges = sorted(
            edges,
            key=lambda e: candidate_graph[e[0]][e[1]]["distance_km"],
        )[:MAX_QAOA_QUBITS]

    q = len(edges)

    if q == 0:
        raise ValueError("No candidate corridors were available.")

    linear = np.zeros(q, dtype=float)
    quadratic = np.zeros((q, q), dtype=float)

    # Distance term.
    for k, (u, v) in enumerate(edges):
        linear[k] += candidate_graph[u][v]["distance_km"]

    # Degree targets.
    target = {}

    for node in candidate_graph.nodes:
        if node == start or node == end:
            target[node] = 1
        else:
            target[node] = 2

    penalty = max(
        10.0,
        float(np.nanmax(linear)) * 2.0,
    )

    incident = {
        node: [
            k
            for k, (u, v) in enumerate(edges)
            if u == node or v == node
        ]
        for node in candidate_graph.nodes
    }

    # P * (sum incident x - target)^2
    for node, edge_indices in incident.items():
        t = target[node]

        for k in edge_indices:
            linear[k] += penalty * (1 - 2 * t)

        for a, b in combinations(edge_indices, 2):
            quadratic[a, b] += 2 * penalty

        linear_node_constant = penalty * (t ** 2)

        # The constant is handled separately.
        # It does not affect the optimizer.
        _ = linear_node_constant

    return edges, linear, quadratic, penalty


def qubo_to_ising(
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> SparsePauliOp:

    n = len(linear)

    constant = 0.0
    z_terms = {}

    def add_z(index: int, coefficient: float):
        z_terms[index] = z_terms.get(index, 0.0) + coefficient

    # x = (1 - Z) / 2
    for i in range(n):
        constant += linear[i] / 2.0
        add_z(i, -linear[i] / 2.0)

    for i in range(n):
        for j in range(i + 1, n):
            coefficient = quadratic[i, j]

            if coefficient == 0:
                continue

            constant += coefficient / 4.0

            add_z(i, -coefficient / 4.0)
            add_z(j, -coefficient / 4.0)

            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"

            z_terms[tuple(label)] = (
                z_terms.get(tuple(label), 0.0)
                + coefficient / 4.0
            )

    paulis = []
    coefficients = []

    # Single Z terms.
    for i, coefficient in list(z_terms.items()):
        if isinstance(i, int):
            label = ["I"] * n
            label[n - 1 - i] = "Z"

            paulis.append("".join(label))
            coefficients.append(coefficient)

    # ZZ terms.
    for label, coefficient in list(z_terms.items()):
        if isinstance(label, tuple):
            paulis.append("".join(label))
            coefficients.append(coefficient)

    # Constant identity term.
    paulis.append("I" * n)
    coefficients.append(constant)

    return SparsePauliOp.from_list(
        list(zip(paulis, coefficients))
    )


# ============================================================
# QUANTUM ROUTING
# ============================================================

def bits_from_key(key, num_bits: int) -> np.ndarray:
    if isinstance(key, int):
        bitstring = format(
            key,
            f"0{num_bits}b",
        )
    else:
        bitstring = str(key).replace(" ", "")

    bitstring = bitstring.zfill(num_bits)

    # Qiskit bitstrings display highest-index qubit first.
    return np.array(
        [int(x) for x in bitstring[::-1]],
        dtype=int,
    )


def selected_edges_to_route(
    candidate_graph: nx.Graph,
    edges: List[Tuple[int, int]],
    bits: np.ndarray,
    start: int,
    end: int,
) -> Optional[List[int]]:

    selected = [
        edges[i]
        for i, bit in enumerate(bits)
        if bit == 1
    ]

    if not selected:
        return None

    subgraph = nx.Graph()
    subgraph.add_edges_from(selected)

    if start not in subgraph or end not in subgraph:
        return None

    degrees = dict(subgraph.degree())

    if degrees.get(start, 0) != 1:
        return None

    if degrees.get(end, 0) != 1:
        return None

    for node in candidate_graph.nodes:
        if node in (start, end):
            continue

        if degrees.get(node, 0) != 2:
            return None

    if not nx.is_connected(subgraph):
        return None

    if subgraph.number_of_nodes() != candidate_graph.number_of_nodes():
        return None

    try:
        route = nx.shortest_path(
            subgraph,
            start,
            end,
        )
    except nx.NetworkXNoPath:
        return None

    if len(route) != candidate_graph.number_of_nodes():
        return None

    return route


def run_qaoa(
    candidate_graph: nx.Graph,
    start: int,
    end: int,
) -> Dict:

    edges, linear, quadratic, penalty = build_path_qubo(
        candidate_graph,
        start,
        end,
    )

    if len(edges) > MAX_QAOA_QUBITS:
        raise ValueError(
            "Quantum model exceeded the configured qubit budget."
        )

    hamiltonian = qubo_to_ising(
        linear,
        quadratic,
    )

    sampler = StatevectorSampler(
        default_shots=QAOA_SHOTS,
        seed=42,
    )

    optimizer = COBYLA(
        maxiter=QAOA_MAXITER,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=QAOA_REPS,
    )

    start_time = time.perf_counter()

    result = qaoa.compute_minimum_eigenvalue(
        hamiltonian
    )

    elapsed = time.perf_counter() - start_time

    distribution = result.eigenstate

    probabilities = {}

    if distribution is not None:
        for key, value in distribution.items():
            probabilities[str(key)] = max(
                0.0,
                float(np.real(value)),
            )

    probability_sum = sum(probabilities.values())

    if probability_sum > 0:
        probabilities = {
            key: value / probability_sum
            for key, value in probabilities.items()
        }

    ranked_states = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    quantum_route = None
    quantum_probability = 0.0
    quantum_bits = None

    # Inspect the highest-probability states and select the
    # highest-probability valid route.
    for key, probability in ranked_states:
        bits = bits_from_key(
            key,
            len(edges),
        )

        route = selected_edges_to_route(
            candidate_graph,
            edges,
            bits,
            start,
            end,
        )

        if route is not None:
            quantum_route = route
            quantum_probability = probability
            quantum_bits = bits
            break

    return {
        "route": quantum_route,
        "probabilities": probabilities,
        "edges": edges,
        "bits": quantum_bits,
        "probability": quantum_probability,
        "runtime": elapsed,
        "qubo_qubits": len(edges),
        "penalty": penalty,
        "eigenvalue": float(np.real(result.eigenvalue)),
        "optimizer_evals": getattr(
            result,
            "cost_function_evals",
            None,
        ),
    }


# ============================================================
# OBSTACLE MODEL
# ============================================================

def apply_obstacles(
    graph: nx.Graph,
    enabled: bool,
    seed: int = 42,
) -> Tuple[nx.Graph, List[Tuple]]:

    if not enabled:
        return graph.copy(), []

    rng = np.random.default_rng(seed)

    result = graph.copy()

    edges = list(result.edges())

    if len(edges) < 3:
        return result, []

    count = max(
        1,
        min(
            len(edges) // 8,
            4,
        ),
    )

    indices = rng.choice(
        len(edges),
        size=count,
        replace=False,
    )

    blocked = []

    for index in indices:
        edge = edges[int(index)]
        blocked.append(edge)
        result.remove_edge(*edge)

    return result, blocked


# ============================================================
# NODE COORDINATES
# ============================================================

def graph_node_coordinates(
    graph: nx.Graph,
    selected_nodes: List,
) -> List[Dict]:

    records = []

    for index, node in enumerate(selected_nodes):
        data = graph.nodes[node]

        lat = data.get("latitude")
        lon = data.get("longitude")

        if lat is None:
            lat = data.get("lat")

        if lon is None:
            lon = data.get("lon")

        records.append(
            {
                "node": index,
                "source_id": node,
                "latitude": float(lat),
                "longitude": float(lon),
                "type": (
                    "Distribution Hub"
                    if index == 0
                    else "Delivery Point"
                ),
            }
        )

    return records


# ============================================================
# ROUTE GEOMETRY
# ============================================================

def route_geometry(
    route: List[int],
    selected_nodes: List,
    graph: nx.Graph,
) -> List[Dict]:

    records = []

    for sequence, (a, b) in enumerate(
        zip(route[:-1], route[1:])
    ):
        source = selected_nodes[a]
        target = selected_nodes[b]

        try:
            path = nx.shortest_path(
                graph,
                source,
                target,
                weight="distance_km",
            )
        except nx.NetworkXNoPath:
            path = [source, target]

        for road_node in path:
            data = graph.nodes[road_node]

            lat = data.get("latitude")
            lon = data.get("longitude")

            if lat is None:
                lat = data.get("lat")

            if lon is None:
                lon = data.get("lon")

            records.append(
                {
                    "route_segment": sequence,
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )

    return records


# ============================================================
# MAIN OPTIMIZATION PIPELINE
# ============================================================

def optimize_disaster_route(
    location_query: str,
    node_count: int,
    obstacles: bool,
) -> Dict:

    total_start = time.perf_counter()

    location = geocode_location(location_query)

    if location is None:
        raise ValueError(
            f"Location not found: {location_query}"
        )

    latitude = location["latitude"]
    longitude = location["longitude"]

    osm_data = fetch_osm_network(
        latitude,
        longitude,
    )

    source_type = "OpenStreetMap"

    if osm_data is not None:
        road_graph = build_osm_graph(osm_data)
    else:
        road_graph = build_fallback_graph(
            latitude,
            longitude,
        )
        source_type = "Synthetic fallback network"

    if road_graph.number_of_nodes() < node_count:
        road_graph = build_fallback_graph(
            latitude,
            longitude,
        )
        source_type = "Synthetic fallback network"

    disrupted_graph, blocked_edges = apply_obstacles(
        road_graph,
        obstacles,
    )

    selected_raw = select_spatial_nodes(
        disrupted_graph,
        latitude,
        longitude,
        node_count,
    )

    if len(selected_raw) < node_count:
        raise ValueError(
            "The local network did not provide enough usable nodes."
        )

    # Ensure all selected nodes belong to the same connected
    # component when possible.
    components = sorted(
        nx.connected_components(disrupted_graph),
        key=len,
        reverse=True,
    )

    largest_component = components[0]

    selected_raw = [
        node
        for node in selected_raw
        if node in largest_component
    ]

    if len(selected_raw) < node_count:
        selected_raw = list(largest_component)[:node_count]

    if len(selected_raw) < node_count:
        raise ValueError(
            "The disrupted network is too fragmented."
        )

    selected_raw = selected_raw[:node_count]

    matrix = build_node_matrix(
        disrupted_graph,
        selected_raw,
    )

    # Replace unreachable pairs using geographic distance.
    # This gives the optimization model a finite fallback cost.
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue

            if not np.isfinite(matrix[i, j]):
                a = disrupted_graph.nodes[selected_raw[i]]
                b = disrupted_graph.nodes[selected_raw[j]]

                matrix[i, j] = (
                    geodesic(
                        (
                            a.get("latitude"),
                            a.get("longitude"),
                        ),
                        (
                            b.get("latitude"),
                            b.get("longitude"),
                        ),
                    ).km
                    * 4.0
                )

    candidate_graph = make_candidate_graph(
        matrix,
        max_edges=MAX_QAOA_QUBITS,
    )

    # Start at the first node. Finish at the farthest node from it.
    start = 0

    end = max(
        range(1, node_count),
        key=lambda x: matrix[start, x],
    )

    # Classical baseline.
    classical_start = time.perf_counter()

    classical_route = nearest_neighbor_route(matrix)

    if not classical_route:
        raise ValueError(
            "A classical baseline route could not be constructed."
        )

    classical_route = two_opt(
        classical_route,
        matrix,
    )

    classical_time = (
        time.perf_counter()
        - classical_start
    )

    classical_distance = route_distance(
        classical_route,
        matrix,
    )

    # Quantum stage.
    quantum = run_qaoa(
        candidate_graph,
        start,
        end,
    )

    quantum_route = quantum["route"]

    # QAOA samples are probabilistic. If none of the sampled
    # states satisfies the full connectivity condition, use
    # the classical feasible route as the safety fallback.
    if quantum_route is None:
        quantum_route = classical_route
        quantum_status = (
            "QAOA completed, but no sampled state produced "
            "a fully connected feasible route."
        )
    else:
        quantum_status = (
            "QAOA produced a feasible multi-stop route."
        )

    quantum_distance = route_distance(
        quantum_route,
        matrix,
    )

    if (
        np.isfinite(classical_distance)
        and classical_distance > 0
    ):
        improvement = (
            (
                classical_distance
                - quantum_distance
            )
            / classical_distance
            * 100.0
        )
    else:
        improvement = 0.0

    total_time = (
        time.perf_counter()
        - total_start
    )

    coordinates = graph_node_coordinates(
        disrupted_graph,
        selected_raw,
    )

    route_coordinates = route_geometry(
        quantum_route,
        selected_raw,
        disrupted_graph,
    )

    return {
        "location": location,
        "source_type": source_type,
        "road_graph": disrupted_graph,
        "selected_nodes": selected_raw,
        "coordinates": coordinates,
        "route_coordinates": route_coordinates,
        "matrix": matrix,
        "candidate_graph": candidate_graph,
        "blocked_edges": blocked_edges,
        "classical_route": classical_route,
        "quantum_route": quantum_route,
        "classical_distance": classical_distance,
        "quantum_distance": quantum_distance,
        "improvement": improvement,
        "classical_time": classical_time,
        "quantum_time": quantum["runtime"],
        "total_time": total_time,
        "probabilities": quantum["probabilities"],
        "quantum_probability": quantum["probability"],
        "qubo_qubits": quantum["qubo_qubits"],
        "optimizer_evals": quantum["optimizer_evals"],
        "eigenvalue": quantum["eigenvalue"],
        "quantum_status": quantum_status,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🚑 Mission Control")

    location_query = st.text_input(
        "Emergency location",
        value="Charlotte, North Carolina",
        placeholder="Tokyo, Japan",
    )

    node_count = st.slider(
        "Active delivery nodes",
        min_value=4,
        max_value=12,
        value=8,
        step=1,
    )

    st.subheader("Scenario")

    road_closures = st.checkbox(
        "Simulate road closures",
        value=False,
    )

    bridge_washouts = st.checkbox(
        "Simulate bridge washouts",
        value=False,
    )

    optimize_button = st.button(
        "⚛️ RUN QUANTUM OPTIMIZATION",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "Quantum engine: QAOA + StatevectorSampler"
    )

    st.caption(
        f"QAOA qubit budget: ≤ {MAX_QAOA_QUBITS}"
    )

    st.caption(
        "Mapping: OpenStreetMap / Overpass"
    )


# ============================================================
# HEADER
# ============================================================

st.title("⚛️ Quantum Relief Router")

st.markdown(
    "Adaptive disaster-response routing using geospatial networks, "
    "QAOA optimization, and classical route validation."
)

if optimize_button:

    obstacle_enabled = (
        road_closures or bridge_washouts
    )

    with st.spinner(
        "Fetching local network and executing QAOA..."
    ):
        try:
            result = optimize_disaster_route(
                location_query=location_query,
                node_count=node_count,
                obstacles=obstacle_enabled,
            )

            st.session_state.result = result
            st.session_state.last_location = (
                result["location"]["display_name"]
            )

            st.session_state.view_state = {
                "latitude": result["location"]["latitude"],
                "longitude": result["location"]["longitude"],
                "zoom": 10,
                "pitch": 45,
            }

        except Exception as exc:
            st.error(
                f"Optimization failed: {exc}"
            )


result = st.session_state.result


# ============================================================
# EMPTY STATE
# ============================================================

if result is None:

    st.info(
        "Enter an emergency location and run the quantum optimizer "
        "to generate a relief corridor."
    )

    st.subheader("System Architecture")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "1. Geocode",
            "Nominatim",
        )

    with col2:
        st.metric(
            "2. Network",
            "OSM + NetworkX",
        )

    with col3:
        st.metric(
            "3. Optimize",
            "QAOA",
        )

    with col4:
        st.metric(
            "4. Validate",
            "Classical",
        )

    st.stop()


# ============================================================
# METRICS
# ============================================================

st.caption(
    f"Network source: {result['source_type']} • "
    f"{result['location']['display_name']}"
)

metric_cols = st.columns(5)

with metric_cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">QUANTUM ROUTE</div>
            <div class="metric-value">
                {result['quantum_distance']:.2f} km
            </div>
            <div class="metric-detail">
                {len(result['quantum_route'])} stops
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CLASSICAL BASELINE</div>
            <div class="metric-value">
                {result['classical_distance']:.2f} km
            </div>
            <div class="metric-detail">
                Network heuristic
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">DISTANCE DELTA</div>
            <div class="metric-value">
                {result['improvement']:+.2f}%
            </div>
            <div class="metric-detail">
                Relative to baseline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">QAOA EXECUTION</div>
            <div class="metric-value">
                {result['quantum_time']:.2f}s
            </div>
            <div class="metric-detail">
                {result['qubo_qubits']} qubits
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_cols[4]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">OPTIMAL STATE</div>
            <div class="metric-value">
                {result['quantum_probability'] * 100:.1f}%
            </div>
            <div class="metric-detail">
                Sample probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STATUS
# ============================================================

if result["quantum_route"] is not None:
    st.success(result["quantum_status"])
else:
    st.warning(result["quantum_status"])


# ============================================================
# MAP
# ============================================================

st.subheader("🌐 Global Response Map")

node_df = pd.DataFrame(
    result["coordinates"]
)

route_df = pd.DataFrame(
    result["route_coordinates"]
)

if not route_df.empty:

    route_layer = pdk.Layer(
        "PathLayer",
        data=route_df,
        get_path=[
            "longitude",
            "latitude",
        ],
        get_width=6,
        width_min_pixels=4,
        pickable=True,
    )

else:
    route_layer = None


point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=node_df,
    get_position=[
        "longitude",
        "latitude",
    ],
    get_radius=900,
    radius_min_pixels=6,
    radius_max_pixels=20,
    get_fill_color=[
        30,
        144,
        255,
        210,
    ],
    pickable=True,
)

layers = [
    point_layer
]

if route_layer is not None:
    layers.append(route_layer)

view_state = pdk.ViewState(
    latitude=st.session_state.view_state["latitude"],
    longitude=st.session_state.view_state["longitude"],
    zoom=st.session_state.view_state["zoom"],
    pitch=st.session_state.view_state["pitch"],
    bearing=0,
)

deck = pdk.Deck(
    map_style=None,
    initial_view_state=view_state,
    layers=layers,
    tooltip={
        "html": (
            "<b>{type}</b><br/>"
            "Node: {node}<br/>"
            "Lat: {latitude}<br/>"
            "Lon: {longitude}"
        ),
        "style": {
            "backgroundColor": "#111827",
            "color": "white",
        },
    },
)

st.pydeck_chart(
    deck,
    use_container_width=True,
)


# ============================================================
# ROUTE DETAILS
# ============================================================

map_col1, map_col2 = st.columns(2)

with map_col1:
    st.subheader("Optimized Relief Sequence")

    route_display = [
        f"Node {node + 1}"
        for node in result["quantum_route"]
    ]

    st.markdown(
        '<div class="route-box">'
        + " → ".join(route_display)
        + "</div>",
        unsafe_allow_html=True,
    )

with map_col2:
    st.subheader("Classical Comparison")

    classical_display = [
        f"Node {node + 1}"
        for node in result["classical_route"]
    ]

    st.markdown(
        '<div class="route-box">'
        + " → ".join(classical_display)
        + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

st.subheader("📊 Quantum State Distribution")

probabilities = result["probabilities"]

if probabilities:

    probability_df = (
        pd.DataFrame(
            [
                {
                    "State": state,
                    "Probability": probability,
                }
                for state, probability in probabilities.items()
            ]
        )
        .sort_values(
            "Probability",
            ascending=False,
        )
        .head(20)
    )

    probability_df["State"] = (
        probability_df["State"]
        .astype(str)
    )

    probability_df = probability_df.set_index(
        "State"
    )

    st.bar_chart(
        probability_df,
        y="Probability",
        height=360,
    )

    st.caption(
        "The distribution represents measured QAOA states from "
        "the optimized quantum circuit. Higher probability states "
        "are favored by the variational optimization."
    )

else:
    st.warning(
        "The QAOA sampler did not return a probability distribution."
    )


# ============================================================
# TELEMETRY
# ============================================================

st.subheader("⚙️ Optimization Telemetry")

telemetry = pd.DataFrame(
    [
        {
            "Metric": "Total processing time",
            "Value": f"{result['total_time']:.3f} s",
        },
        {
            "Metric": "QAOA execution time",
            "Value": f"{result['quantum_time']:.3f} s",
        },
        {
            "Metric": "Classical baseline time",
            "Value": f"{result['classical_time']:.3f} s",
        },
        {
            "Metric": "QAOA qubits",
            "Value": str(result["qubo_qubits"]),
        },
        {
            "Metric": "QAOA repetitions",
            "Value": str(QAOA_REPS),
        },
        {
            "Metric": "QAOA shots",
            "Value": str(QAOA_SHOTS),
        },
        {
            "Metric": "QAOA optimizer evaluations",
            "Value": (
                str(result["optimizer_evals"])
                if result["optimizer_evals"] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "QAOA eigenvalue",
            "Value": f"{result['eigenvalue']:.4f}",
        },
        {
            "Metric": "Blocked road segments",
            "Value": str(
                len(result["blocked_edges"])
            ),
        },
    ]
)

st.dataframe(
    telemetry,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DATA TABLE
# ============================================================

with st.expander("View delivery-node coordinates"):

    st.dataframe(
        node_df[
            [
                "node",
                "type",
                "latitude",
                "longitude",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Quantum Relief Router • Congressional App Challenge prototype • "
    "QAOA runs locally through Qiskit's StatevectorSampler."
)
