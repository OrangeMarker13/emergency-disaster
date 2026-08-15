"""
Map data and geographic utilities for Quantum Relief Router.

This module handles:
- Address/geographic coordinate lookup through Nominatim
- OpenStreetMap road-network retrieval through Overpass
- Conversion of OSM road data into a NetworkX graph
- Nearest-node lookup
- A synthetic fallback network when external map services fail
"""

from __future__ import annotations

import math
import time
from typing import Any

import networkx as nx
import requests

from config import CONFIG


def _haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate distance between two geographic coordinates.

    Returns:
        Distance in meters.
    """

    earth_radius = 6_371_000.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def geocode_location(query: str) -> dict[str, Any]:
    """
    Convert an address or place name into geographic coordinates.

    Args:
        query: Address, city, landmark, or other geographic search.

    Returns:
        Dictionary containing:
        - latitude
        - longitude
        - display_name

    Raises:
        ValueError: If no location is found.
        RuntimeError: If the geocoding service fails.
    """

    query = query.strip()

    if not query:
        raise ValueError("Location cannot be empty.")

    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
    }

    headers = {
        "User-Agent": CONFIG.mapping.user_agent,
    }

    try:
        response = requests.get(
            CONFIG.mapping.nominatim_url,
            params=params,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        results = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to contact the geocoding service: {exc}"
        ) from exc

    if not results:
        raise ValueError(f"No location found for '{query}'.")

    result = results[0]

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result.get("display_name", query),
    }


def _build_overpass_query(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> str:
    """
    Build an Overpass query for drivable roads.
    """

    return f"""
    [out:json][timeout:30];

    (
        way["highway"]["highway"!~"footway|path|cycleway|steps|pedestrian|construction|proposed"]
        (around:{radius_m},{latitude},{longitude});
    );

    out body;
    >;
    out skel qt;
    """


def fetch_osm_network(
    latitude: float,
    longitude: float,
    radius_m: int | None = None,
) -> nx.Graph:
    """
    Download a road network surrounding a geographic coordinate.

    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        radius_m: Search radius in meters.

    Returns:
        NetworkX graph containing road nodes and edges.

    Raises:
        RuntimeError: If OpenStreetMap data cannot be retrieved.
        ValueError: If insufficient road data is returned.
    """

    if radius_m is None:
        radius_m = CONFIG.mapping.default_radius_m

    query = _build_overpass_query(
        latitude,
        longitude,
        radius_m,
    )

    try:
        response = requests.post(
            CONFIG.mapping.overpass_url,
            data=query,
            headers={
                "User-Agent": CONFIG.mapping.user_agent,
            },
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to retrieve OpenStreetMap data: {exc}"
        ) from exc

    elements = data.get("elements", [])

    if not elements:
        raise ValueError("OpenStreetMap returned no road data.")

    graph = nx.Graph()

    nodes: dict[int, tuple[float, float]] = {}

    for element in elements:
        if element.get("type") != "node":
            continue

        node_id = element.get("id")
        lat = element.get("lat")
        lon = element.get("lon")

        if node_id is None or lat is None or lon is None:
            continue

        nodes[int(node_id)] = (
            float(lat),
            float(lon),
        )

    for element in elements:
        if element.get("type") != "way":
            continue

        way_nodes = element.get("nodes", [])
        tags = element.get("tags", {})

        if len(way_nodes) < 2:
            continue

        highway_type = tags.get("highway", "unknown")
        road_name = tags.get("name", "Unnamed road")

        for start_id, end_id in zip(
            way_nodes[:-1],
            way_nodes[1:],
        ):
            if start_id not in nodes or end_id not in nodes:
                continue

            start_lat, start_lon = nodes[start_id]
            end_lat, end_lon = nodes[end_id]

            distance = _haversine_distance(
                start_lat,
                start_lon,
                end_lat,
                end_lon,
            )

            graph.add_node(
                start_id,
                latitude=start_lat,
                longitude=start_lon,
            )

            graph.add_node(
                end_id,
                latitude=end_lat,
                longitude=end_lon,
            )

            graph.add_edge(
                start_id,
                end_id,
                distance=distance,
                road_name=road_name,
                highway_type=highway_type,
                blocked=False,
            )

    if graph.number_of_nodes() < 2:
        raise ValueError(
            "Insufficient road-network data was returned."
        )

    graph = _limit_graph_size(graph)

    return graph


def _limit_graph_size(graph: nx.Graph) -> nx.Graph:
    """
    Limit graph size while retaining nodes near the network center.

    This prevents extremely large OpenStreetMap networks from making
    the optimization stage unnecessarily expensive.
    """

    max_nodes = CONFIG.mapping.max_nodes

    if graph.number_of_nodes() <= max_nodes:
        return graph

    degrees = dict(graph.degree())

    ranked_nodes = sorted(
        degrees,
        key=degrees.get,
        reverse=True,
    )[:max_nodes]

    limited_graph = graph.subgraph(ranked_nodes).copy()

    return limited_graph


def find_nearest_node(
    graph: nx.Graph,
    latitude: float,
    longitude: float,
) -> Any:
    """
    Find the graph node closest to a geographic coordinate.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph contains no nodes.")

    nearest_node = None
    nearest_distance = float("inf")

    for node, data in graph.nodes(data=True):
        node_lat = data.get("latitude")
        node_lon = data.get("longitude")

        if node_lat is None or node_lon is None:
            continue

        distance = _haversine_distance(
            latitude,
            longitude,
            node_lat,
            node_lon,
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_node = node

    if nearest_node is None:
        raise ValueError("Graph contains no geographic node data.")

    return nearest_node


def graph_to_map_data(
    graph: nx.Graph,
) -> list[dict[str, Any]]:
    """
    Convert graph edges into data suitable for map visualization.
    """

    map_data = []

    for start, end, data in graph.edges(data=True):

        start_data = graph.nodes[start]
        end_data = graph.nodes[end]

        map_data.append(
            {
                "start_lat": start_data["latitude"],
                "start_lon": start_data["longitude"],
                "end_lat": end_data["latitude"],
                "end_lon": end_data["longitude"],
                "road_name": data.get(
                    "road_name",
                    "Unnamed road",
                ),
                "distance": data.get(
                    "distance",
                    0.0,
                ),
                "blocked": data.get(
                    "blocked",
                    False,
                ),
            }
        )

    return map_data


def create_fallback_network(
    latitude: float,
    longitude: float,
    grid_size: int = 5,
    spacing: float = 0.002,
) -> nx.Graph:
    """
    Create a synthetic road network.

    This provides a functional demonstration when external
    OpenStreetMap services are unavailable.
    """

    graph = nx.Graph()

    node_id = 0

    for row in range(grid_size):
        for col in range(grid_size):

            node_lat = latitude + (
                row - grid_size // 2
            ) * spacing

            node_lon = longitude + (
                col - grid_size // 2
            ) * spacing

            graph.add_node(
                node_id,
                latitude=node_lat,
                longitude=node_lon,
                synthetic=True,
            )

            node_id += 1

    for row in range(grid_size):
        for col in range(grid_size):

            current = row * grid_size + col

            if col < grid_size - 1:
                right = current + 1

                start = graph.nodes[current]
                end = graph.nodes[right]

                distance = _haversine_distance(
                    start["latitude"],
                    start["longitude"],
                    end["latitude"],
                    end["longitude"],
                )

                graph.add_edge(
                    current,
                    right,
                    distance=distance,
                    road_name="Synthetic Road",
                    highway_type="simulated",
                    blocked=False,
                    synthetic=True,
                )

            if row < grid_size - 1:
                below = current + grid_size

                start = graph.nodes[current]
                end = graph.nodes[below]

                distance = _haversine_distance(
                    start["latitude"],
                    start["longitude"],
                    end["latitude"],
                    end["longitude"],
                )

                graph.add_edge(
                    current,
                    below,
                    distance=distance,
                    road_name="Synthetic Road",
                    highway_type="simulated",
                    blocked=False,
                    synthetic=True,
                )

    return graph


def load_network(
    latitude: float,
    longitude: float,
    radius_m: int | None = None,
    use_fallback: bool = True,
) -> tuple[nx.Graph, bool]:
    """
    Load a road network.

    The function first attempts to retrieve real OpenStreetMap data.
    If retrieval fails and fallback is enabled, it creates a synthetic
    network.

    Returns:
        (graph, is_fallback)
    """

    try:
        graph = fetch_osm_network(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
        )

        return graph, False

    except Exception:

        if not use_fallback:
            raise

        graph = create_fallback_network(
            latitude=latitude,
            longitude=longitude,
        )

        return graph, True
