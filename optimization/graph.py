from __future__ import annotations

import math
import networkx as nx


def build_grid_graph(rows: int = 8, cols: int = 8, seed: int = 42, directed: bool = True) -> nx.Graph:
    """Create a deterministic simulated road network; edge weights are road, not aerial, distances."""
    if rows < 2 or cols < 2:
        raise ValueError("grid needs at least two rows and columns")
    graph = nx.DiGraph() if directed else nx.Graph()
    for r in range(rows):
        for c in range(cols):
            graph.add_node(f"n{r}_{c}", latitude=12.90 + r * .003, longitude=77.55 + c * .003, simulated=True)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if rr >= rows or cc >= cols:
                    continue
                a, b = f"n{r}_{c}", f"n{rr}_{cc}"
                # Road links deliberately vary by coordinates but are reproducible.
                distance = round(0.34 + ((r * 17 + c * 7 + seed) % 8) * .015, 3)
                minutes = round(distance / 24 * 60, 3)
                graph.add_edge(a, b, distance_km=distance, travel_minutes=minutes)
                graph.add_edge(b, a, distance_km=round(distance * 1.03, 3), travel_minutes=round(minutes * 1.03, 3))
    validate_graph(graph)
    return graph


def validate_graph(graph: nx.Graph) -> None:
    if not graph.nodes:
        raise ValueError("road graph is empty")
    for _, _, data in graph.edges(data=True):
        for field in ("distance_km", "travel_minutes"):
            value = data.get(field)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"edge has invalid {field}")


def shortest_leg(graph: nx.Graph, source: str, target: str) -> tuple[list[str], float, float]:
    if source not in graph or target not in graph:
        raise ValueError("unknown road node")
    try:
        path = nx.shortest_path(graph, source, target, weight="distance_km", method="dijkstra")
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"unreachable: {source} -> {target}") from exc
    distance = sum(graph[u][v]["distance_km"] for u, v in zip(path, path[1:]))
    minutes = sum(graph[u][v]["travel_minutes"] for u, v in zip(path, path[1:]))
    return path, distance, minutes
