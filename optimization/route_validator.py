from __future__ import annotations

import math
import networkx as nx
from .problem import Route, Stop, Vehicle


def validate_routes(graph: nx.Graph, depot: str, routes: list[Route], required_stops: list[Stop], vehicles: list[Vehicle]) -> dict:
    errors: list[str] = []
    vehicle_by_id = {v.vehicle_id: v for v in vehicles}
    required = {s.bin_id: s for s in required_stops}
    visited: list[str] = []
    for route in routes:
        vehicle = vehicle_by_id.get(route.vehicle_id)
        if not vehicle:
            errors.append(f"unknown vehicle {route.vehicle_id}"); continue
        if not route.road_path:
            if route.stops: errors.append(f"{route.vehicle_id}: non-empty route has no road path")
            continue
        expected_start = vehicle.current_node or depot
        if route.road_path[0] != expected_start or route.road_path[-1] != depot:
            errors.append(f"{route.vehicle_id}: route must start at current location and end at depot")
        distance = travel = 0.0
        for a, b in zip(route.road_path, route.road_path[1:]):
            if not graph.has_edge(a, b): errors.append(f"{route.vehicle_id}: invalid edge {a}->{b}"); continue
            distance += graph[a][b]["distance_km"]; travel += graph[a][b]["travel_minutes"]
        demand = sum(s.demand_kg for s in route.stops); service = sum(s.service_minutes for s in route.stops)
        if vehicle.current_load_kg + demand > vehicle.capacity_kg + 1e-8: errors.append(f"{route.vehicle_id}: capacity exceeded")
        if vehicle.max_distance_km is not None and distance > vehicle.max_distance_km + 1e-8: errors.append(f"{route.vehicle_id}: distance limit exceeded")
        if vehicle.max_duration_minutes is not None and travel + service + route.waiting_minutes > vehicle.max_duration_minutes + 1e-8: errors.append(f"{route.vehicle_id}: duration limit exceeded")
        if not math.isclose(distance, route.distance_km, abs_tol=1e-6): errors.append(f"{route.vehicle_id}: reported distance differs from graph")
        if not math.isclose(travel, route.travel_minutes, abs_tol=1e-6): errors.append(f"{route.vehicle_id}: reported travel time differs from graph")
        visited.extend(s.bin_id for s in route.stops)
    duplicates = {x for x in visited if visited.count(x) > 1}
    if duplicates: errors.append(f"duplicate service: {sorted(duplicates)}")
    missing = sorted(set(required) - set(visited))
    if missing: errors.append(f"unserved required bins: {missing}")
    return {"valid": not errors, "errors": errors, "served_bin_ids": sorted(set(visited)), "missing_bin_ids": missing}
