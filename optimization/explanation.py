from __future__ import annotations

import networkx as nx

from .graph import shortest_leg
from .problem import Route, Vehicle


def _route_cost(graph: nx.Graph, depot: str, start: str, stops: list) -> tuple[float, float, float]:
    current = start
    distance = travel = service = 0.0
    for stop in stops:
        _, leg_distance, leg_time = shortest_leg(graph, current, stop.node)
        distance += leg_distance
        travel += leg_time
        service += stop.service_minutes
        current = stop.node
    _, leg_distance, leg_time = shortest_leg(graph, current, depot)
    return distance + leg_distance, travel + leg_time, service


def explain_assignments(
    graph: nx.Graph, depot: str, routes: list[Route], vehicles: list[Vehicle], solver: str = "",
    live_travel_matrix: dict[tuple[str, str], dict[str, float]] | None = None,
) -> list[dict]:
    """Explain each chosen bin assignment and compare truck approach ETAs."""
    vehicle_by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    route_by_id = {route.vehicle_id: route for route in routes}
    explanations = []

    for chosen_route in routes:
        chosen_vehicle = vehicle_by_id.get(chosen_route.vehicle_id)
        if chosen_vehicle is None:
            continue
        start = chosen_vehicle.current_node or depot
        elapsed = 0.0
        current = start
        current_load = chosen_vehicle.current_load_kg
        for stop in chosen_route.stops:
            _, leg_distance, leg_time = shortest_leg(graph, current, stop.node)
            elapsed += leg_time
            current_load += stop.demand_kg
            candidates = []
            for vehicle in vehicles:
                if not vehicle.available:
                    continue
                vehicle_route = route_by_id.get(vehicle.vehicle_id, Route(vehicle_id=vehicle.vehicle_id))
                base_stops = list(vehicle_route.stops)
                if vehicle.vehicle_id == chosen_vehicle.vehicle_id:
                    base_stops.remove(stop)
                start_node = vehicle.current_node or depot
                _, _, simulated_approach_minutes = shortest_leg(graph, start_node, stop.node)
                live_leg = (live_travel_matrix or {}).get((vehicle.vehicle_id, stop.bin_id))
                approach_minutes = live_leg["duration_minutes"] if live_leg else simulated_approach_minutes
                base_distance, base_travel, base_service = _route_cost(graph, depot, start_node, base_stops)
                best = None
                best_any = None
                for position in range(len(base_stops) + 1):
                    proposed = base_stops[:position] + [stop] + base_stops[position:]
                    load_after = vehicle.current_load_kg + sum(item.demand_kg for item in proposed)
                    distance, travel, service = _route_cost(graph, depot, start_node, proposed)
                    duration = travel + service
                    feasible = (
                        load_after <= vehicle.capacity_kg + 1e-8
                        and (vehicle.max_distance_km is None or distance <= vehicle.max_distance_km + 1e-8)
                        and (vehicle.max_duration_minutes is None or duration <= vehicle.max_duration_minutes + 1e-8)
                    )
                    added_minutes = duration - base_travel - base_service
                    added_distance = distance - base_distance
                    estimate = {
                        "vehicle_id": vehicle.vehicle_id,
                        "feasible": feasible,
                        "approach_minutes": round(approach_minutes, 2),
                        "estimated_added_minutes": round(added_minutes, 2),
                        "estimated_added_distance_km": round(added_distance, 3),
                        "load_after_kg": round(load_after, 2),
                        "capacity_remaining_after_kg": round(vehicle.capacity_kg - load_after, 2),
                    }
                    if best_any is None or added_minutes < best_any["estimated_added_minutes"]:
                        best_any = estimate
                    if feasible and (best is None or added_minutes < best["estimated_added_minutes"]):
                        best = estimate
                if best is None:
                    rejected = best_any or {
                        "vehicle_id": vehicle.vehicle_id,
                        "feasible": False,
                        "approach_minutes": round(approach_minutes, 2),
                    }
                    rejected["feasible"] = False
                    rejected["infeasible_reason"] = (
                        "payload capacity exceeded" if rejected.get("load_after_kg", 0) > vehicle.capacity_kg + 1e-8 else
                        "route distance limit exceeded" if vehicle.max_distance_km is not None and base_distance + rejected.get("estimated_added_distance_km", 0) > vehicle.max_distance_km + 1e-8 else
                        "route time limit exceeded" if vehicle.max_duration_minutes is not None and base_travel + base_service + rejected.get("estimated_added_minutes", 0) > vehicle.max_duration_minutes + 1e-8 else
                        "no feasible insertion in planned route"
                    )
                    candidates.append(rejected)
                else:
                    candidates.append(best)

            selected = next(
                (item for item in candidates if item["vehicle_id"] == chosen_vehicle.vehicle_id),
                {"vehicle_id": chosen_vehicle.vehicle_id, "feasible": False},
            )
            node_data = graph.nodes[start]
            explanations.append({
                "bin_id": stop.bin_id,
                "urgency": stop.urgency,
                "chosen_vehicle_id": chosen_vehicle.vehicle_id,
                "mock_gps": {"node": start, "latitude": node_data.get("latitude"), "longitude": node_data.get("longitude")},
                "estimated_arrival_minutes_from_route_start": round(elapsed, 2),
                "leg_distance_from_previous_stop_km": round(leg_distance, 3),
                "demand_kg": round(stop.demand_kg, 2),
                "load_after_pickup_kg": round(current_load, 2),
                "capacity_remaining_after_pickup_kg": round(chosen_vehicle.capacity_kg - current_load, 2),
                "chosen_vehicle_insertion_estimate": selected,
                "vehicle_insertion_estimates": candidates,
                "decision_basis": (
                    "Truck is selected by Google Routes traffic-aware ETA from its mock GPS location, among trucks with enough remaining payload. Pickup order is then optimized within each truck's assigned stops."
                    if live_travel_matrix and (chosen_vehicle.vehicle_id, stop.bin_id) in live_travel_matrix else
                    "Truck is selected by traffic-weighted simulated road time from its mock GPS location, among trucks with enough remaining payload. Pickup order is then optimized within each truck's assigned stops."
                    if solver.startswith("ortools") else
                    "Deterministic heuristic selected the nearest currently feasible vehicle, prioritizing higher urgency first."
                ),
            })
            elapsed += stop.service_minutes
            current = stop.node

    return explanations
