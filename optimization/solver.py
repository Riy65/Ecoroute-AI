from __future__ import annotations

import time
import networkx as nx
from .graph import shortest_leg
from .problem import Route, RoutingResult, Stop, Vehicle
from .route_validator import validate_routes

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:  # Explicit fallback for environments where OR-Tools is unavailable.
    ORTOOLS_AVAILABLE = False


def _append_leg(route: Route, graph: nx.Graph, source: str, target: str) -> None:
    path, distance, minutes = shortest_leg(graph, source, target)
    route.road_path.extend(path if not route.road_path else path[1:])
    route.distance_km += distance; route.travel_minutes += minutes


def _can_append(route: Route, stop: Stop, vehicle: Vehicle, graph: nx.Graph, depot: str) -> bool:
    if vehicle.current_load_kg + sum(x.demand_kg for x in route.stops) + stop.demand_kg > vehicle.capacity_kg + 1e-9:
        return False
    here = route.road_path[-1] if route.road_path else (vehicle.current_node or depot)
    try:
        _, out_d, out_t = shortest_leg(graph, here, stop.node)
        _, back_d, back_t = shortest_leg(graph, stop.node, depot)
    except ValueError:
        return False
    projected_distance = route.distance_km + out_d + back_d
    projected_duration = route.travel_minutes + out_t + back_t + route.service_minutes + stop.service_minutes
    return ((vehicle.max_distance_km is None or projected_distance <= vehicle.max_distance_km + 1e-9) and
            (vehicle.max_duration_minutes is None or projected_duration <= vehicle.max_duration_minutes + 1e-9))


def _solve_heuristic(graph: nx.Graph, depot: str, stops: list[Stop], vehicles: list[Vehicle]) -> RoutingResult:
    """Fallback used only when OR-Tools cannot be imported."""
    started = time.perf_counter()
    if depot not in graph: raise ValueError("depot is not on graph")
    if len({s.bin_id for s in stops}) != len(stops): raise ValueError("duplicate bin IDs")
    routes = [Route(vehicle_id=v.vehicle_id, road_path=[v.current_node or depot], starting_load_kg=v.current_load_kg) for v in vehicles if v.available]
    active = [v for v in vehicles if v.available]
    remaining = sorted(stops, key=lambda s: ({"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[s.urgency], s.bin_id))
    unserved: list[dict] = []
    while remaining:
        stop = remaining.pop(0)
        candidates = []
        for route, vehicle in zip(routes, active):
            if _can_append(route, stop, vehicle, graph, depot):
                here = route.road_path[-1]
                try: candidates.append((shortest_leg(graph, here, stop.node)[2], route, vehicle))
                except ValueError: pass
        if not candidates:
            reason = "unreachable_or_constraint_infeasible"
            if not active: reason = "no_available_vehicle"
            elif all(stop.demand_kg > v.capacity_kg - v.current_load_kg for v in active): reason = "demand_exceeds_any_vehicle_capacity"
            unserved.append({"bin_id": stop.bin_id, "urgency": stop.urgency, "reason": reason})
            continue
        _, route, vehicle = min(candidates, key=lambda item: (item[0], item[1].vehicle_id))
        _append_leg(route, graph, route.road_path[-1], stop.node)
        route.stops.append(stop); route.service_minutes += stop.service_minutes
    for route in routes:
        if route.stops: _append_leg(route, graph, route.road_path[-1], depot)
        else: route.road_path = []
    validation = validate_routes(graph, depot, routes, stops, vehicles)
    status = "FEASIBLE" if validation["valid"] else "INFEASIBLE"
    # Expose no false global-optimality claim: this is a deterministic heuristic.
    return RoutingResult(status=status, solver="deterministic_nearest_feasible_heuristic", routes=routes,
                         unserved=unserved, validation=validation, runtime_seconds=time.perf_counter() - started,
                         objective_distance_km=sum(r.distance_km for r in routes))


def _expanded_route(graph: nx.Graph, depot: str, vehicle_id: str, stops: list[Stop], start_node: str | None = None) -> Route:
    route = Route(vehicle_id=vehicle_id, road_path=[start_node or depot])
    for stop in stops:
        _append_leg(route, graph, route.road_path[-1], stop.node)
        route.stops.append(stop)
        route.service_minutes += stop.service_minutes
    if stops:
        _append_leg(route, graph, route.road_path[-1], depot)
    else:
        route.road_path = []
    return route


def _precheck(graph: nx.Graph, depot: str, stops: list[Stop], vehicles: list[Vehicle]) -> tuple[list[Stop], list[dict]]:
    active = [vehicle for vehicle in vehicles if vehicle.available]
    for vehicle in active:
        if vehicle.current_node is not None and vehicle.current_node not in graph:
            raise ValueError(f"unknown current node for vehicle {vehicle.vehicle_id}: {vehicle.current_node}")
    eligible: list[Stop] = []
    unserved: list[dict] = []
    for stop in stops:
        reason = None
        if not active:
            reason = "no_available_vehicle"
        elif all(stop.demand_kg > vehicle.capacity_kg - vehicle.current_load_kg for vehicle in active):
            reason = "demand_exceeds_any_vehicle_capacity"
        else:
            reachable = False
            for vehicle in active:
                if stop.demand_kg > vehicle.capacity_kg - vehicle.current_load_kg:
                    continue
                try:
                    shortest_leg(graph, vehicle.current_node or depot, stop.node)
                    shortest_leg(graph, stop.node, depot)
                    reachable = True
                    break
                except ValueError:
                    continue
            if not reachable:
                reason = "unreachable_from_or_to_depot"
        if reason:
            unserved.append({"bin_id": stop.bin_id, "urgency": stop.urgency, "reason": reason})
        else:
            eligible.append(stop)
    return eligible, unserved


def _solve_ortools(graph: nx.Graph, depot: str, stops: list[Stop], vehicles: list[Vehicle], time_limit_seconds: int) -> RoutingResult:
    """Solve a CVRP with OR-Tools Guided Local Search and independently reconstruct every route."""
    started = time.perf_counter()
    active = [vehicle for vehicle in vehicles if vehicle.available]
    eligible, precheck_unserved = _precheck(graph, depot, stops, vehicles)
    if not active:
        routes: list[Route] = []
        validation = validate_routes(graph, depot, routes, stops, vehicles)
        return RoutingResult("INFEASIBLE", "ortools_guided_local_search", routes, precheck_unserved, validation,
                             time.perf_counter() - started, 0.0)
    if not eligible:
        routes = [Route(vehicle_id=vehicle.vehicle_id, starting_load_kg=vehicle.current_load_kg) for vehicle in active]
        validation = validate_routes(graph, depot, routes, stops, vehicles)
        status = "FEASIBLE" if validation["valid"] else "INFEASIBLE"
        return RoutingResult(status, "ortools_guided_local_search", routes, precheck_unserved, validation,
                             time.perf_counter() - started, 0.0)

    locations = [depot] + [stop.node for stop in eligible]
    start_indices = []
    for vehicle in active:
        start_indices.append(len(locations))
        locations.append(vehicle.current_node or depot)
    count = len(locations)
    distance = [[0] * count for _ in range(count)]
    duration = [[0] * count for _ in range(count)]
    for i, source in enumerate(locations):
        for j, target in enumerate(locations):
            if i == j:
                continue
            _, km, minutes = shortest_leg(graph, source, target)
            distance[i][j] = round(km * 1000)  # metres (integer constraint units)
            duration[i][j] = round(minutes * 1000)  # milliminutes preserve edge precision

    manager = pywrapcp.RoutingIndexManager(count, len(active), start_indices, [0] * len(active))
    routing = pywrapcp.RoutingModel(manager)
    distance_callback = routing.RegisterTransitCallback(lambda a, b: distance[manager.IndexToNode(a)][manager.IndexToNode(b)])
    routing.AddDimensionWithVehicleCapacity(distance_callback, 0,
                                            [round((v.max_distance_km if v.max_distance_km is not None else 1_000_000) * 1000) for v in active],
                                            True, "Distance")

    demands = [0] + [round(stop.demand_kg * 1000) for stop in eligible] + [0] * len(active)
    demand_callback = routing.RegisterUnaryTransitCallback(lambda index: demands[manager.IndexToNode(index)])
    routing.AddDimensionWithVehicleCapacity(demand_callback, 0,
                                            [round((v.capacity_kg - v.current_load_kg) * 1000) for v in active], True, "Capacity")

    # Each arc accrues driving time plus service time at the node being left.
    service = [0] + [round(stop.service_minutes * 1000) for stop in eligible] + [0] * len(active)
    time_callback = routing.RegisterTransitCallback(lambda a, b: duration[manager.IndexToNode(a)][manager.IndexToNode(b)] + service[manager.IndexToNode(a)])
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback)
    routing.AddDimensionWithVehicleCapacity(time_callback, 0,
                                            [round((v.max_duration_minutes if v.max_duration_minutes is not None else 1_000_000) * 1000) for v in active],
                                            True, "Time")

    # Critical stops are hard mandatory. Others carry a prohibitive penalty, so any dropped stop is explicit.
    for index, stop in enumerate(eligible, start=1):
        routing_index = manager.NodeToIndex(index)
        if stop.urgency != "CRITICAL":
            routing.AddDisjunction([routing_index], 10**12)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromSeconds(max(1, int(time_limit_seconds)))
    parameters.log_search = False
    assignment = routing.SolveWithParameters(parameters)

    routes: list[Route] = []
    unserved = list(precheck_unserved)
    if assignment:
        served_indices: set[int] = set()
        for vehicle_index, vehicle in enumerate(active):
            ordered: list[Stop] = []
            index = routing.Start(vehicle_index)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if 1 <= node <= len(eligible):
                    ordered.append(eligible[node - 1]); served_indices.add(node)
                index = assignment.Value(routing.NextVar(index))
            route = _expanded_route(graph, depot, vehicle.vehicle_id, ordered, vehicle.current_node)
            route.starting_load_kg = vehicle.current_load_kg
            routes.append(route)
        for index, stop in enumerate(eligible, start=1):
            if index not in served_indices:
                unserved.append({"bin_id": stop.bin_id, "urgency": stop.urgency, "reason": "solver_dropped_or_constraint_infeasible"})
    else:
        routes = [Route(vehicle_id=vehicle.vehicle_id, starting_load_kg=vehicle.current_load_kg) for vehicle in active]
        unserved.extend({"bin_id": stop.bin_id, "urgency": stop.urgency, "reason": "no_feasible_solution"} for stop in eligible)

    validation = validate_routes(graph, depot, routes, stops, vehicles)
    status = "FEASIBLE" if validation["valid"] else "INFEASIBLE"
    return RoutingResult(status, "ortools_cvrp_guided_local_search", routes, unserved, validation,
                         time.perf_counter() - started, sum(route.distance_km for route in routes))


def solve(graph: nx.Graph, depot: str, stops: list[Stop], vehicles: list[Vehicle], time_limit_seconds: int = 5) -> RoutingResult:
    """Solve the active rolling-horizon planning snapshot with OR-Tools, or a named fallback."""
    if depot not in graph:
        raise ValueError("depot is not on graph")
    if len({stop.bin_id for stop in stops}) != len(stops):
        raise ValueError("duplicate bin IDs")
    for vehicle in vehicles:
        if vehicle.current_node is not None and vehicle.current_node not in graph:
            raise ValueError(f"unknown current node for vehicle {vehicle.vehicle_id}: {vehicle.current_node}")
    if ORTOOLS_AVAILABLE:
        return _solve_ortools(graph, depot, stops, vehicles, time_limit_seconds)
    return _solve_heuristic(graph, depot, stops, vehicles)
