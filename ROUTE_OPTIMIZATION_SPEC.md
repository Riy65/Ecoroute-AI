# EcoRoute AI routing specification

## Problem definition

EcoRoute AI solves a capacitated vehicle-routing problem on a directed, weighted road graph. Given a depot, a collection manifest, and available vehicles, it finds routes that start and end at the depot and service required bins while enforcing configured hard constraints.

## Inputs and outputs

Each manifest item has a unique `bin_id`, graph `node`, non-negative demand in kilograms, service time in minutes, and urgency category. Vehicles have an identifier, non-negative capacity, availability, and optional per-route distance and duration limits. The graph edges carry non-negative distance (km) and travel time (minutes).

The solver returns per-vehicle ordered stops, expanded road-node paths, reconstructed distance, travel time, service time, total duration, metrics, solver identity/status, unserved bins and reasons, plus an independent validation report.

## Objective and constraints

The primary objective is minimum total shortest-path road distance. Travel time, fuel, and emissions are reported metrics rather than silently mixed objective terms. Hard constraints are vehicle capacity, graph reachability, return to depot, vehicle availability, and configured distance, duration, and time-window limits. Critical bins are mandatory; a critical bin that cannot be served is reported as unserved and makes the result infeasible.

## Algorithm

The graph service uses Dijkstra shortest paths between locations. The routing service uses Google OR-Tools CVRP with a PATH_CHEAPEST_ARC initial solution and Guided Local Search improvement under a caller-configured time limit. Each request is one rolling-horizon planning snapshot; it is re-run on a scheduled interval or urgency event. If OR-Tools cannot be imported, it uses a deterministic nearest-feasible heuristic. Neither strategy reports global optimality. Both paths are passed to a separate validator that reconstructs costs from graph edges.

## Urgency policy

Prediction is separate from routing. Categories are LOW, MEDIUM, HIGH, and CRITICAL. The default policy selects HIGH/CRITICAL predictions and marks CRITICAL bins mandatory. A caller may instead provide an explicit manifest for testing.

## Baselines and metrics

Operational comparison contrasts a fixed route visiting all planned bins with the selected-bin route and labels differing stop sets. Same-stop comparison keeps the exact manifest and vehicle assumptions, changing only stop order. Fuel is `distance_km * liters_per_100km / 100`; estimated emissions are fuel times `kg_co2_per_liter`.

## Validation and edge cases

The validator checks graph transitions, depot boundaries, stop coverage, duplicate stops, capacity, reconstructed costs, limits, time windows, and mandatory critical coverage. Invalid inputs raise validation errors; unreachable or constraint-infeasible requests return structured unserved reasons. No bin is silently discarded or split across vehicles.

## Acceptance criteria

A route is valid only when the independent validator reports no errors and every required bin is served. A result with unserved required bins is explicitly marked `INFEASIBLE`.
