# Fleet Routing and Dispatch Guide

This guide explains the fleet-routing and dispatch features added to EcoRoute AI, how the dashboard presents a plan, and what the prototype simulates.

## What the system does

For one planning cycle, EcoRoute AI takes a list of bins to collect and a snapshot of the available trucks. It assigns and orders pickups across the fleet, estimates routes and arrival times, and checks the result with an independent validator.

The optimizer considers each truck's current graph location and remaining payload capacity. It plans the fleet together rather than assigning every bin to whichever truck is geographically closest.

## How a truck is selected

Each truck has a capacity, current load, availability, current graph node, and optional maximum route distance and duration. The usable capacity is:

```text
remaining capacity = truck capacity - current load
```

Each pickup has a graph node, estimated waste demand, service time, and urgency. The planner:

1. Builds a directed, simulated road graph. Its road links have distance and travel-time weights. Segment congestion is deterministic for a given seed; a traffic multiplier can increase the simulated travel times.
2. Finds travel-time-weighted shortest paths between the depot, truck locations, and candidate pickups.
3. Uses OR-Tools CVRP with Guided Local Search to choose a joint set of truck routes and pickup order, subject to availability, capacity, reachability, return-to-depot, and configured distance and duration limits.
4. Treats critical pickups as mandatory. If a required pickup cannot be served, it is reported as unserved and the plan is marked infeasible. Other unserved pickups are also reported with a reason.
5. Reconstructs route distance and time from graph edges and independently validates the result.

The solver aims to minimize simulated driving and service time. It returns a good feasible plan within its time limit; it does not claim a globally optimal plan. If OR-Tools is unavailable, a deterministic nearest-feasible heuristic is used instead.

## What the dashboard shows

Open **Route simulation**, set each truck's mock GPS node and current load in the sidebar, choose the urgency threshold, and select **Optimize current planning cycle**.

The result includes:

- A fleet snapshot showing the mock GPS node, coordinates, current load, and free capacity used for the plan.
- A dispatch overview showing each truck's next pickup, estimated arrival, number of assigned stops, and capacity left after the planned pickups.
- An expandable truck detail with its ordered stop list, each stop's ETA from route start, pickup demand, and remaining capacity after each pickup.
- A map with all active trucks. An idle truck remains at its mock GPS location. A truck with pickups is animated along its planned route when route geometry is available.
- A clickable truck marker and clickable route line. Selecting either opens the truck's pickup order, per-stop ETAs, estimated route duration and distance, and capacity details. Selecting an idle truck shows its GPS node, coordinates, and idle status.
- Per-bin assignment explanations. These compare feasible insertions into each planned truck route to help explain the selected assignment. They are explanatory estimates; the final assignment comes from the joint fleet optimization.

Arrival times are estimated minutes from the current plan's start, not live ETAs. Re-run the optimizer after changing truck locations, loads, traffic settings, or the pickup manifest to create an updated plan.

## API endpoint

The FastAPI optimizer accepts an explicit manifest and fleet at `POST /api/routes/optimize`. For example:

```json
{
  "depot": "n0_0",
  "stops": [
    {"bin_id": "BIN-001", "node": "n1_1", "demand_kg": 40, "urgency": "CRITICAL"},
    {"bin_id": "BIN-002", "node": "n2_3", "demand_kg": 55, "urgency": "HIGH"}
  ],
  "vehicles": [
    {"vehicle_id": "truck-1", "capacity_kg": 600, "current_load_kg": 120, "current_node": "n2_2", "max_duration_minutes": 480},
    {"vehicle_id": "truck-2", "capacity_kg": 500, "current_load_kg": 0, "current_node": "n6_6", "max_duration_minutes": 480}
  ],
  "traffic_multiplier": 1.4,
  "solver_time_limit_seconds": 5
}
```

The response contains ordered routes, graph-node road paths, route metrics, fleet snapshots, unserved pickups and reasons, assignment explanations, and a validation report. `POST /api/routes/validate` independently validates a plan for the supplied request. See [README.md](README.md) for the complete API workflow and [ROUTE_OPTIMIZATION_SPEC.md](ROUTE_OPTIMIZATION_SPEC.md) for formal constraints.

Run the API and dashboard in separate terminals from the project root:

```powershell
python -m uvicorn app.main:app --reload
python -m streamlit run dashboard.py
```

The API documentation is at `http://127.0.0.1:8000/docs`; the dashboard is at `http://127.0.0.1:8501/`.

## Google Maps setup

The dashboard uses two Google Maps services when a key is configured:

- **Maps JavaScript API** renders the interactive base map and its markers.
- **Routes API** supplies road-following route geometry for planned truck routes.

Enable both APIs in the same billing-enabled Google Cloud project. The API key must allow both APIs and the dashboard's local website referrer. Configure the key as `GOOGLE_MAPS_API_KEY` in `.env` or Streamlit secrets. Keep the key private and out of version control.

Google Routes is used for map road geometry and truck animation. The optimizer's route assignment, traffic adjustment, route distance, route duration, and stop ETAs come from the local simulated graph; they do not use Google's live traffic or travel-time result. If the Routes API request fails, the map draws a faint straight-line fallback between planned locations. The dashboard displays a warning in that case.

## Prototype boundaries

- GPS positions, coordinates, road links, congestion, bin demand, and pickup ETAs are simulated.
- There is no live GPS ingestion, live traffic feed, automatic background re-optimization scheduler, or real driver push notification. The dashboard's “Dispatch notice” is an on-screen explanation of the first planned pickup.
- The rolling-horizon interval is a planning setting/response description; the operator must trigger a new optimization cycle in this prototype.
- The Google map displays road geometry when the Routes API is available, but Google does not choose which truck gets each bin.
- The optimizer does not split a bin's demand across trucks. Time windows are not enforced.

These boundaries make the demo explainable while keeping the fleet assignment and capacity decisions visible in the code and dashboard.
