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
3. Assigns pickups by urgency and demand order. For each pickup, it compares the traffic-aware truck-to-bin approach time for every available truck that has enough remaining payload and a feasible route insertion. It reserves that truck's capacity before assigning the next pickup. If a complete live route matrix is unavailable, it uses simulated road travel times for this comparison.
4. Uses OR-Tools CVRP with Guided Local Search to optimize the pickup order within those truck assignments, subject to reachability, return-to-depot, and configured distance and duration limits.
5. Treats critical pickups as mandatory. If a required pickup cannot be served, it is reported as unserved and the plan is marked infeasible. Other unserved pickups are also reported with a reason.
6. Reconstructs route distance and time from graph edges and independently validates the result.

For each bin, the planner selects the truck with the shortest traffic-aware approach time from its mock GPS location among trucks with enough remaining payload and a feasible route insertion. It reserves capacity in urgency and demand order, then uses OR-Tools to order each truck's assigned pickups. Route distance, duration, and stop order are still based on the local simulated graph; Google traffic-aware ETAs guide the truck-to-bin assignment when available. The solver returns a feasible plan within its time limit; it does not claim a globally optimal plan. If OR-Tools is unavailable, a deterministic nearest-feasible heuristic is used instead.

### Explain it simply

EcoRoute plans pickups in two stages:

1. **Assign bins to trucks.** It considers urgent and heavier pickups first. For each bin, it checks trucks that have enough remaining payload and can fit the pickup into a route that respects configured distance and time limits. It chooses the eligible truck with the shortest approach time from its mock GPS location, then reserves that truck's capacity before assigning the next bin. Google Routes traffic-aware ETAs are used for the comparison when available; otherwise the project uses its simulated road times.
2. **Order each truck's pickups.** After bins have been assigned, OR-Tools arranges each truck's stops into a good route order while respecting its capacity and route limits. The route order and those limits use the project's simulated road network.

A short description for a presentation is:

> “We assign each bin to the closest truck that has enough capacity and can complete a valid route. Then an optimizer arranges that truck's pickups into an efficient order.”

### What OR-Tools means here

**OR-Tools** is Google's open-source toolkit for solving optimization problems. This project uses its vehicle-routing solver to order pickups for the truck assignments that were already selected. The routing problem is a **capacitated vehicle routing problem (CVRP)**: multiple vehicles must visit pickup locations while observing each vehicle's carrying capacity and route limits.

The solver uses **Guided Local Search**. In practical terms, it tries different stop orders and steers away from route patterns that have been difficult to improve, aiming to find a better plan before its time limit. It is a search method, so the result is a good feasible plan when one is found; it is not a proof that the entire fleet has the globally best assignment and route.

OR-Tools handles optimization in this project. It does not provide the map, GPS positions, or live traffic. Google Routes can provide traffic-aware travel times for comparing trucks and road geometry and leg metrics for the map. The project still calculates pickup order and route constraints with its simulated road network. If OR-Tools is unavailable, the project falls back to a deterministic nearest-feasible heuristic.

## What the dashboard shows

Open **Route simulation**, set each truck's mock GPS node and current load in the sidebar, choose the urgency threshold, and select **Optimize current planning cycle**.

The result includes:

- A fleet snapshot showing the mock GPS node, coordinates, current load, and free capacity used for the plan.
- A dispatch overview showing each truck's next pickup, estimated arrival, number of assigned stops, and capacity left after the planned pickups.
- An expandable truck detail with its ordered stop list, each stop's ETA from route start, pickup demand, and remaining capacity after each pickup.
- A map with each bin as a **B** marker. Its color shows fill level: green (0–39%), yellow (40–69%), orange (70–89%), and red (90–100%). Clicking it shows fill, zone, urgency, collection prediction, and coordinates.
- Each truck appears as a **T** marker at its mock GPS position. Gray means idle; blue means it has pickups assigned. Clicking a truck shows its status, current load, remaining capacity, next pickup, and planned pickup sequence.
- Active truck popups compare the next pickup's approach ETA from every available truck's mock GPS location. The comparison labels the assigned truck, identifies other feasible trucks, and gives the capacity or route-limit reason when a truck is ineligible. Times use Google Routes traffic-aware ETAs when the full live matrix is available, otherwise simulated road-time estimates.
- Active truck popups include **Check optimized route**. It draws and zooms to that truck's road route, then shows the optimizer's simulated route estimate plus Google road distance and traffic-aware ETA by pickup leg when Routes API is available. Only the currently selected truck's route is visible; selecting another truck hides the previous route. Idle trucks stay as GPS markers without a route line. If Routes API fails, the popup falls back to simulated leg distance and ETA.
- A **Show truck route on map** action in each truck card focuses the map on that truck. Route lines stay hidden until an operator checks an active truck's route.
- Per-bin assignment explanations show approach time and remaining capacity for each truck, making the nearest capable assignment visible in the dashboard table as well as the truck popup.

Truck assignment uses the Google Routes traffic-aware matrix when available, and the popup's comparison lists each truck's approach ETA for the next pickup. The optimizer's route order, route limits, distance, duration, and simulated ETA remain based on the local graph. Google route distance and traffic-aware time are also provided for explanation in the route popup; arrival estimates add the prototype's service time at prior pickups. Re-run the optimizer after changing truck locations, loads, traffic settings, or the pickup manifest to create an updated plan.

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

Enable the Maps JavaScript API and Routes API in a billing-enabled Google Cloud project. Configure `GOOGLE_MAPS_API_KEY` as a browser key with an HTTP-referrer restriction for the dashboard. For live assignment ETAs, configure a separate `GOOGLE_ROUTES_API_KEY` as a server key restricted to the server IP and Routes API. The server key must remain private and out of version control. The route matrix uses traffic-aware drive durations for truck-to-bin comparisons; if that request fails or the server key is missing, the dashboard explicitly reports that assignment used simulated travel times.

Google Routes supplies road geometry and traffic-aware distance/time for the route popup. The optimizer's route assignment, traffic adjustment, route distance, route duration, and stop ETAs come from the local simulated graph; they do not use Google's live traffic or travel-time result. If a Routes API request fails, the popup labels and uses simulated leg distance and ETA.

## Prototype boundaries

- GPS positions, coordinates, road links, and bin demand are simulated. Assignment uses Google traffic-aware approach ETAs when the server Routes key is available and the matrix request succeeds; local graph ETAs are the fallback. Route constraints and route-start ETAs remain simulated graph estimates.
- There is no live GPS ingestion, live traffic feed, automatic background re-optimization scheduler, or real driver push notification. The dashboard's “Dispatch notice” is an on-screen explanation of the first planned pickup.
- The rolling-horizon interval is a planning setting/response description; the operator must trigger a new optimization cycle in this prototype.
- Google supplies traffic-aware truck-to-bin comparison ETAs for assignment and road geometry plus traffic-aware leg metrics for the selected popup when Routes API is available. The application uses those ETAs to select a feasible truck; Google does not make the assignment decision. Pickup order and route constraints still use the simulated graph optimizer.
- The optimizer does not split a bin's demand across trucks. Time windows are not enforced.

These boundaries make the demo explainable while keeping the fleet assignment and capacity decisions visible in the code and dashboard.
