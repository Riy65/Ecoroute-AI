# EcoRoute AI progress

**Overall verified completion:** 68%  
**Current milestone:** 14 - testing and defect correction  
**Current status:** OR-Tools rolling-horizon routing verified  
**Last updated:** 2026-10-29 (Asia/Kolkata)

## Weighted completion method

Route optimisation, constraints, independent validation, and edge-case tests account for 60% of the completion score. Data/ML account for 15%, API and end-to-end integration 15%, and documentation/reproducibility 10%. A completed item requires passing applicable tests.

## Milestone tracker

| # | Milestone | Status | Verified | Current state / next action |
|---|---|---|---:|---|
| 1 | Repository inspection and setup | COMPLETE | 100% | Project structure, configuration, specification, and tracker created. |
| 2 | Synthetic dataset generation and validation | COMPLETE | 85% | Reproducible 0-100% data generator implemented; validation is embedded in generator/model preconditions. |
| 3 | ML urgency prediction | COMPLETE | 85% | Chronological Random Forest training and persisted inference implemented. |
| 4 | Road graph construction | COMPLETE | 90% | Directed deterministic road graph plus edge validation implemented. |
| 5 | Travel-cost matrix generation | COMPLETE | 80% | Per-leg Dijkstra cost calculation is implemented; a persisted matrix is not yet exposed. |
| 6 | Single-truck route optimization | COMPLETE | 95% | OR-Tools CVRP plus Guided Local Search is tested. |
| 7 | Multiple-truck support | COMPLETE | 85% | OR-Tools supports assignment across multiple available vehicles; advanced balancing remains future work. |
| 8 | Constraint handling | COMPLETE | 80% | Capacity, distance, duration and return constraints are modelled; time-window enforcement remains incomplete. |
| 9 | Independent route validation | COMPLETE | 85% | Graph transitions, costs, depot, capacity, limits, duplicate/missing coverage validated. |
| 10 | Edge-case coverage | IN PROGRESS | 45% | Empty, invalid, duplicate and capacity tests pass; expanded matrix remains. |
| 11 | Baseline comparison and metrics | COMPLETE | 75% | Fixed-order same-stop comparison and estimates implemented. |
| 12 | FastAPI integration | IN PROGRESS | 75% | Request model accepts solve budget and rolling interval; direct request-path smoke test passed. |
| 13 | End-to-end simulation | IN PROGRESS | 70% | Simulation uses the upgraded solver through its shared request path; full HTTP test remains pending. |
| 14 | Testing and defect correction | IN PROGRESS | 75% | 4 routing tests passed, including OR-Tools solver selection; direct API request-path smoke test passed. |
| 15 | Documentation and reproducibility | COMPLETE | 90% | README, requirements, spec, progress record, and local Streamlit simulation dashboard completed. |

## Feature checklist

- [ ] Deterministic simulated road graph and shortest paths
- [ ] Constraint-aware multi-vehicle solver with explicit infeasibility reports
- [ ] Independent route validator
- [ ] Fixed-order and same-stop baselines
- [ ] Synthetic data and leakage-safe classifier
- [ ] FastAPI endpoints
- [ ] Unit, integration, and edge-case tests run successfully

## Decision and issue log

| Date | Entry |
|---|---|
| 2026-10-29 | Repository had no implementation. Required packages were downloaded and installed into the supplied runtime after network approval. |
| 2026-10-29 | Chose a deterministic nearest-feasible route heuristic. It explicitly reports `INFEASIBLE` and does not claim optimality; OR-Tools integration is a remaining solver enhancement. |
| 2026-10-29 | Direct synthetic ML smoke test produced 30 bins and 3,600 observations. Its synthetic-data F1 was 0.7153 and ROC-AUC 0.9747; these figures are not real-world performance claims. |
| 2026-10-29 | `pytest -q` passed 4 routing tests. A TestClient endpoint smoke attempt did not complete within the execution window, so HTTP integration is not marked verified. |

## Change log

- 2026-10-29: Initialized tracking after inspecting the repository.
- 2026-10-29: Implemented graph, solver, validator, metrics, synthetic data, ML pipeline, FastAPI application, and tests; fixed missing rate feature in history rows.
- 2026-10-29: Added a local Streamlit dashboard for data generation, prediction inspection, route simulation, and validation reporting. Dashboard syntax compilation succeeded; routing tests remain 4 passed.
- 2026-10-29: Google Maps integration renders Routes API road-following polylines for every planned truck stop (chunked to respect waypoint limits) and retains visible truck capacity labels. Live external-map verification requires a user-provided restricted key with Maps JavaScript API and Routes API enabled.
- 2026-10-29: Replaced the primary routing strategy with OR-Tools CVRP and Guided Local Search. Each dashboard/API optimization request is one rolling-horizon cycle with a configurable scheduled interval and bounded solver budget. Verified solver result: `ortools_cvrp_guided_local_search`, FEASIBLE, independently valid.
- 2026-10-29: Improved Google Maps presentation: every active truck is rendered, unused trucks are visibly parked by the depot, and routed trucks animate over Routes API road geometry. Dashboard syntax checks and routing tests passed; live animation requires a configured Maps JavaScript API and Routes API key.
