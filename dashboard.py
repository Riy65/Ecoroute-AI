from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from optimization.graph import build_grid_graph
from optimization.metrics import totals
from optimization.problem import Stop, Vehicle
from optimization.solver import solve
from prediction.pipeline import generate_data, predict_latest, train

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
EXP = ROOT / "experiments"

st.set_page_config(page_title="EcoRoute AI", page_icon="♻️", layout="wide")
st.title("EcoRoute AI")
st.caption("Simulated waste collection prediction and route optimization")


def have_data() -> bool:
    return (RAW / "bins.csv").exists() and (RAW / "bin_history.csv").exists()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(RAW / "bins.csv"), pd.read_csv(RAW / "bin_history.csv", parse_dates=["timestamp"])


def route_map(points: pd.DataFrame, route_result: dict | None, capacity: float) -> pdk.Deck:
    """Map simulated bin state and planned truck paths; all paths use the visible bin positions."""
    urgency_color = {"LOW": [76, 175, 80], "MEDIUM": [255, 193, 7], "HIGH": [255, 112, 67], "CRITICAL": [220, 53, 69]}
    bins_layer = points.assign(
        color=points["urgency_category"].map(urgency_color),
        label=points.apply(lambda row: f"{row.bin_id}  {row.urgency_category}", axis=1),
    )
    center = {"lat": float(points.latitude.mean()), "lon": float(points.longitude.mean())}
    layers: list = [
        pdk.Layer("ScatterplotLayer", bins_layer, get_position="[longitude, latitude]", get_fill_color="color", get_radius=105, pickable=True),
        pdk.Layer("TextLayer", bins_layer, get_position="[longitude, latitude]", get_text="label", get_color=[20, 20, 20], get_size=11, get_alignment_baseline="bottom", get_pixel_offset=[0, -8]),
    ]
    depot = [center["lon"], center["lat"]]
    if route_result:
        by_id = points.set_index("bin_id")
        paths, trucks = [], []
        colors = [[30, 136, 229], [123, 31, 162], [0, 121, 107], [251, 140, 0]]
        for index, route in enumerate(route_result["routes"]):
            stop_ids = [stop["bin_id"] for stop in route["stops"]]
            if not stop_ids:
                continue
            positions = [depot] + [[float(by_id.loc[s].longitude), float(by_id.loc[s].latitude)] for s in stop_ids] + [depot]
            remaining = max(0.0, capacity - sum(stop["demand_kg"] for stop in route["stops"]))
            color = colors[index % len(colors)]
            paths.append({"path": positions, "color": color, "vehicle": route["vehicle_id"]})
            last = positions[-2]
            trucks.append({"longitude": last[0], "latitude": last[1], "label": f"🚚 {route['vehicle_id']}\n{remaining:.0f} kg free", "color": color})
        if paths:
            layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_color="color", get_width=5, width_min_pixels=3, pickable=True))
            layers.append(pdk.Layer("ScatterplotLayer", trucks, get_position="[longitude, latitude]", get_fill_color="color", get_radius=180, pickable=True))
            layers.append(pdk.Layer("TextLayer", trucks, get_position="[longitude, latitude]", get_text="label", get_color=[0, 0, 0], get_size=14, get_alignment_baseline="top", get_pixel_offset=[0, 12]))
    depot_layer = pd.DataFrame([{"longitude": depot[0], "latitude": depot[1], "label": "DEPOT"}])
    layers.append(pdk.Layer("ScatterplotLayer", depot_layer, get_position="[longitude, latitude]", get_fill_color=[33, 33, 33], get_radius=190))
    layers.append(pdk.Layer("TextLayer", depot_layer, get_position="[longitude, latitude]", get_text="label", get_color=[0, 0, 0], get_size=14, get_alignment_baseline="top"))
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=13, pitch=0), tooltip={"text": "{bin_id}\nUrgency: {urgency_category}\nFill: {current_fill_level_pct}%"}, map_style="light")


def google_maps_key() -> str | None:
    """Read a browser-restricted Maps key without hard-coding it into source control."""
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY")
        if key:
            return str(key)
    except FileNotFoundError:
        pass
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_MAPS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"') or None
    return None


def render_google_map(points: pd.DataFrame, route_result: dict | None, capacity: float) -> None:
    key = google_maps_key()
    if not key:
        st.warning("Add `GOOGLE_MAPS_API_KEY` to `.env` to use Google Maps. Showing the local fallback map until a key is configured.")
        st.pydeck_chart(route_map(points, route_result, capacity), use_container_width=True)
        return
    urgency_color = {"LOW": "#4caf50", "MEDIUM": "#ffc107", "HIGH": "#ff7043", "CRITICAL": "#dc3545"}
    bins = [{"id": row.bin_id, "lat": float(row.latitude), "lng": float(row.longitude), "urgency": row.urgency_category, "fill": round(float(row.current_fill_level_pct), 1), "color": urgency_color[row.urgency_category]} for _, row in points.iterrows()]
    center = {"lat": float(points.latitude.mean()), "lng": float(points.longitude.mean())}
    depot = center.copy()
    routes, trucks = [], []
    if route_result:
        by_id = points.set_index("bin_id")
        colors = ["#1e88e5", "#7b1fa2", "#00796b", "#fb8c00"]
        for index, route in enumerate(route_result["routes"]):
            stops = route["stops"]
            path = [depot] + [{"lat": float(by_id.loc[s["bin_id"]].latitude), "lng": float(by_id.loc[s["bin_id"]].longitude)} for s in stops] + [depot]
            remaining = max(0.0, capacity - sum(float(stop["demand_kg"]) for stop in stops))
            color = colors[index % len(colors)]
            if stops:
                routes.append({"path": path, "color": color, "vehicle": route["vehicle_id"]})
            # Every active truck remains visible, including unused trucks parked at the depot.
            trucks.append({"lat": depot["lat"], "lng": depot["lng"], "vehicle": route["vehicle_id"], "remaining": round(remaining, 1), "color": color, "has_route": bool(stops), "number": index + 1})
    payload = json.dumps({"bins": bins, "center": center, "depot": depot, "routes": routes, "trucks": trucks})
    map_html = f"""
    <div id="map-status" style="font:13px sans-serif;margin:0 0 8px;color:#555">Loading Google road routes...</div>
    <div id="map" style="height:560px;width:100%;border-radius:10px"></div>
    <script>window.ecorouteData = {payload};</script>
    <script>
      async function initEcoRouteMap() {{
        const d = window.ecorouteData;
        const status = document.getElementById('map-status');
        const map = new google.maps.Map(document.getElementById('map'), {{center:d.center, zoom:13, mapTypeControl:false, streetViewControl:false, mapId:'DEMO_MAP_ID'}});
        const info = new google.maps.InfoWindow();
        new google.maps.Marker({{position:d.depot, map, title:'Depot', label:{{text:'D', color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.CIRCLE, fillColor:'#212121', fillOpacity:1, strokeColor:'#fff', strokeWeight:2, scale:12}}}});
        d.bins.forEach(bin => {{
          const marker = new google.maps.Marker({{position:bin, map, title:`${{bin.id}} — ${{bin.urgency}}`, label:{{text:bin.urgency[0], color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.CIRCLE, fillColor:bin.color, fillOpacity:1, strokeColor:'#fff', strokeWeight:2, scale:10}}}});
          marker.addListener('click', () => info.open({{map, anchor:marker, content:`<b>${{bin.id}}</b><br>Urgency: ${{bin.urgency}}<br>Fill level: ${{bin.fill}}%`}}));
        }});
        const {{Route}} = await google.maps.importLibrary('routes');
        let roadRouteCount = 0;
        let roadRouteFailures = 0;
        const roadTracks = {{}};
        for (const route of d.routes) {{
          try {{
            let completeTrack = [];
            // Routes API accepts a limited number of intermediate waypoints. Split long truck runs
            // into contiguous chunks, preserving every planned stop and road-following geometry.
            for (let start = 0; start < route.path.length - 1;) {{
              const end = Math.min(start + 24, route.path.length - 1);
              const request = {{origin:route.path[start], destination:route.path[end], intermediates:route.path.slice(start + 1, end), travelMode:'DRIVING', fields:['path']}};
              const {{routes}} = await Route.computeRoutes(request);
              if (!routes || !routes.length) throw new Error('No driving route returned');
              routes[0].createPolylines().forEach(polyline => {{ polyline.setOptions({{strokeColor:route.color, strokeOpacity:.9, strokeWeight:5}}); polyline.setMap(map); }});
              const legTrack = (routes[0].path || []).map(point => ({{lat:point.lat(), lng:point.lng()}}));
              completeTrack = completeTrack.concat(completeTrack.length ? legTrack.slice(1) : legTrack);
              start = end;
            }}
            // Downsample to a smooth, bounded animation loop.
            const stride = Math.max(1, Math.ceil(completeTrack.length / 180));
            roadTracks[route.vehicle] = completeTrack.filter((_, index) => index % stride === 0);
            roadRouteCount += 1;
          }} catch (error) {{
            console.error('Google road route unavailable for ' + route.vehicle, error);
            new google.maps.Polyline({{path:route.path, geodesic:true, strokeColor:route.color, strokeOpacity:.45, strokeWeight:3, map}});
            roadTracks[route.vehicle] = route.path;
            roadRouteFailures += 1;
          }}
        }}
        const animatedTrucks = [];
        d.trucks.forEach((truck, index) => {{
          // A small deterministic offset prevents unused trucks from overlapping at the depot.
          const parked = {{lat:truck.lat + (index - (d.trucks.length - 1) / 2) * .00018, lng:truck.lng + (index % 2 ? .00014 : -.00014)}};
          const marker = new google.maps.Marker({{position:truck.has_route ? truck : parked, map, title:`${{truck.vehicle}} — ${{truck.remaining}} kg free`, label:{{text:`T${{truck.number}}`, color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.FORWARD_CLOSED_ARROW, fillColor:truck.color, fillOpacity:1, strokeColor:'#222', strokeWeight:1, scale:6}}}});
          const capacityLabel = new google.maps.InfoWindow({{content:`<b>Truck ${{truck.number}}</b><br>${{truck.remaining}} kg free<br>${{truck.has_route ? 'Moving on planned route' : 'Parked at depot'}}`}});
          capacityLabel.open({{map, anchor:marker}});
          marker.addListener('click', () => info.open({{map, anchor:marker, content:`<b>🚚 ${{truck.vehicle}}</b><br>Remaining capacity: ${{truck.remaining}} kg`}}));
          animatedTrucks.push({{truck, marker, capacityLabel, step:0}});
        }});
        // Animation follows the road geometry returned by Google Routes, then loops for repeated simulation.
        setInterval(() => animatedTrucks.forEach(item => {{
          const track = roadTracks[item.truck.vehicle];
          if (item.truck.has_route && track && track.length > 1) {{
            item.step = (item.step + 1) % track.length;
            item.marker.setPosition(track[item.step]);
            item.capacityLabel.setPosition(track[item.step]);
          }}
        }}), 160);
        status.textContent = roadRouteFailures === 0 ? `Google road routes shown for ${{roadRouteCount}} truck(s). All ${{d.trucks.length}} truck markers are visible and animated.` : `Google Routes API failed for ${{roadRouteFailures}} truck(s); those routes use faint straight fallback lines. Enable Routes API and check key restrictions.`;
      }}
    </script>
    <script async src="https://maps.googleapis.com/maps/api/js?key={key}&libraries=routes&callback=initEcoRouteMap"></script>
    """
    components.html(map_html, height=580, scrolling=False)


with st.sidebar:
    st.header("Simulation settings")
    bin_count = st.slider("Simulated bins", 20, 100, 64, 4)
    days = st.slider("History days", 5, 60, 30, 5)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    truck_count = st.slider("Available trucks", 1, 4, 2)
    truck_capacity = st.number_input("Truck capacity (kg)", min_value=100.0, value=1600.0, step=100.0)
    rolling_interval = st.select_slider("Rolling re-optimization interval (minutes)", options=[5, 15, 30, 60], value=15)
    solver_seconds = st.slider("OR-Tools solver budget (seconds)", 1, 15, 5)

    if st.button("Generate scattered bins and train model", type="primary", use_container_width=True):
        with st.spinner("Generating simulated history and training the urgency model..."):
            bins, history = generate_data(str(RAW), bins=bin_count, days=days, seed=int(seed))
            metrics = train(history, str(MODELS / "urgency.joblib"))
            EXP.mkdir(exist_ok=True)
            (EXP / "model_evaluation.json").write_text(json.dumps(metrics, indent=2))
        st.success(f"Generated {len(bins):,} bins and trained the model.")

if not have_data() or not (MODELS / "urgency.joblib").exists():
    st.info("Choose the simulation settings and select **Generate data and train model** to begin.")
    st.stop()

bins, history = load_data()
predictions = pd.DataFrame(predict_latest(history, bins, str(MODELS / "urgency.joblib")))
latest = predictions.merge(bins, on="bin_id", how="left")

tabs = st.tabs(["Overview", "Predictions", "Route simulation", "About results"])

with tabs[0]:
    st.subheader("Current simulated system")
    a, b, c, d = st.columns(4)
    a.metric("Bins", len(bins))
    b.metric("Observations", f"{len(history):,}")
    c.metric("Urgent bins", int(latest.urgency_category.isin(["HIGH", "CRITICAL"]).sum()))
    d.metric("Critical bins", int((latest.urgency_category == "CRITICAL").sum()))
    render_google_map(latest, None, float(truck_capacity))
    st.caption("Click a colored bin to see its urgency and fill level. Positions and the depot are simulated.")

with tabs[1]:
    st.subheader("Urgency predictions")
    show = latest[["bin_id", "zone_id", "current_fill_level_pct", "urgency_probability", "predicted_collection_required", "urgency_category"]].sort_values(["urgency_category", "urgency_probability"], ascending=[True, False])
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.bar_chart(latest.groupby("urgency_category").size().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0))

with tabs[2]:
    st.subheader("Run collection-route simulation")
    policy = st.selectbox("Select bins at or above urgency", ["HIGH", "MEDIUM", "LOW"], index=0)
    levels = {"HIGH": {"HIGH", "CRITICAL"}, "MEDIUM": {"MEDIUM", "HIGH", "CRITICAL"}, "LOW": {"LOW", "MEDIUM", "HIGH", "CRITICAL"}}
    selected = latest[latest.urgency_category.isin(levels[policy])].copy().reset_index(drop=True)
    selected = selected.iloc[:64]
    node_for = {bin_id: f"n{i // 8}_{i % 8}" for i, bin_id in enumerate(bins.bin_id.iloc[:64])}
    selected["node"] = selected.bin_id.map(node_for)
    selected["demand_kg"] = selected.capacity_kg * selected.current_fill_level_pct / 100
    st.write(f"**{len(selected)}** selected bins will be considered. Critical bins remain mandatory; infeasible bins are reported, never silently removed.")
    if st.button("Optimize current planning cycle", type="primary"):
        graph = build_grid_graph(seed=int(seed))
        stops = [Stop(bin_id=row.bin_id, node=row.node, demand_kg=float(row.demand_kg), urgency=row.urgency_category) for _, row in selected.iterrows()]
        vehicles = [Vehicle(f"truck-{i + 1}", float(truck_capacity)) for i in range(truck_count)]
        result = solve(graph, "n0_0", stops, vehicles, time_limit_seconds=int(solver_seconds))
        metrics = totals(result)
        st.session_state["route_result"] = result.to_dict() | {"metrics": metrics, "rolling_interval": rolling_interval, "solver_budget": solver_seconds}
    result = st.session_state.get("route_result")
    if result:
        status = result["status"]
        (st.success if status == "FEASIBLE" else st.error)(f"Solver status: {status} ({result['solver']})")
        st.caption(f"Rolling-horizon cycle: refresh this plan every {result['rolling_interval']} minutes or immediately when a bin becomes critical. Solver budget: {result['solver_budget']} seconds.")
        values = result["metrics"]
        e, f, g, h = st.columns(4)
        e.metric("Distance", f"{values['distance_km']:.2f} km")
        f.metric("Travel time", f"{values['travel_minutes']:.0f} min")
        g.metric("Fuel estimate", f"{values['fuel_liters_estimate']:.2f} L")
        h.metric("CO2 estimate", f"{values['co2_kg_estimate']:.2f} kg")
        st.subheader("Vehicle routes")
        for route in result["routes"]:
            ids = [stop["bin_id"] for stop in route["stops"]]
            st.write(f"**{route['vehicle_id']}** — {len(ids)} stops, {route['distance_km']:.2f} km")
            st.code("Depot → " + " → ".join(ids) + " → Depot" if ids else "Unused vehicle")
        if result["unserved"]:
            st.warning("Unserved bins")
            st.dataframe(pd.DataFrame(result["unserved"]), use_container_width=True, hide_index=True)
        with st.expander("Independent validation report"):
            st.json(result["validation"])
        st.subheader("Bins and planned truck routes")
        render_google_map(latest, result, float(truck_capacity))
        st.caption("Solid colored lines follow Google road directions. Each truck's remaining capacity stays visible next to its marker.")

with tabs[3]:
    st.subheader("How to interpret this prototype")
    st.write("The dashboard uses synthetic records and simulated roads. Fuel and CO2 are estimates. The solver is a deterministic feasible-route heuristic, not a claim of globally optimal routing.")
    st.write("If a route cannot satisfy capacity or reachability constraints, the affected bin is shown as unserved along with the reason.")
