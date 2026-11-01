from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from optimization.graph import build_grid_graph
from optimization.explanation import explain_assignments
from optimization.metrics import totals
from optimization.problem import Stop, Vehicle
from optimization.google_routes import fetch_live_travel_matrix
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


def route_map(points: pd.DataFrame, route_result: dict | None, capacity: float, selected_vehicle_id: str | None = None) -> pdk.Deck:
    """Map simulated bin state and planned truck paths; all paths use the visible bin positions."""
    def fill_color(fill: float) -> list[int]:
        if fill >= 90:
            return [211, 47, 47]
        if fill >= 70:
            return [239, 108, 0]
        if fill >= 40:
            return [251, 192, 45]
        return [46, 125, 50]

    bins_layer = points.assign(
        color=points["current_fill_level_pct"].map(fill_color),
        label="B",
    )
    center = {"lat": float(points.latitude.mean()), "lon": float(points.longitude.mean())}
    layers: list = [
        pdk.Layer("ScatterplotLayer", bins_layer, get_position="[longitude, latitude]", get_fill_color="color", get_radius=105, get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True, pickable=True),
        pdk.Layer("TextLayer", bins_layer, get_position="[longitude, latitude]", get_text="label", get_color=[255, 255, 255], get_size=12, get_alignment_baseline="center"),
    ]
    depot = [center["lon"], center["lat"]]
    if route_result:
        by_id = points.set_index("bin_id")
        paths, trucks = [], []
        colors = [[30, 136, 229], [123, 31, 162], [0, 121, 107], [251, 140, 0]]
        fleet_by_id = {vehicle["vehicle_id"]: vehicle for vehicle in route_result.get("fleet_snapshot", [])}
        explanation_by_bin = {item["bin_id"]: item for item in route_result.get("assignment_explanations", [])}
        for index, route in enumerate(route_result["routes"]):
            stop_ids = [stop["bin_id"] for stop in route["stops"]]
            fleet_state = fleet_by_id.get(route["vehicle_id"], {})
            start_node = route["road_path"][0] if route["road_path"] else fleet_state.get("mock_gps_node", "n0_0")
            start_row, start_col = (int(part) for part in start_node[1:].split("_"))
            start_position = [float(fleet_state.get("longitude", 77.55 + start_col * .003)), float(fleet_state.get("latitude", 12.90 + start_row * .003))]
            positions = [start_position] + [[float(by_id.loc[s].longitude), float(by_id.loc[s].latitude)] for s in stop_ids] + [depot]
            remaining = max(0.0, capacity - route.get("starting_load_kg", 0.0) - sum(stop["demand_kg"] for stop in route["stops"]))
            color = colors[index % len(colors)]
            next_stop = route["stops"][0] if stop_ids else None
            next_explanation = explanation_by_bin.get(next_stop["bin_id"], {}) if next_stop else {}
            next_eta = next_explanation.get("estimated_arrival_minutes_from_route_start")
            next_label = f"Next {next_stop['bin_id']} · +{next_eta:.1f} min" if next_stop and next_eta is not None else "No assigned pickup"
            trucks.append({"longitude": start_position[0], "latitude": start_position[1], "label": "T", "status": "Active on pickup route" if stop_ids else "Idle", "color": [21, 101, 192] if stop_ids else [117, 117, 117], "vehicle": route["vehicle_id"], "next_label": next_label, "remaining": remaining})
        if paths:
            layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_color="color", get_width=5, width_min_pixels=3, pickable=True))
        if trucks:
            layers.append(pdk.Layer("ScatterplotLayer", trucks, get_position="[longitude, latitude]", get_fill_color="color", get_radius=180, get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True, pickable=True))
            layers.append(pdk.Layer("TextLayer", trucks, get_position="[longitude, latitude]", get_text="label", get_color=[255, 255, 255], get_size=14, get_alignment_baseline="center", pickable=True))
    depot_layer = pd.DataFrame([{"longitude": depot[0], "latitude": depot[1], "label": "DEPOT"}])
    layers.append(pdk.Layer("ScatterplotLayer", depot_layer, get_position="[longitude, latitude]", get_fill_color=[33, 33, 33], get_radius=190))
    layers.append(pdk.Layer("TextLayer", depot_layer, get_position="[longitude, latitude]", get_text="label", get_color=[0, 0, 0], get_size=14, get_alignment_baseline="top"))
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=13, pitch=0), tooltip={"html": "<b>{bin_id}</b><br/>Fill: {current_fill_level_pct}%<br/>{urgency_category}<b>{vehicle}</b><br/>{status}<br/>{next_label}<br/>Free capacity: {remaining} kg", "style": {"backgroundColor": "#fff", "color": "#222"}}, map_style="light")


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


def google_routes_server_key() -> str | None:
    """Read a server-side Routes key; browser-referrer keys cannot authenticate backend calls."""
    try:
        key = st.secrets.get("GOOGLE_ROUTES_API_KEY")
        if key:
            return str(key)
    except FileNotFoundError:
        pass
    key = os.getenv("GOOGLE_ROUTES_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_ROUTES_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"') or None
    return None


def render_google_map(points: pd.DataFrame, route_result: dict | None, capacity: float, selected_vehicle_id: str | None = None) -> None:
    key = google_maps_key()
    if not key:
        st.warning("Add `GOOGLE_MAPS_API_KEY` to `.env` to use Google Maps. Showing the local fallback map until a key is configured.")
        st.pydeck_chart(route_map(points, route_result, capacity, selected_vehicle_id), use_container_width=True)
        return
    bins = [{
        "id": row.bin_id, "lat": float(row.latitude), "lng": float(row.longitude),
        "zone": str(getattr(row, "zone_id", "Unknown zone")),
        "urgency": str(row.urgency_category), "fill": round(float(row.current_fill_level_pct), 1),
        "urgency_probability": round(float(getattr(row, "urgency_probability", 0.0)) * 100, 1),
        "needs_collection": bool(getattr(row, "predicted_collection_required", False)),
        "color": "#d32f2f" if float(row.current_fill_level_pct) >= 90 else "#ef6c00" if float(row.current_fill_level_pct) >= 70 else "#fbc02d" if float(row.current_fill_level_pct) >= 40 else "#2e7d32",
    } for _, row in points.iterrows()]
    center = {"lat": float(points.latitude.mean()), "lng": float(points.longitude.mean())}
    depot = center.copy()
    routes, trucks = [], []
    if route_result:
        by_id = points.set_index("bin_id")
        colors = ["#1e88e5", "#7b1fa2", "#00796b", "#fb8c00"]
        fleet_by_id = {vehicle["vehicle_id"]: vehicle for vehicle in route_result.get("fleet_snapshot", [])}
        explanation_by_bin = {item["bin_id"]: item for item in route_result.get("assignment_explanations", [])}
        for index, route in enumerate(route_result["routes"]):
            stops = route["stops"]
            state = fleet_by_id.get(route["vehicle_id"], {})
            start = {"lat": float(state.get("latitude", depot["lat"])), "lng": float(state.get("longitude", depot["lng"]))}
            path = [start] + [{"lat": float(by_id.loc[s["bin_id"]].latitude), "lng": float(by_id.loc[s["bin_id"]].longitude)} for s in stops] + [depot]
            remaining = max(0.0, capacity - float(route.get("starting_load_kg", 0.0)) - sum(float(stop["demand_kg"]) for stop in stops))
            color = colors[index % len(colors)]
            if stops:
                routes.append({"path": path, "color": color, "vehicle": route["vehicle_id"]})
            first_stop = stops[0] if stops else None
            first_details = explanation_by_bin.get(first_stop["bin_id"], {}) if first_stop else {}
            assignment_comparison = sorted(
                [
                    {
                        "vehicle": candidate.get("vehicle_id"),
                        "eta_minutes": candidate.get("approach_minutes"),
                        "feasible": bool(candidate.get("feasible", False)),
                        "reason": candidate.get("infeasible_reason"),
                        "selected": candidate.get("vehicle_id") == route["vehicle_id"],
                    }
                    for candidate in first_details.get("vehicle_insertion_estimates", [])
                ],
                key=lambda item: item["eta_minutes"] if item["eta_minutes"] is not None else float("inf"),
            )
            comparison_uses_live_traffic = bool(
                route_result.get("assignment_live_pairs")
                and route_result.get("assignment_live_pairs") == route_result.get("assignment_expected_pairs")
            )
            # Every active truck remains visible at its mock GPS location, even if it has no stops.
            trucks.append({
                "lat": start["lat"], "lng": start["lng"], "vehicle": route["vehicle_id"],
                "remaining": round(remaining, 1), "color": color, "has_route": bool(stops), "number": index + 1,
                "mock_gps_node": state.get("mock_gps_node", "n0_0"),
                "current_load": round(float(route.get("starting_load_kg", 0.0)), 1),
                "route_distance": round(float(route.get("distance_km", 0.0)), 2),
                "route_duration": round(float(route.get("duration_minutes", 0.0)), 1),
                "next_bin": first_stop["bin_id"] if first_stop else None,
                "next_eta": first_details.get("estimated_arrival_minutes_from_route_start"),
                "assignment_comparison": assignment_comparison,
                "comparison_traffic_source": "Google Routes traffic-aware ETA" if comparison_uses_live_traffic else "simulated road-time estimate",
                "route_summary": " → ".join(stop["bin_id"] for stop in stops) if stops else "No assigned pickups",
                "stop_details": [
                    {
                        "bin": stop["bin_id"],
                        "from": state.get("mock_gps_node", "truck GPS") if stop_index == 0 else stops[stop_index - 1]["bin_id"],
                        "node": stop["node"],
                        "eta": explanation_by_bin.get(stop["bin_id"], {}).get("estimated_arrival_minutes_from_route_start"),
                        "simulated_leg_distance_km": explanation_by_bin.get(stop["bin_id"], {}).get("leg_distance_from_previous_stop_km"),
                        "urgency": stop["urgency"],
                        "service_minutes": float(stop.get("service_minutes", 0.0)),
                    }
                    for stop_index, stop in enumerate(stops)
                ],
            })
    payload = json.dumps({"bins": bins, "center": center, "depot": depot, "routes": routes, "trucks": trucks, "selected_vehicle_id": selected_vehicle_id})
    map_html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;font:12px Arial,sans-serif;margin:0 0 7px;color:#333">
      <span><b>Bins</b></span><span style="color:#2e7d32">●</span><span>0–39% full</span><span style="color:#fbc02d">●</span><span>40–69%</span><span style="color:#ef6c00">●</span><span>70–89%</span><span style="color:#d32f2f">●</span><span>90–100%</span>
      <span style="margin-left:8px"><b>Trucks</b></span><span style="color:#757575">● T</span><span>Idle</span><span style="color:#1565c0">● T</span><span>Pickup assigned</span>
    </div>
    <div id="map-status" style="font:12px sans-serif;margin:0 0 7px;color:#555">Loading map...</div>
    <div id="map" style="height:560px;width:100%;border-radius:10px"></div>
    <script>window.ecorouteData = {payload};</script>
    <script>
      let googleMapsAuthFailed = false;
      window.gm_authFailure = () => {{
        googleMapsAuthFailed = true;
        const mapStatus = document.getElementById('map-status');
        if (mapStatus) mapStatus.textContent = 'Google Maps rejected GOOGLE_MAPS_API_KEY. Check that Maps JavaScript API is enabled and the browser key allows this dashboard URL.';
      }};
      async function initEcoRouteMap() {{
        const d = window.ecorouteData;
        const status = document.getElementById('map-status');
        const map = new google.maps.Map(document.getElementById('map'), {{center:d.center, zoom:13, mapTypeControl:false, streetViewControl:false, mapId:'DEMO_MAP_ID'}});
        const info = new google.maps.InfoWindow();
        const fleetBounds = new google.maps.LatLngBounds();
        const selectedRoute = d.routes.find(route => route.vehicle === d.selected_vehicle_id);
        const selectedTruck = d.trucks.find(truck => truck.vehicle === d.selected_vehicle_id);
        if (selectedRoute) selectedRoute.path.forEach(point => fleetBounds.extend(point));
        else if (selectedTruck) fleetBounds.extend(selectedTruck);
        else d.bins.forEach(bin => fleetBounds.extend(bin));
        new google.maps.Marker({{position:d.depot, map, title:'Depot', label:{{text:'D', color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.CIRCLE, fillColor:'#212121', fillOpacity:1, strokeColor:'#fff', strokeWeight:2, scale:10}}}});
        d.bins.forEach(bin => {{
          const marker = new google.maps.Marker({{position:bin, map, title:`${{bin.id}} — ${{bin.fill}}% full`, label:{{text:'B', color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.CIRCLE, fillColor:bin.color, fillOpacity:1, strokeColor:'#fff', strokeWeight:2, scale:10}}}});
          marker.addListener('click', () => {{ info.setContent(`<div style="font:13px Arial;line-height:1.55;min-width:190px"><b>🗑️ ${{bin.id}}</b><br><b>Fill:</b> ${{bin.fill}}%<br><b>Zone:</b> ${{bin.zone}}<br><b>Urgency:</b> ${{bin.urgency}}<br><b>Collection needed:</b> ${{bin.needs_collection ? 'Yes' : 'No'}}<br><b>Urgency probability:</b> ${{bin.urgency_probability}}%<br><b>Coordinates:</b> ${{bin.lat.toFixed(5)}}, ${{bin.lng.toFixed(5)}}</div>`); info.open({{map, anchor:marker}}); }});
        }});
        const {{Route}} = await google.maps.importLibrary('routes');
        let roadRouteCount = 0;
        let roadRouteFailures = 0;
        const roadRouteErrors = [];
        const routeLines = {{}};
        const showRouteInfo = new Set();
        const routeBounds = {{}};
        const routeTracks = {{}};
        const truckPopup = truck => {{
          const assignmentRows = truck.assignment_comparison && truck.assignment_comparison.length
            ? `<hr style="margin:8px 0"><b>Why this truck for ${{truck.next_bin}}?</b><br><small>Approach time from each truck's mock GPS · ${{truck.comparison_traffic_source}}; capacity and route limits checked.</small><ol style="padding-left:20px;margin:4px 0">${{truck.assignment_comparison.map(candidate => `<li><b>${{candidate.vehicle}}</b>: ${{candidate.eta_minutes == null ? 'ETA unavailable' : `+${{Number(candidate.eta_minutes).toFixed(1)}} min`}} — ${{candidate.selected ? 'assigned to this bin' : candidate.feasible ? 'feasible, but another truck was assigned' : `not eligible (${{candidate.reason || 'capacity/route constraint'}})`}}</li>`).join('')}}</ol>`
            : '';
          let googleElapsedMinutes = 0;
          const stopRows = truck.has_route ? truck.stop_details.map((stop, index) => {{
            const liveLeg = truck.google_legs && truck.google_legs[index];
            let legSummary;
            let arrivalSummary;
            if (liveLeg) {{
              googleElapsedMinutes += liveLeg.duration_minutes;
              legSummary = `Google road leg: ${{liveLeg.distance_km.toFixed(2)}} km · ${{liveLeg.duration_minutes.toFixed(1)}} min`;
              arrivalSummary = `traffic-aware ETA +${{googleElapsedMinutes.toFixed(1)}} min`;
            }} else {{
              legSummary = `simulated leg: ${{Number(stop.simulated_leg_distance_km || 0).toFixed(2)}} km`;
              arrivalSummary = `simulated ETA +${{Number(stop.eta).toFixed(1)}} min`;
            }}
            googleElapsedMinutes += Number(stop.service_minutes || 0);
            return `<li><b>${{stop.bin}}</b> (${{stop.node}}) from ${{stop.from}} · ${{stop.urgency}} · ${{legSummary}} · ${{arrivalSummary}}</li>`;
          }}).join('') : '';
          const googleRouteSummary = truck.google_route
            ? `<br><b>Google road route:</b> ${{truck.google_route.distance_km.toFixed(2)}} km · ${{truck.google_route.duration_minutes.toFixed(1)}} min drive (traffic-aware, includes return to depot)`
            : '';
          const truckInfo = truck.has_route
            ? `<b>Status:</b> Active on pickup route<br><b>Next pickup:</b> ${{truck.next_bin}} (+${{truck.next_eta}} min simulated ETA)<br><b>Planned stops:</b> ${{truck.route_summary}}<br><b>Current mock GPS:</b> ${{truck.mock_gps_node}} (${{truck.lat.toFixed(5)}}, ${{truck.lng.toFixed(5)}})<br><b>Current load:</b> ${{truck.current_load}} kg<br><b>Free capacity after planned pickups:</b> ${{truck.remaining}} kg<br><button id="check-route-${{truck.number}}" style="margin-top:8px;padding:7px 10px;background:#1565c0;color:#fff;border:0;border-radius:5px;cursor:pointer">Check optimized route</button>`
            : `<b>Status:</b> Idle — no pickup assigned<br><b>Current mock GPS:</b> ${{truck.mock_gps_node}} (${{truck.lat.toFixed(5)}}, ${{truck.lng.toFixed(5)}})<br><b>Current load:</b> ${{truck.current_load}} kg<br><b>Free capacity:</b> ${{truck.remaining}} kg`;
          return `<div style="font:13px Arial;line-height:1.55;max-width:380px"><b>🚚 ${{truck.vehicle}}</b><br>${{truckInfo}}${{truck.has_route ? assignmentRows : ''}}${{truck.has_route && showRouteInfo.has(truck.vehicle) ? `<hr style="margin:8px 0"><b>Optimized route details</b><br><b>Optimizer estimate:</b> ${{truck.route_distance}} km · ${{truck.route_duration}} min (simulated)${{googleRouteSummary}}<br><b>Pickup legs and arrival times:</b><ol style="padding-left:20px;margin:4px 0">${{stopRows}}</ol>` : ''}}</div>`;
        }};
        const revealOptimizedRoute = truck => {{
          Object.values(routeLines).flat().forEach(line => line.setMap(null));
          showRouteInfo.clear();
          showRouteInfo.add(truck.vehicle);
          let lines = routeLines[truck.vehicle] || [];
          // Keep a visible road-following fallback ready if a Routes polyline could not be created.
          if (!lines.length && routeTracks[truck.vehicle]?.length) {{
            const fallbackLine = new google.maps.Polyline({{path:routeTracks[truck.vehicle], geodesic:true, strokeColor:truck.color, strokeOpacity:.95, strokeWeight:6}});
            routeLines[truck.vehicle] = [fallbackLine];
            lines = routeLines[truck.vehicle];
          }}
          lines.forEach(line => {{ line.setOptions({{strokeOpacity:.98, strokeWeight:7}}); line.setVisible(true); line.setMap(map); }});
          if (routeBounds[truck.vehicle]) map.fitBounds(routeBounds[truck.vehicle], 56);
          const marker = truckMarkers[truck.vehicle];
          info.setContent(truckPopup(truck));
          info.open({{map, ...(marker ? {{anchor:marker}} : {{position:truck}})}});
        }};
        info.addListener('domready', () => {{
          document.querySelectorAll('[id^="check-route-"]').forEach(button => {{
            button.onclick = event => {{
              event.preventDefault();
              event.stopPropagation();
              const truck = d.trucks.find(item => `check-route-${{item.number}}` === button.id);
              if (!truck) return;
              revealOptimizedRoute(truck);
            }};
          }});
        }});
        const truckMarkers = {{}};
        for (const route of d.routes) {{
          try {{
            let completeTrack = [];
            // Routes API accepts a limited number of intermediate waypoints. Split long truck runs
            // into contiguous chunks, preserving every planned stop and road-following geometry.
            for (let start = 0; start < route.path.length - 1;) {{
              const end = Math.min(start + 24, route.path.length - 1);
              const request = {{origin:route.path[start], destination:route.path[end], intermediates:route.path.slice(start + 1, end), travelMode:'DRIVING', routingPreference:'TRAFFIC_AWARE', fields:['path','legs']}};
              const {{routes}} = await Route.computeRoutes(request);
              if (!routes || !routes.length) throw new Error('No driving route returned');
              const truck = d.trucks.find(item => item.vehicle === route.vehicle);
              const computedRoute = routes[0];
              const legMetrics = (computedRoute.legs || []).map(leg => ({{distance_km:leg.distanceMeters / 1000, duration_minutes:leg.durationMillis / 60000}}));
              if (legMetrics.length !== end - start || legMetrics.some(leg => !Number.isFinite(leg.distance_km) || !Number.isFinite(leg.duration_minutes))) throw new Error('Google route leg distance or ETA was missing');
              if (truck) {{
                truck.google_legs = (truck.google_legs || []).concat(legMetrics);
                truck.google_route = truck.google_legs.reduce((summary, leg) => ({{distance_km:summary.distance_km + leg.distance_km, duration_minutes:summary.duration_minutes + leg.duration_minutes}}), {{distance_km:0, duration_minutes:0}});
              }}
              const focused = !d.selected_vehicle_id || route.vehicle === d.selected_vehicle_id;
              const bounds = routeBounds[route.vehicle] || (routeBounds[route.vehicle] = new google.maps.LatLngBounds());
              route.path.forEach(point => bounds.extend(point));
              computedRoute.createPolylines().forEach(polyline => {{
                polyline.setOptions({{strokeColor:route.color, strokeOpacity:focused ? .95 : .18, strokeWeight:focused ? 6 : 2}});
                polyline.setMap(null);
                (routeLines[route.vehicle] || (routeLines[route.vehicle] = [])).push(polyline);
                if (truck) polyline.addListener('click', event => {{ info.setContent(truckPopup(truck)); info.open({{map, position:event.latLng}}); }});
              }});
              const legTrack = (computedRoute.path || []).map(point => ({{
                lat:typeof point.lat === 'function' ? point.lat() : point.lat,
                lng:typeof point.lng === 'function' ? point.lng() : point.lng,
              }}));
              completeTrack = completeTrack.concat(completeTrack.length ? legTrack.slice(1) : legTrack);
              start = end;
            }}
            routeTracks[route.vehicle] = completeTrack;
            if (completeTrack.length) {{
              const bounds = routeBounds[route.vehicle] || (routeBounds[route.vehicle] = new google.maps.LatLngBounds());
              completeTrack.forEach(point => bounds.extend(point));
            }}
            // Keep route geometry and metrics ready, but only draw the route when requested.
            roadRouteCount += 1;
          }} catch (error) {{
            console.error('Google road route unavailable for ' + route.vehicle, error);
            roadRouteErrors.push(route.vehicle + ': ' + (error && error.message ? error.message : String(error)));
            const truck = d.trucks.find(item => item.vehicle === route.vehicle);
            if (truck) {{ delete truck.google_legs; delete truck.google_route; }}
            const focused = !d.selected_vehicle_id || route.vehicle === d.selected_vehicle_id;
            const fallbackLine = new google.maps.Polyline({{path:route.path, geodesic:true, strokeColor:route.color, strokeOpacity:.9, strokeWeight:4, map:null}});
            routeLines[route.vehicle] = [fallbackLine];
            if (truck) fallbackLine.addListener('click', event => {{ info.setContent(truckPopup(truck)); info.open({{map, position:event.latLng}}); }});
            roadRouteFailures += 1;
          }}
        }}
        d.trucks.forEach((truck, index) => {{
          if (!selectedRoute && !selectedTruck) fleetBounds.extend(truck);
          const markerTitle = truck.next_bin ? `${{truck.vehicle}} → ${{truck.next_bin}} (+${{truck.next_eta}} min)` : `${{truck.vehicle}} — idle`;
          const isSelected = truck.vehicle === d.selected_vehicle_id;
          const marker = new google.maps.Marker({{position:truck, map, title:markerTitle, zIndex:isSelected ? 1000 : index, label:{{text:'T', color:'white', fontWeight:'bold'}}, icon:{{path:google.maps.SymbolPath.CIRCLE, fillColor:truck.has_route ? '#1565c0' : '#757575', fillOpacity:1, strokeColor:'#fff', strokeWeight:isSelected ? 3 : 2, scale:11}}}});
          truckMarkers[truck.vehicle] = marker;
          marker.addListener('click', () => {{ info.setContent(truckPopup(truck)); info.open({{map, anchor:marker}}); }});
        }});
        if (d.bins.length || d.trucks.length) map.fitBounds(fleetBounds, 48);
        status.textContent = googleMapsAuthFailed
          ? 'Google Maps rejected GOOGLE_MAPS_API_KEY. Check that Maps JavaScript API is enabled and the browser key allows this dashboard URL.'
          : roadRouteFailures === 0
            ? `Map ready. Google road route available for ${{roadRouteCount}} active truck(s); route lines appear when you check a truck's optimized route.`
            : `Map ready. Google Routes failed for ${{roadRouteFailures}} truck(s); ${{roadRouteErrors.join(' | ')}}`;
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
    truck_count = st.slider("Available trucks", 1, 10, 10)
    truck_capacity = st.number_input("Truck capacity (kg)", min_value=100.0, value=1600.0, step=100.0)
    traffic_multiplier = st.slider("Simulated traffic level", 1.0, 3.0, 1.0, 0.1, help="Scales the graph's reproducible per-road congestion estimates; this is not live traffic.")
    truck_nodes = []
    truck_loads = []
    graph_nodes = [f"n{r}_{c}" for r in range(8) for c in range(8)]
    for i in range(truck_count):
        default_node = "n1_1" if i == 0 else "n6_6" if i == 1 else graph_nodes[(i * 13) % len(graph_nodes)]
        truck_nodes.append(st.selectbox(f"Truck {i + 1} mock GPS location", graph_nodes, index=graph_nodes.index(default_node)))
        truck_loads.append(st.number_input(f"Truck {i + 1} current load (kg)", min_value=0.0, max_value=float(truck_capacity), value=0.0, step=50.0))
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
    st.caption("Each bin is assigned to the closest truck by traffic-weighted road time, as long as that truck has enough payload and route time. Capacity is reserved as assignments are made; OR-Tools then orders pickups for each truck.")
    if st.button("Optimize current planning cycle", type="primary"):
        graph = build_grid_graph(seed=int(seed), traffic_multiplier=float(traffic_multiplier))
        stops = [Stop(bin_id=row.bin_id, node=row.node, demand_kg=float(row.demand_kg), urgency=row.urgency_category) for _, row in selected.iterrows()]
        vehicles = [Vehicle(f"truck-{i + 1}", float(truck_capacity), current_node=truck_nodes[i], current_load_kg=truck_loads[i], max_duration_minutes=480) for i in range(truck_count)]
        live_travel_matrix = {}
        expected_matrix_pairs = len(vehicles) * len(stops)
        matrix_error = None
        routes_key = google_routes_server_key()
        if routes_key:
            try:
                live_travel_matrix = fetch_live_travel_matrix(routes_key, graph, vehicles, stops)
                if not live_travel_matrix:
                    matrix_error = "Google Routes returned no usable truck-to-bin routes."
            except Exception as exc:
                matrix_error = str(exc)
        else:
            matrix_error = "Set GOOGLE_ROUTES_API_KEY to a server-restricted Routes API key to use live traffic for assignment."
        result = solve(
            graph, "n0_0", stops, vehicles, time_limit_seconds=int(solver_seconds),
            live_travel_matrix=live_travel_matrix or None,
        )
        metrics = totals(result)
        st.session_state["selected_route_vehicle"] = None
        fleet_snapshot = []
        for vehicle in vehicles:
            gps = graph.nodes[vehicle.current_node]
            fleet_snapshot.append({
                "vehicle_id": vehicle.vehicle_id, "mock_gps_node": vehicle.current_node,
                "latitude": gps["latitude"], "longitude": gps["longitude"],
                "capacity_kg": vehicle.capacity_kg, "current_load_kg": vehicle.current_load_kg,
                "free_capacity_kg": vehicle.capacity_kg - vehicle.current_load_kg,
            })
        st.session_state["route_result"] = result.to_dict() | {
            "metrics": metrics,
            "fleet_snapshot": fleet_snapshot,
            "assignment_explanations": explain_assignments(
                graph, "n0_0", result.routes, vehicles, result.solver, live_travel_matrix
            ),
            "rolling_interval": rolling_interval,
            "solver_budget": solver_seconds,
            "assignment_live_pairs": len(live_travel_matrix),
            "assignment_expected_pairs": expected_matrix_pairs,
            "assignment_matrix_error": matrix_error,
        }
    result = st.session_state.get("route_result")
    if result:
        status = result["status"]
        (st.success if status == "FEASIBLE" else st.error)(f"Solver status: {status} ({result['solver']})")
        live_pairs = int(result.get("assignment_live_pairs", 0))
        expected_pairs = int(result.get("assignment_expected_pairs", 0))
        if live_pairs and live_pairs == expected_pairs:
            st.success(f"Truck assignment used Google Routes traffic-aware ETAs for all {live_pairs} truck-to-bin pairs.")
        elif live_pairs:
            st.warning(f"Google Routes returned live ETAs for {live_pairs} of {expected_pairs} truck-to-bin pairs; remaining comparisons use simulated road times.")
        else:
            st.warning(f"Truck assignment is using simulated traffic. {result.get('assignment_matrix_error') or 'Google Routes live matrix is unavailable.'}")
        st.caption(f"Rolling-horizon cycle: refresh this plan every {result['rolling_interval']} minutes or immediately when a bin becomes critical. Solver budget: {result['solver_budget']} seconds.")
        if result.get("fleet_snapshot"):
            st.subheader("Mock GPS and truck capacity snapshot used for this plan")
            st.dataframe(pd.DataFrame(result["fleet_snapshot"]), use_container_width=True, hide_index=True)
        values = result["metrics"]
        e, f, g, h = st.columns(4)
        e.metric("Distance", f"{values['distance_km']:.2f} km")
        f.metric("Travel time", f"{values['travel_minutes']:.0f} min")
        g.metric("Fuel estimate", f"{values['fuel_liters_estimate']:.2f} L")
        h.metric("CO2 estimate", f"{values['co2_kg_estimate']:.2f} kg")
        st.subheader("Truck dispatch overview")
        st.caption("Open a truck below for its pickup sequence and estimated arrival times. Times are minutes from this plan's route start.")
        fleet_by_id = {truck["vehicle_id"]: truck for truck in result.get("fleet_snapshot", [])}
        explanation_by_bin = {item["bin_id"]: item for item in result.get("assignment_explanations", [])}
        dispatch_rows = []
        for route in result["routes"]:
            state = fleet_by_id.get(route["vehicle_id"], {})
            next_stop = route["stops"][0] if route["stops"] else None
            next_info = explanation_by_bin.get(next_stop["bin_id"], {}) if next_stop else {}
            dispatch_rows.append({
                "Truck": route["vehicle_id"], "Mock GPS": state.get("mock_gps_node", route["road_path"][0] if route["road_path"] else "n0_0"),
                "Next pickup": next_stop["bin_id"] if next_stop else "No assigned pickup",
                "Next ETA (+min)": round(float(next_info["estimated_arrival_minutes_from_route_start"]), 1) if next_info.get("estimated_arrival_minutes_from_route_start") is not None else None,
                "Assigned stops": len(route["stops"]),
                "Free capacity (kg)": round(float(state.get("free_capacity_kg", 0)) - sum(float(stop["demand_kg"]) for stop in route["stops"]), 1),
            })
        st.dataframe(pd.DataFrame(dispatch_rows), use_container_width=True, hide_index=True)
        for route in result["routes"]:
            state = fleet_by_id.get(route["vehicle_id"], {})
            stops = route["stops"]
            if not stops:
                with st.expander(f"{route['vehicle_id']} — no assigned pickups"):
                    st.info(f"This truck is available at mock GPS node {state.get('mock_gps_node', 'n0_0')} but has no pickups in this plan.")
                    if st.button(f"Show {route['vehicle_id']} GPS on map", key=f"show_route_{route['vehicle_id']}"):
                        st.session_state["selected_route_vehicle"] = route["vehicle_id"]
                continue
            first = stops[0]
            first_info = explanation_by_bin.get(first["bin_id"], {})
            eta = first_info.get("estimated_arrival_minutes_from_route_start", 0.0)
            with st.expander(f"{route['vehicle_id']} — next pickup {first['bin_id']} (+{eta:.1f} min)"):
                st.info(f"Dispatch notice: {route['vehicle_id']} should go to {first['bin_id']} first. Estimated arrival is +{eta:.1f} minutes from route start.")
                st.write(f"Starting at mock GPS node **{state.get('mock_gps_node', route['road_path'][0])}** · current load **{route['starting_load_kg']:.1f} kg** · route distance **{route['distance_km']:.2f} km** · estimated duration **{route['duration_minutes']:.1f} min**.")
                if st.button(f"Show {route['vehicle_id']} route on map", key=f"show_route_{route['vehicle_id']}"):
                    st.session_state["selected_route_vehicle"] = route["vehicle_id"]
                stop_rows = []
                for index, stop in enumerate(stops, start=1):
                    detail = explanation_by_bin.get(stop["bin_id"], {})
                    stop_rows.append({
                        "Order": index, "Pickup": stop["bin_id"], "Location node": stop["node"],
                        "Urgency": stop["urgency"],
                        "ETA from route start (simulated min)": detail.get("estimated_arrival_minutes_from_route_start"),
                        "Leg distance (simulated km)": detail.get("leg_distance_from_previous_stop_km"),
                        "Pickup demand (kg)": detail.get("demand_kg"),
                        "Truck load after pickup (kg)": detail.get("load_after_pickup_kg"),
                        "Free capacity after pickup (kg)": detail.get("capacity_remaining_after_pickup_kg"),
                    })
                st.dataframe(pd.DataFrame(stop_rows), use_container_width=True, hide_index=True)
        st.subheader("Why each bin was assigned to that truck")
        st.caption("Each bin goes to the closest truck by traffic-weighted road time, provided that truck has enough remaining payload. Bin assignments reserve capacity in urgency and load order; the optimizer then orders each truck's pickups.")
        if result.get("assignment_explanations"):
            explain_rows = []
            for item in result["assignment_explanations"]:
                options = "; ".join(
                    f"{candidate['vehicle_id']}: " + (f"{candidate['approach_minutes']:.1f} min from GPS, {candidate['capacity_remaining_after_kg']:.0f} kg free" if candidate["feasible"] else f"{candidate.get('approach_minutes', 0):.1f} min from GPS, not feasible")
                    for candidate in item["vehicle_insertion_estimates"]
                )
                explain_rows.append({
                    "Bin": item["bin_id"], "Urgency": item["urgency"], "Assigned truck": item["chosen_vehicle_id"],
                    "Mock GPS node": item["mock_gps"]["node"], "ETA from route start (min)": item["estimated_arrival_minutes_from_route_start"],
                    "Bin load (kg)": item["demand_kg"], "Truck load after pickup (kg)": item["load_after_pickup_kg"],
                    "Capacity free after pickup (kg)": item["capacity_remaining_after_pickup_kg"], "Truck insertion estimates": options,
                })
            st.dataframe(pd.DataFrame(explain_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No bins were assigned, so there are no per-bin assignment explanations.")
        if result["unserved"]:
            st.warning("Unserved bins")
            st.dataframe(pd.DataFrame(result["unserved"]), use_container_width=True, hide_index=True)
        with st.expander("Independent validation report"):
            st.json(result["validation"])
        st.subheader("Bins and planned truck routes")
        selected_route_vehicle = st.session_state.get("selected_route_vehicle")
        if selected_route_vehicle:
            st.info(f"Map focused on {selected_route_vehicle}. Its truck marker opens Google road distance and traffic-aware ETA for each pickup leg when Routes API is available.")
            if st.button("Show all truck routes on map", key="show_all_routes"):
                st.session_state["selected_route_vehicle"] = None
                selected_route_vehicle = None
        render_google_map(latest, result, float(truck_capacity), selected_route_vehicle)
        st.caption("Select a truck marker or route line for its ordered stops. Choose a truck's Show route on map button above to focus the map and open per-leg distance and ETA details. Google road distances and traffic-aware drive times come from Routes API; optimizer distance and ETA are simulated graph estimates.")

with tabs[3]:
    st.subheader("How to interpret this prototype")
    st.write("The dashboard uses synthetic records, truck locations, and traffic-adjusted simulated roads. It assigns bins using current graph location, remaining capacity, shift duration, and estimated travel time. Traffic is simulated, not live. Fuel and CO2 are estimates.")
    st.write("If a route cannot satisfy capacity or reachability constraints, the affected bin is shown as unserved along with the reason.")
