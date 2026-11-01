from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .problem import Stop, Vehicle


ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
MAX_MATRIX_ELEMENTS = 625


def fetch_live_travel_matrix(api_key: str, graph, vehicles: list[Vehicle], stops: list[Stop]) -> dict[tuple[str, str], dict[str, float]]:
    """Fetch traffic-aware truck-to-bin drive times and distances from Google Routes."""
    if not api_key:
        raise ValueError("Google Routes server API key is not configured.")
    if not vehicles or not stops:
        return {}

    def waypoint(node: str) -> dict:
        data = graph.nodes[node]
        return {"waypoint": {"location": {"latLng": {
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
        }}}}

    origins = [waypoint(vehicle.current_node) for vehicle in vehicles]
    matrix: dict[tuple[str, str], dict[str, float]] = {}
    chunk_size = max(1, MAX_MATRIX_ELEMENTS // len(origins))

    for offset in range(0, len(stops), chunk_size):
        stop_chunk = stops[offset:offset + chunk_size]
        body = {
            "origins": origins,
            "destinations": [waypoint(stop.node) for stop in stop_chunk],
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
        }
        request = Request(
            ROUTE_MATRIX_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "originIndex,destinationIndex,status,condition,distanceMeters,duration",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                elements = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Google Routes API returned HTTP {exc.code}.") from exc
        except URLError as exc:
            raise RuntimeError("Google Routes API could not be reached from this app server.") from exc

        if not isinstance(elements, list):
            raise RuntimeError("Google Routes API returned an unexpected route matrix response.")
        for element in elements:
            status = element.get("status") or {}
            if status.get("code", 0) != 0 or element.get("condition") != "ROUTE_EXISTS":
                continue
            duration = element.get("duration")
            distance_meters = element.get("distanceMeters")
            if duration is None or distance_meters is None:
                continue
            try:
                duration_seconds = float(str(duration).removesuffix("s"))
                origin_index = int(element["originIndex"])
                destination_index = int(element["destinationIndex"])
                stop = stop_chunk[destination_index]
                vehicle = vehicles[origin_index]
            except (ValueError, KeyError, IndexError):
                continue
            matrix[(vehicle.vehicle_id, stop.bin_id)] = {
                "duration_minutes": duration_seconds / 60,
                "distance_km": float(distance_meters) / 1000,
            }
    return matrix
