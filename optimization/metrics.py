from __future__ import annotations
from .problem import RoutingResult

def totals(result: RoutingResult, liters_per_100km: float = 32.0, kg_co2_per_liter: float = 2.68) -> dict:
    distance = sum(r.distance_km for r in result.routes)
    travel = sum(r.travel_minutes for r in result.routes)
    service = sum(r.service_minutes for r in result.routes)
    fuel = distance * liters_per_100km / 100
    return {"distance_km": round(distance, 3), "travel_minutes": round(travel, 3),
            "service_minutes": round(service, 3), "total_duration_minutes": round(travel + service, 3),
            "fuel_liters_estimate": round(fuel, 3), "co2_kg_estimate": round(fuel * kg_co2_per_liter, 3),
            "bins_served": sum(len(r.stops) for r in result.routes), "unserved_bins": len(result.unserved)}

def compare(reference: dict, candidate: dict) -> dict:
    out = {"reference": reference, "candidate": candidate, "note": "Savings are estimates; routes may serve different stop sets."}
    for key in ("distance_km", "travel_minutes", "fuel_liters_estimate", "co2_kg_estimate"):
        base, value = reference[key], candidate[key]
        out[f"{key}_change"] = round(value - base, 3)
        out[f"{key}_change_pct"] = None if base == 0 else round((value - base) / base * 100, 2)
    return out
