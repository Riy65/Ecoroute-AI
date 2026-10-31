from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Urgency = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class Stop:
    bin_id: str
    node: str
    demand_kg: float
    service_minutes: float = 4.0
    urgency: Urgency = "HIGH"
    time_window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.bin_id or not self.node:
            raise ValueError("bin_id and node are required")
        if self.demand_kg < 0 or self.service_minutes < 0:
            raise ValueError("demand and service time must be non-negative")
        if self.urgency not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise ValueError("invalid urgency")
        if self.time_window and (self.time_window[0] < 0 or self.time_window[1] < self.time_window[0]):
            raise ValueError("invalid time window")


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    capacity_kg: float
    available: bool = True
    max_distance_km: float | None = None
    max_duration_minutes: float | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_id or self.capacity_kg < 0:
            raise ValueError("vehicle needs an ID and non-negative capacity")
        if self.max_distance_km is not None and self.max_distance_km < 0:
            raise ValueError("max distance must be non-negative")
        if self.max_duration_minutes is not None and self.max_duration_minutes < 0:
            raise ValueError("max duration must be non-negative")


@dataclass
class Route:
    vehicle_id: str
    stops: list[Stop] = field(default_factory=list)
    road_path: list[str] = field(default_factory=list)
    distance_km: float = 0.0
    travel_minutes: float = 0.0
    service_minutes: float = 0.0
    waiting_minutes: float = 0.0

    @property
    def duration_minutes(self) -> float:
        return self.travel_minutes + self.service_minutes + self.waiting_minutes

    def to_dict(self) -> dict:
        return {**asdict(self), "duration_minutes": self.duration_minutes}


@dataclass
class RoutingResult:
    status: str
    solver: str
    routes: list[Route]
    unserved: list[dict]
    validation: dict
    runtime_seconds: float
    objective_distance_km: float

    def to_dict(self) -> dict:
        return {"status": self.status, "solver": self.solver,
                "routes": [r.to_dict() for r in self.routes], "unserved": self.unserved,
                "validation": self.validation, "runtime_seconds": self.runtime_seconds,
                "objective_distance_km": self.objective_distance_km}
