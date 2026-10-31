from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from optimization.graph import build_grid_graph
from optimization.problem import Stop, Vehicle
from optimization.solver import solve
from optimization.explanation import explain_assignments
from optimization.route_validator import validate_routes
from optimization.metrics import totals, compare
from prediction.pipeline import generate_data, train, predict_latest

app = FastAPI(title="EcoRoute AI", version="0.1.0", description="Synthetic-data backend for prediction-guided municipal collection routing.")
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data"/"raw"; MODELS=ROOT/"models"; EXP=ROOT/"experiments"

class GenerateRequest(BaseModel): bins: int = Field(default=100, ge=2); days: int = Field(default=30, ge=2); seed: int = 42
class VehicleIn(BaseModel): vehicle_id: str; capacity_kg: float = Field(ge=0); available: bool=True; max_distance_km: float|None=Field(default=None,ge=0); max_duration_minutes: float|None=Field(default=None,ge=0); current_node: str|None=None; current_load_kg: float=Field(default=0,ge=0)
class StopIn(BaseModel): bin_id: str; node: str; demand_kg: float=Field(ge=0); service_minutes: float=Field(default=4,ge=0); urgency: str="HIGH"
class OptimizeRequest(BaseModel): depot: str="n0_0"; stops: list[StopIn]; vehicles: list[VehicleIn]; rows: int=8; cols: int=8; seed:int=42; traffic_multiplier: float=Field(default=1.0,ge=1.0,le=5.0); solver_time_limit_seconds:int=Field(default=5,ge=1,le=60); reoptimization_interval_minutes:int=Field(default=15,ge=1,le=1440)

def read_data() -> tuple[pd.DataFrame,pd.DataFrame]:
    try: return pd.read_csv(RAW/"bins.csv"), pd.read_csv(RAW/"bin_history.csv",parse_dates=["timestamp"])
    except FileNotFoundError: raise HTTPException(404,"Generate synthetic data first")

@app.get("/health")
def health(): return {"status":"ok","service":"EcoRoute AI"}

@app.post("/api/data/generate")
def data_generate(request: GenerateRequest):
    bins, history=generate_data(str(RAW),request.bins,request.days,request.seed)
    return {"bins":len(bins),"history_rows":len(history),"simulated":True}

@app.get("/api/data/summary")
def data_summary():
    bins, h=read_data(); return {"bins":len(bins),"observations":len(h),"first_timestamp":str(h.timestamp.min()),"last_timestamp":str(h.timestamp.max()),"simulated":True}

@app.post("/api/model/train")
def model_train():
    _, history=read_data()
    try: metrics=train(history,str(MODELS/"urgency.joblib"))
    except ValueError as exc: raise HTTPException(422,str(exc))
    EXP.mkdir(exist_ok=True); (EXP/"model_evaluation.json").write_text(json.dumps(metrics,indent=2)); return metrics

@app.get("/api/model/evaluation")
def model_evaluation():
    try: return json.loads((EXP/"model_evaluation.json").read_text())
    except FileNotFoundError: raise HTTPException(404,"Train a model first")

@app.post("/api/predictions/run")
def predictions_run():
    bins, history=read_data()
    try: predictions=predict_latest(history,bins,str(MODELS/"urgency.joblib"))
    except FileNotFoundError: raise HTTPException(404,"Train a model first")
    EXP.mkdir(exist_ok=True); (EXP/"predictions.json").write_text(json.dumps(predictions,indent=2)); return predictions

@app.get("/api/predictions")
def predictions_get():
    try: return json.loads((EXP/"predictions.json").read_text())
    except FileNotFoundError: raise HTTPException(404,"Run predictions first")

def run_opt(request: OptimizeRequest):
    try:
        graph=build_grid_graph(request.rows,request.cols,request.seed,traffic_multiplier=request.traffic_multiplier)
        stops=[Stop(**x.model_dump()) for x in request.stops]; vehicles=[Vehicle(**x.model_dump()) for x in request.vehicles]
        return graph,stops,vehicles,solve(graph,request.depot,stops,vehicles,request.solver_time_limit_seconds)
    except ValueError as exc: raise HTTPException(422,str(exc))

@app.post("/api/routes/optimize")
def routes_optimize(request: OptimizeRequest):
    graph,stops,vehicles,result=run_opt(request)
    payload=result.to_dict()
    payload["metrics"]=totals(result)
    payload["fleet_snapshot"]=[
        {"vehicle_id":vehicle.vehicle_id,"available":vehicle.available,
         "mock_gps_node":vehicle.current_node or request.depot,
         "latitude":graph.nodes[vehicle.current_node or request.depot]["latitude"],
         "longitude":graph.nodes[vehicle.current_node or request.depot]["longitude"],
         "capacity_kg":vehicle.capacity_kg,"current_load_kg":vehicle.current_load_kg,
         "free_capacity_kg":vehicle.capacity_kg-vehicle.current_load_kg}
        for vehicle in vehicles
    ]
    payload["assignment_explanations"]=explain_assignments(graph,request.depot,result.routes,vehicles,result.solver)
    payload["rolling_horizon"]={"reoptimization_interval_minutes":request.reoptimization_interval_minutes,"trigger":"scheduled interval or bin urgency event"}
    return payload

@app.post("/api/routes/validate")
def routes_validate(request: OptimizeRequest):
    graph,stops,vehicles,result=run_opt(request); return validate_routes(graph,request.depot,result.routes,stops,vehicles)

@app.post("/api/routes/compare")
def routes_compare(request: OptimizeRequest):
    graph, stops, vehicles, optimized=run_opt(request)
    # Fixed manifest ordering is a transparent same-stop baseline.
    fixed=solve(graph,request.depot,sorted(stops,key=lambda x:x.bin_id),vehicles)
    return {"optimized":totals(optimized),"fixed_order_baseline":totals(fixed),"comparison":compare(totals(fixed),totals(optimized)),"same_stop_set":True}

@app.post("/api/simulation/run")
def simulation_run():
    bins, history = generate_data(str(RAW),bins=100,days=30,seed=42); train(history,str(MODELS/"urgency.joblib")); predictions=predict_latest(history,bins,str(MODELS/"urgency.joblib"))
    selected=[p for p in predictions if p["urgency_category"] in {"HIGH","CRITICAL"}]
    node_map={row.bin_id:f"n{idx//8}_{idx%8}" for idx,row in bins.reset_index(drop=True).iterrows() if idx<64}
    selected=[p for p in selected if p["bin_id"] in node_map]
    req=OptimizeRequest(stops=[StopIn(bin_id=p["bin_id"],node=node_map[p["bin_id"]],demand_kg=float(bins.loc[bins.bin_id.eq(p["bin_id"]),"capacity_kg"].iloc[0])*p["current_fill_level_pct"]/100,urgency=p["urgency_category"]) for p in selected],vehicles=[VehicleIn(vehicle_id="truck-1",capacity_kg=1600),VehicleIn(vehicle_id="truck-2",capacity_kg=1600)])
    _,_,_,route=run_opt(req); result={"prediction_count":len(predictions),"selected_count":len(selected),"route":route.to_dict(),"metrics":totals(route)}
    EXP.mkdir(exist_ok=True); (EXP/"latest_simulation.json").write_text(json.dumps(result,indent=2)); return result

@app.get("/api/experiments/latest")
def latest_experiment():
    try:return json.loads((EXP/"latest_simulation.json").read_text())
    except FileNotFoundError:raise HTTPException(404,"Run a simulation first")
