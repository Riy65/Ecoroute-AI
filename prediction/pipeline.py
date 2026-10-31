from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

FEATURES = ["fill_level_pct", "recent_fill_rate", "hours_since_collection", "waste_generation_rate_kg_per_hour", "hour_of_day", "day_of_week"]

def generate_data(output_dir: str = "data/raw", bins: int = 100, days: int = 30, seed: int = 42, horizon_hours: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    bin_rows = []
    start = pd.Timestamp("2026-01-01T00:00:00")
    for i in range(bins):
        rate = float(rng.uniform(.15, 1.4)); capacity = float(rng.choice([90, 120, 180, 240])); fill = float(rng.uniform(5, 55)); since = 0
        # Independent seeded coordinates deliberately avoid a visual grid: these are simulated bin placements.
        latitude = float(rng.uniform(12.900, 12.930)); longitude = float(rng.uniform(77.550, 77.585))
        bin_id = f"BIN-{i+1:03d}"; bin_rows.append({"bin_id": bin_id, "latitude": latitude, "longitude": longitude, "capacity_kg":capacity, "zone_id":f"Z{(i%5)+1}", "waste_generation_rate_kg_per_hour":rate, "simulated":True})
        future = []
        for h in range(days * 24 + horizon_hours):
            hour = (start + pd.Timedelta(hours=h)).hour
            actual_rate = rate * (1.15 if 8 <= hour <= 20 else .65) * float(rng.uniform(.85, 1.15))
            fill = min(100., fill + actual_rate / capacity * 100); since += 1
            collect = bool(fill >= 92 and rng.random() < .6)
            future.append((fill, since, actual_rate / capacity * 100, collect))
            if collect: fill = float(rng.uniform(2, 8)); since = 0
        for h in range(days * 24):
            fill, since, recent, collect = future[h]; future_fill = future[horizon_hours + h][0]
            ts = start + pd.Timedelta(hours=h)
            records.append({"bin_id":bin_id,"timestamp":ts,"fill_level_pct":round(fill,3),"estimated_waste_kg":round(fill/100*capacity,3),"hours_since_collection":since,"recent_fill_rate":round(recent,3),"waste_generation_rate_kg_per_hour":rate,"hour_of_day":ts.hour,"day_of_week":ts.dayofweek,"collection_event":collect,"future_fill_level_pct":round(future_fill,3),"collection_required_within_horizon":bool(future_fill >= 85)})
    bins_df, history = pd.DataFrame(bin_rows), pd.DataFrame(records)
    bins_df.to_csv(out / "bins.csv", index=False); history.to_csv(out / "bin_history.csv", index=False)
    return bins_df, history

def train(history: pd.DataFrame, model_path: str = "models/urgency.joblib") -> dict:
    required = set(FEATURES + ["collection_required_within_horizon", "timestamp"])
    if not required.issubset(history.columns): raise ValueError("history lacks required columns")
    history = history.sort_values("timestamp"); cut = int(len(history)*.8); x, y = history[FEATURES], history["collection_required_within_horizon"].astype(int)
    if y.iloc[:cut].nunique() < 2: raise ValueError("training target has only one class")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1).fit(x.iloc[:cut], y.iloc[:cut])
    pred = model.predict(x.iloc[cut:]); prob = model.predict_proba(x.iloc[cut:])[:,1]; p,r,f,_ = precision_recall_fscore_support(y.iloc[cut:],pred,average="binary",zero_division=0)
    metrics = {"precision":float(p),"recall":float(r),"f1":float(f),"roc_auc":float(roc_auc_score(y.iloc[cut:],prob)) if y.iloc[cut:].nunique()==2 else None,"train_rows":cut,"test_rows":len(y)-cut}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True); dump({"model":model,"features":FEATURES,"metrics":metrics},model_path); return metrics

def predict_latest(history: pd.DataFrame, bins_df: pd.DataFrame, model_path: str = "models/urgency.joblib") -> list[dict]:
    artifact=load(model_path); latest=history.sort_values("timestamp").groupby("bin_id").tail(1)
    if "waste_generation_rate_kg_per_hour" not in latest:
        latest=latest.merge(bins_df[["bin_id","waste_generation_rate_kg_per_hour"]],on="bin_id")
    prob=artifact["model"].predict_proba(latest[FEATURES])[:,1]; rows=[]
    for (_, row), p in zip(latest.iterrows(),prob):
        category = "CRITICAL" if row.fill_level_pct >= 90 or p >= .85 else "HIGH" if p >= .60 else "MEDIUM" if p >= .30 else "LOW"
        rows.append({"bin_id":row.bin_id,"current_fill_level_pct":round(float(row.fill_level_pct),2),"urgency_probability":round(float(p),4),"predicted_collection_required":bool(p>=.60),"urgency_category":category})
    return rows
