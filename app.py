# pyrefly: ignore [missing-import]
import json
from pathlib import Path
from typing import Dict
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load model at startup ──
MODEL_NAME = "rf_pipeline_model"
MODEL_VERSION = "1"
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

# ── Config File Setup ──
CONFIG_PATH = Path("config.json")
DEFAULT_THRESHOLDS = {
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.75,
    "f1_score": 0.78,
    "latency_ms": 200.0,
}

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_THRESHOLDS)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(data: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)

# ── Schemas ──
class PredictRequest(BaseModel):
    Quantity: float
    UnitPrice: float
    Country: str

class PredictResponse(BaseModel):
    prediction: int
    label: str

class AddMetricThreshold(BaseModel):
    metric: str = Field(..., description="Metric name, e.g., 'roc_auc'")
    threshold: float = Field(..., description="Threshold value, e.g., 0.88")

# ── App ──
app = FastAPI(title="MLflow Model & Metric Thresholds API")

# Root endpoint
@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "endpoints": {
            "GET /config": "Read all metric thresholds from config file",
            "GET /config/{metric}": "Read threshold for a specific metric",
            "POST /config": "Add a new metric threshold to config file",
            "PUT /config": "Full update / replace all metric thresholds in config file",
            "PATCH /config": "Partial update of specific metric thresholds in config file",
            "DELETE /config/{metric}": "Remove a metric threshold from config file",
            "POST /predict": "Run model prediction"
        }
    }

# POST — ML Model Prediction
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    df = pd.DataFrame([data.model_dump()])
    pred = int(model.predict(df)[0])
    label = "High-Value" if pred == 1 else "Low-Value"
    return PredictResponse(prediction=pred, label=label)

# ── HTTP METHODS ON CONFIG FILE (METRIC THRESHOLDS) ──

# 1. GET — Read all thresholds or a specific metric threshold from config.json
@app.get("/config")
def get_all_thresholds():
    """Read and return all metric thresholds from config.json"""
    config = load_config()
    return {"status": "success", "thresholds": config}

@app.get("/config/{metric}")
def get_metric_threshold(metric: str):
    """Read threshold for a specific metric from config.json"""
    config = load_config()
    if metric not in config:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found in config.json")
    return {"metric": metric, "threshold": config[metric]}

# 2. POST — Create / Add a new metric threshold into config.json
@app.post("/config")
def add_metric_threshold(data: AddMetricThreshold):
    """Add a new metric threshold to config.json (fails if metric already exists)"""
    config = load_config()
    if data.metric in config:
        raise HTTPException(status_code=400, detail=f"Metric '{data.metric}' already exists in config.json. Use PUT or PATCH to update.")
    config[data.metric] = data.threshold
    save_config(config)
    return {"message": f"Added metric threshold for '{data.metric}'", "config": config}

# 3. PUT — Full Update / Replace entire config.json with a new set of thresholds
@app.put("/config")
def replace_all_thresholds(data: Dict[str, float]):
    """Replace all metric thresholds in config.json with the provided dictionary"""
    if not data:
        raise HTTPException(status_code=400, detail="Cannot replace config with an empty dictionary.")
    save_config(data)
    return {"message": "Entire thresholds config replaced successfully", "config": data}

# 4. PATCH — Partial Update: modify only specified metric thresholds without altering others
@app.patch("/config")
def patch_thresholds(data: Dict[str, float]):
    """Partially update only the provided metric thresholds in config.json"""
    config = load_config()
    for key, value in data.items():
        config[key] = value
    save_config(config)
    return {"message": "Updated specified metric thresholds", "config": config}

# 5. DELETE — Remove a metric threshold from config.json
@app.delete("/config/{metric}")
def delete_metric_threshold(metric: str):
    """Delete a specific metric threshold from config.json"""
    config = load_config()
    if metric not in config:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found in config.json")
    removed_val = config.pop(metric)
    save_config(config)
    return {"message": f"Deleted metric '{metric}' with threshold {removed_val}", "config": config}
