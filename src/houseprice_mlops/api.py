import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from houseprice_mlops.monitoring.metrics import request_monitor

PROJECT_ROOT = Path("/app")
MODEL_PATH = PROJECT_ROOT / "artifacts" / "cloud-run" / "baseline_model.joblib"

app = FastAPI(
    title="House Price Prediction API",
    description="A lightweight house price prediction service.",
    version="1.0.0",
)
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()

    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        raise

    finally:
        latency = time.perf_counter() - start_time

        request_monitor.record_request(
            status_code=status_code,
            latency=latency,
        )
model = None
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(ge=0, description="Median house age")
    AveRooms: float = Field(gt=0, description="Average rooms")
    AveBedrms: float = Field(gt=0, description="Average bedrooms")
    Population: float = Field(ge=0, description="Block group population")
    AveOccup: float = Field(gt=0, description="Average household occupancy")
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.DataFrame([data.model_dump()])

    try:
        prediction = float(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc

    return {
        "predicted_price": prediction*100000,
        "currency": "USD",
        "price_scale": "100,000 USD",
    }
@app.get("/metrics")
def get_metrics():
    return request_monitor.get_metrics()