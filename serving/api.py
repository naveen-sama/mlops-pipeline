"""
FastAPI model serving server.

Loads the registered MLflow model on startup, exposes prediction and health
endpoints, and emits Prometheus metrics via prometheus-fastapi-instrumentator.

Endpoints:
    POST /predict     — Run inference on a list of feature vectors
    GET  /health      — Liveness + readiness check
    GET  /model-info  — Current model name, version, and stage
    GET  /metrics     — Prometheus scrape endpoint (auto-added by instrumentator)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment at import time)
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME: str = os.getenv("MODEL_NAME", "MLOpsDemoModel")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")
MODEL_VERSION: str | None = os.getenv("MODEL_VERSION")  # e.g. "3"; overrides stage

# California Housing feature names (must match training order)
FEATURE_NAMES: list[str] = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    model: mlflow.pyfunc.PyFuncModel | None = None
    model_uri: str = ""
    model_name: str = MODEL_NAME
    model_version: str = "unknown"
    model_stage: str = "unknown"
    load_time: float = 0.0
    loaded: bool = False


state = ModelState()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """Resolve the model URI and load it into global state."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)

    if MODEL_VERSION:
        uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        version = MODEL_VERSION
        stage = "N/A"
    else:
        uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        # Resolve actual version from the stage alias
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        matched = [v for v in versions if v.current_stage == MODEL_STAGE]
        version = str(max(int(v.version) for v in matched)) if matched else "unknown"
        stage = MODEL_STAGE

    logger.info("Loading model from URI: %s", uri)
    t0 = time.time()
    state.model = mlflow.pyfunc.load_model(uri)
    state.load_time = time.time() - t0
    state.model_uri = uri
    state.model_version = version
    state.model_stage = stage
    state.loaded = True

    logger.info(
        "Model loaded — name=%s  version=%s  stage=%s  load_time=%.2fs",
        MODEL_NAME,
        version,
        stage,
        state.load_time,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release resources on shutdown."""
    try:
        _load_model()
    except Exception as exc:
        logger.error("Failed to load model on startup: %s", exc)
        # Allow the server to start even if MLflow is temporarily unavailable;
        # the /health endpoint will report not-ready.
    yield
    logger.info("Shutting down model server.")
    state.model = None
    state.loaded = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLOps Demo Model Server",
    description="Serves XGBoost California Housing predictions via MLflow Model Registry",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics — exposes /metrics endpoint automatically
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Accepts a list of feature rows.

    Each inner list must contain exactly 8 numeric values corresponding to:
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    """
    features: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of feature vectors (each with 8 values)",
        examples=[[[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]],
    )

    @field_validator("features")
    @classmethod
    def validate_feature_length(cls, v: list[list[float]]) -> list[list[float]]:
        expected = len(FEATURE_NAMES)
        for i, row in enumerate(v):
            if len(row) != expected:
                raise ValueError(
                    f"Row {i} has {len(row)} features; expected {expected} "
                    f"({', '.join(FEATURE_NAMES)})"
                )
        return v


class PredictResponse(BaseModel):
    predictions: list[float]
    model_name: str
    model_version: str
    model_stage: str
    n_samples: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_s: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    model_uri: str
    feature_names: list[str]
    load_time_s: float


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

_server_start = time.time()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, summary="Run model inference")
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Accept a JSON body with a `features` list of feature rows and return
    predicted median house values (in units of $100,000).
    """
    if not state.loaded or state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The server may still be initializing.",
        )

    try:
        df = pd.DataFrame(payload.features, columns=FEATURE_NAMES)
        t0 = time.perf_counter()
        raw_preds = state.model.predict(df)
        inference_ms = (time.perf_counter() - t0) * 1000

        predictions = [float(p) for p in np.array(raw_preds).flatten()]

        logger.info(
            "Predicted %d samples in %.2f ms | min=%.3f max=%.3f",
            len(predictions),
            inference_ms,
            min(predictions),
            max(predictions),
        )

        return PredictResponse(
            predictions=predictions,
            model_name=state.model_name,
            model_version=state.model_version,
            model_stage=state.model_stage,
            n_samples=len(predictions),
            inference_time_ms=round(inference_ms, 3),
        )

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """
    Returns HTTP 200 when the model is loaded and ready to serve predictions.
    Returns HTTP 503 when the model is not yet loaded.
    """
    uptime = time.time() - _server_start
    response = HealthResponse(
        status="healthy" if state.loaded else "initializing",
        model_loaded=state.loaded,
        model_name=state.model_name,
        model_version=state.model_version,
        uptime_s=round(uptime, 2),
    )
    if not state.loaded:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response


@app.get("/model-info", response_model=ModelInfoResponse, summary="Current model metadata")
async def model_info() -> ModelInfoResponse:
    """Returns metadata about the currently loaded model."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return ModelInfoResponse(
        model_name=state.model_name,
        model_version=state.model_version,
        model_stage=state.model_stage,
        model_uri=state.model_uri,
        feature_names=FEATURE_NAMES,
        load_time_s=round(state.load_time, 4),
    )


@app.post("/reload", summary="Reload model from registry", include_in_schema=True)
async def reload_model() -> dict[str, str]:
    """
    Trigger a hot-reload of the model from the MLflow registry.
    Useful when a new version is promoted to Production.
    """
    try:
        _load_model()
        return {
            "status": "reloaded",
            "model_name": state.model_name,
            "model_version": state.model_version,
        }
    except Exception as exc:
        logger.exception("Hot-reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {
        "service": "MLOps Demo Model Server",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }
