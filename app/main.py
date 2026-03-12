import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")

    TransactionID: int | None = Field(default=None)
    TransactionDT: float | None = Field(default=None)
    TransactionAmt: float


class PredictionRequest(BaseModel):
    transactions: List[Transaction]


class PredictionItem(BaseModel):
    transaction_id: int | None
    fraud_probability: float
    fraud_flag: bool


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    predictions: List[PredictionItem]


def _load_deployed_model(app: FastAPI) -> None:
    """
    Load the latest deployed model artifact and its metadata into app.state.

    State keys set:
        app.state.model         – fitted sklearn estimator
        app.state.feature_list  – ordered list of feature column names
        app.state.threshold     – classification threshold
        app.state.model_name    – logical model name (e.g. "gb")
        app.state.model_version – version string (e.g. "v1")
    """
    model_files = sorted(MODELS_DIR.glob("gb_v1_*.pkl"))
    if not model_files:
        raise RuntimeError("No deployed model artifact found in artifacts/models")

    model_path = model_files[-1]
    meta_path = model_path.with_name(model_path.stem + "_meta.json")

    app.state.model = joblib.load(model_path)
    app.state.feature_list = None
    app.state.model_name = "gb"
    app.state.model_version = "v1"
    app.state.threshold = 0.5

    if meta_path.exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        app.state.feature_list = meta.get("feature_list")
        metrics = meta.get("metrics") or {}
        app.state.threshold = float(metrics.get("best_threshold", 0.5))

    logger.info(
        "Model loaded: %s  (threshold=%.4f)",
        model_path.name,
        app.state.threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_deployed_model(app)
    yield


app = FastAPI(
    title="Fraud Detection Scoring API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness and readiness probe."""
    model_ready = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok" if model_ready else "unavailable",
        "model_loaded": model_ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, req: Request) -> PredictionResponse:
    model = getattr(req.app.state, "model", None)
    feature_list = getattr(req.app.state, "feature_list", None)
    threshold = getattr(req.app.state, "threshold", 0.5)
    model_name = getattr(req.app.state, "model_name", "gb")
    model_version = getattr(req.app.state, "model_version", "v1")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_list is None:
        raise HTTPException(
            status_code=503,
            detail="Feature list not available in deployed metadata",
        )

    records = [t.model_dump() for t in request.transactions]
    df = pd.DataFrame(records)

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features for inference: {missing}",
        )

    X = df[feature_list].copy().fillna(0.0)
    proba = model.predict_proba(X)[:, 1]

    preds: List[PredictionItem] = [
        PredictionItem(
            transaction_id=row.TransactionID,
            fraud_probability=float(p),
            fraud_flag=bool(p >= threshold),
        )
        for row, p in zip(request.transactions, proba)
    ]

    logger.info(
        "Scored %d transactions  threshold=%.4f  flagged=%d",
        len(preds),
        threshold,
        sum(p.fraud_flag for p in preds),
    )

    return PredictionResponse(
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        predictions=preds,
    )
