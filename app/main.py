import json
import logging
import os
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


# Which artifact family to serve; overridable without a code change
# (e.g. DEPLOYED_MODEL_GLOB="gb_v1_*.pkl" to roll back to the v1 model).
DEFAULT_MODEL_GLOB = "lgbm_v2_*.pkl"


def _load_deployed_model(app: FastAPI) -> None:
    """
    Load the latest deployed model artifact and its metadata into app.state.

    State keys set:
        app.state.model         – fitted estimator (v2: feature-builder +
                                  classifier pipeline taking raw columns)
        app.state.feature_list  – ordered list of feature column names the
                                  request must provide
        app.state.threshold     – classification threshold
        app.state.model_name    – logical model name (e.g. "gb", "lgbm")
        app.state.model_version – version string (e.g. "v1", "v2")
        app.state.native_nan    – True when the artifact handles NaN itself
                                  (v2 / LightGBM); the API must NOT impute
    """
    model_glob = os.environ.get("DEPLOYED_MODEL_GLOB", DEFAULT_MODEL_GLOB)
    model_files = sorted(MODELS_DIR.glob(model_glob))
    if not model_files:
        raise RuntimeError(
            f"No deployed model artifact matching {model_glob!r} in artifacts/models"
        )

    model_path = model_files[-1]
    meta_path = model_path.with_name(model_path.stem + "_meta.json")

    app.state.model = joblib.load(model_path)
    app.state.feature_list = None
    app.state.model_name = "gb"
    app.state.model_version = "v1"
    app.state.threshold = 0.5
    app.state.native_nan = False

    if meta_path.exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        app.state.feature_list = meta.get("feature_list")
        app.state.model_name = meta.get("model_name", app.state.model_name)
        app.state.model_version = meta.get("version", app.state.model_version)
        app.state.native_nan = meta.get("imputation") == "native"
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
def predict(payload: PredictionRequest, req: Request) -> PredictionResponse:
    model = getattr(req.app.state, "model", None)
    feature_list = getattr(req.app.state, "feature_list", None)
    threshold = getattr(req.app.state, "threshold", 0.5)
    model_name = getattr(req.app.state, "model_name", "gb")
    model_version = getattr(req.app.state, "model_version", "v1")
    native_nan = getattr(req.app.state, "native_nan", False)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_list is None:
        raise HTTPException(
            status_code=503,
            detail="Feature list not available in deployed metadata",
        )

    records = [t.model_dump() for t in payload.transactions]
    df = pd.DataFrame(records)

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features for inference: {missing}",
        )

    X = df[feature_list].copy()
    if not native_nan:
        # v1 contract only: v2 artifacts encode/impute internally, and
        # filling raw NaN with 0 would corrupt LightGBM's native handling.
        X = X.fillna(0.0)
    proba = model.predict_proba(X)[:, 1]

    preds: List[PredictionItem] = [
        PredictionItem(
            transaction_id=row.TransactionID,
            fraud_probability=float(p),
            fraud_flag=bool(p >= threshold),
        )
        for row, p in zip(payload.transactions, proba)
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
