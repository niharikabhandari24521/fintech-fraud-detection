"""FastAPI inference service for real-time credit card fraud detection."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MERCHANT_CATEGORIES, METRICS_PATH, MODEL_PATH
from src.predict import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="SentinelPay Fraud Detection API",
    description=(
        "Real-time credit card fraud scoring microservice backed by an Optuna-tuned "
        "LightGBM + CatBoost Platt-calibrated ensemble with exact TreeSHAP explanations."
    ),
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    """Pydantic schema validating incoming payment gateway transaction payloads."""

    transaction_id: Optional[str] = Field(default="TX_API_001", description="Unique transaction ID.")
    transaction_time: float = Field(
        default=50_400.0,
        ge=0.0,
        description="Seconds elapsed since reference epoch (used for cyclical hour/weekend features).",
    )
    card_id: str = Field(default="CARD_0042", min_length=1, description="Tokenized card identifier.")
    merchant_id: str = Field(
        default="MERCH_0015", min_length=1, description="Tokenized merchant identifier."
    )
    merchant_category: str = Field(
        default="online_retail",
        description=f"Merchant category code. Known values: {', '.join(MERCHANT_CATEGORIES)}",
    )
    amount: float = Field(..., gt=0.0, le=250_000.0, description="Transaction amount in USD.")
    card_limit: float = Field(..., gt=0.0, le=500_000.0, description="Card credit limit in USD.")
    distance_from_home_km: Optional[float] = Field(
        default=12.5,
        ge=0.0,
        description="Geodesic distance between billing address and transaction IP/POS (km).",
    )
    is_international: int = Field(
        default=0, ge=0, le=1, description="1 if cross-border transaction, else 0."
    )
    is_card_present: int = Field(
        default=1, ge=0, le=1, description="1 if physical EMV/swipe, 0 if card-not-present."
    )
    device_trust_score: Optional[float] = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Device fingerprint reputation score in [0.0, 1.0].",
    )
    tx_count_last_1h: int = Field(
        default=0, ge=0, le=500, description="Prior card transactions in the trailing 1 hour."
    )
    tx_count_last_24h: int = Field(
        default=1, ge=0, le=2_000, description="Prior card transactions in the trailing 24 hours."
    )
    amount_sum_last_24h: float = Field(
        default=45.0, ge=0.0, description="Cumulative card spend in trailing 24 hours (USD)."
    )
    card_age_days: Optional[float] = Field(
        default=720.0, ge=0.0, description="Days since card issuance."
    )
    failed_pin_attempts_24h: int = Field(
        default=0, ge=0, le=50, description="Failed PIN/CVV attempts in the last 24 hours."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TX_LIVE_9921",
                "transaction_time": 10_200.0,
                "card_id": "CARD_0012",
                "merchant_id": "MERCH_0045",
                "merchant_category": "crypto_exchange",
                "amount": 2850.0,
                "card_limit": 3000.0,
                "distance_from_home_km": 3400.0,
                "is_international": 1,
                "is_card_present": 0,
                "device_trust_score": 0.14,
                "tx_count_last_1h": 5,
                "tx_count_last_24h": 6,
                "amount_sum_last_24h": 1200.0,
                "card_age_days": 120.0,
                "failed_pin_attempts_24h": 3,
            }
        }
    }


class ShapReason(BaseModel):
    """Individual SHAP feature attribution item."""

    feature: str
    display_name: str
    feature_value: Optional[Any]
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    """Structured fraud scoring response."""

    transaction_id: Optional[str]
    predicted_class: int
    is_fraud: bool
    fraud_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    threshold_used: float
    top_reasons: List[ShapReason]
    latency_ms: float


class HealthResponse(BaseModel):
    """Service readiness and model metadata payload."""

    status: str
    model_loaded: bool
    optimal_threshold: Optional[float] = None
    test_pr_auc: Optional[float] = None
    test_f1: Optional[float] = None


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Verify service readiness and report deployed model metrics."""
    if not MODEL_PATH.exists():
        return HealthResponse(status="degraded", model_loaded=False)

    try:
        predictor = get_predictor()
        metrics_data: Dict[str, Any] = {}
        if METRICS_PATH.exists():
            metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        s3 = metrics_data.get("stages", {}).get("stage_3_tuned_ensemble", {})
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            optimal_threshold=predictor.model.optimal_threshold,
            test_pr_auc=s3.get("pr_auc"),
            test_f1=s3.get("f1"),
        )
    except Exception as exc:
        logger.exception("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_fraud(request: TransactionRequest) -> PredictionResponse:
    """Score a single credit card transaction and return top 3 SHAP attributions."""
    t0 = time.perf_counter()
    try:
        predictor = get_predictor()
        result = predictor.predict(request.model_dump(), top_k=3)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return PredictionResponse(
        transaction_id=request.transaction_id,
        predicted_class=result["predicted_class"],
        is_fraud=result["is_fraud"],
        fraud_probability=result["fraud_probability"],
        risk_tier=result["risk_tier"],
        threshold_used=result["threshold_used"],
        top_reasons=[ShapReason(**item) for item in result["top_reasons"]],
        latency_ms=latency_ms,
    )
