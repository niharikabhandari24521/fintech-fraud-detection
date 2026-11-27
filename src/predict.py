"""Standalone inference module for real-time fraud scoring and SHAP explanations."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPLAINER_PATH,
    FEATURE_DISPLAY_NAMES,
    MODEL_PATH,
)
from src.models import FraudDetectionEnsemble, ShapExplainerBundle

logger = logging.getLogger("predict")


def _extract_positive_shap(shap_values: Any) -> np.ndarray:
    """Normalize TreeExplainer output across SHAP/LightGBM/CatBoost versions to 1D/2D array."""
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1], dtype=np.float64)
    else:
        arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr


def _assign_risk_tier(probability: float, threshold: float) -> str:
    """Map calibrated fraud probability to an operational risk tier."""
    if probability >= max(0.75, threshold + 0.25):
        return "CRITICAL"
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * 0.50:
        return "MEDIUM"
    return "LOW"


class FraudPredictor:
    """Loads serialized model & SHAP explainers to score transactions with local attributions."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        explainer_path: Path = EXPLAINER_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run `python src/train.py` first."
            )
        if not explainer_path.exists():
            raise FileNotFoundError(
                f"SHAP artifact not found at {explainer_path}. Run `python src/train.py` first."
            )

        self.model: FraudDetectionEnsemble = joblib.load(model_path)
        self.shap_bundle: ShapExplainerBundle = joblib.load(explainer_path)
        logger.info(
            "Loaded model & SHAP explainer (threshold=%.4f, lgbm_weight=%.2f)",
            self.model.optimal_threshold,
            self.model.lgbm_weight,
        )

    @staticmethod
    def _to_dataframe(payload: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Normalize a single transaction dictionary or DataFrame into a 1-row/N-row DataFrame."""
        if isinstance(payload, pd.DataFrame):
            df = payload.copy()
        elif isinstance(payload, dict):
            df = pd.DataFrame([payload])
        else:
            raise TypeError(f"Unsupported input payload type: {type(payload)}")

        defaults: Dict[str, Any] = {
            "transaction_id": "TX_LIVE",
            "transaction_time": 43_200.0,
            "card_id": "CARD_UNKNOWN",
            "merchant_id": "MERCH_UNKNOWN",
            "merchant_category": "online_retail",
            "amount": 100.0,
            "card_limit": 5_000.0,
            "distance_from_home_km": np.nan,
            "is_international": 0,
            "is_card_present": 1,
            "device_trust_score": np.nan,
            "tx_count_last_1h": 0,
            "tx_count_last_24h": 1,
            "amount_sum_last_24h": 0.0,
            "card_age_days": np.nan,
            "failed_pin_attempts_24h": 0,
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        return df

    def predict(
        self,
        payload: Dict[str, Any] | pd.DataFrame,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Predict fraud class, calibrated probability, and top-K SHAP reasons for one transaction."""
        raw_df = self._to_dataframe(payload)
        if len(raw_df) != 1:
            raise ValueError(
                f"FraudPredictor.predict expects a single transaction row; got {len(raw_df)}."
            )

        X_transformed = self.model.transform_features(raw_df)
        prob = float(self.model.predict_proba_from_transformed(X_transformed)[0])
        threshold = float(self.model.optimal_threshold)
        pred_class = int(prob >= threshold)
        risk_tier = _assign_risk_tier(prob, threshold)

        # Compute exact blended TreeSHAP attributions
        lgb_shap = _extract_positive_shap(self.shap_bundle.lgbm_explainer.shap_values(X_transformed))
        cat_shap = _extract_positive_shap(
            self.shap_bundle.catboost_explainer.shap_values(X_transformed)
        )
        w = self.shap_bundle.lgbm_weight
        blended_shap = (w * lgb_shap[0]) + ((1.0 - w) * cat_shap[0])

        feature_names = self.shap_bundle.feature_names
        row_values = X_transformed.iloc[0]

        contributions: List[Dict[str, Any]] = []
        for idx, feat_name in enumerate(feature_names):
            shap_val = float(blended_shap[idx])
            raw_feat_val = (
                raw_df.iloc[0][feat_name]
                if feat_name in raw_df.columns
                else row_values[feat_name]
            )
            if pd.isna(raw_feat_val):
                display_val: Any = None
            elif isinstance(raw_feat_val, (np.floating, float)):
                display_val = round(float(raw_feat_val), 4)
            elif isinstance(raw_feat_val, (np.integer, int)):
                display_val = int(raw_feat_val)
            else:
                display_val = str(raw_feat_val)

            contributions.append(
                {
                    "feature": feat_name,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                    "feature_value": display_val,
                    "shap_value": round(shap_val, 4),
                    "abs_shap": round(abs(shap_val), 4),
                    "direction": "increases_risk" if shap_val >= 0.0 else "decreases_risk",
                }
            )

        contributions_sorted = sorted(
            contributions, key=lambda item: item["abs_shap"], reverse=True
        )
        top_reasons = [
            {k: v for k, v in item.items() if k != "abs_shap"}
            for item in contributions_sorted[:top_k]
        ]

        return {
            "predicted_class": pred_class,
            "is_fraud": bool(pred_class),
            "fraud_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "threshold_used": round(threshold, 4),
            "top_reasons": top_reasons,
            "all_top_10_reasons": [
                {k: v for k, v in item.items() if k != "abs_shap"}
                for item in contributions_sorted[:10]
            ],
            "engineered_snapshot": {
                "amount_to_limit_ratio": round(float(row_values["amount_to_limit_ratio"]), 4),
                "velocity_burst_ratio": round(float(row_values["velocity_burst_ratio"]), 4),
                "amount_zscore_vs_card": round(float(row_values["amount_zscore_vs_card"]), 4),
                "cnp_intl_interaction": int(row_values["cnp_intl_interaction"]),
            },
        }


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """Return a cached singleton FraudPredictor instance."""
    return FraudPredictor()


def predict_transaction(payload: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    """Convenience function for scoring a single transaction dict."""
    predictor = get_predictor()
    return predictor.predict(payload, top_k=top_k)


if __name__ == "__main__":
    sample_high_risk = {
        "transaction_time": 9_600.0,
        "card_id": "CARD_0012",
        "merchant_id": "MERCH_0045",
        "merchant_category": "crypto_exchange",
        "amount": 2_850.0,
        "card_limit": 3_000.0,
        "distance_from_home_km": 3_400.0,
        "is_international": 1,
        "is_card_present": 0,
        "device_trust_score": 0.14,
        "tx_count_last_1h": 5,
        "tx_count_last_24h": 6,
        "amount_sum_last_24h": 1_200.0,
        "card_age_days": 120.0,
        "failed_pin_attempts_24h": 3,
    }
    import json

    print(json.dumps(predict_transaction(sample_high_risk), indent=2))
