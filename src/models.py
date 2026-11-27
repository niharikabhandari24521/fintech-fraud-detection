"""Serializable model wrapper classes for production inference and SHAP explainability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class PlattCalibratedModel:
    """Wraps a fitted binary classifier with an out-of-fold Platt scaling calibrator."""

    estimator: Any
    calibrator: LogisticRegression

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities of shape ``(n_samples, 2)``."""
        raw_prob = np.clip(self.estimator.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
        logits = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)


class FraudDetectionEnsemble:
    """Production inference artifact bundling feature pipeline, calibrated ensemble, and threshold."""

    def __init__(
        self,
        feature_pipeline: Pipeline,
        lgbm_model: PlattCalibratedModel,
        catboost_model: PlattCalibratedModel,
        lgbm_weight: float,
        optimal_threshold: float,
        feature_names: List[str],
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.lgbm_model = lgbm_model
        self.catboost_model = catboost_model
        self.lgbm_weight = float(lgbm_weight)
        self.optimal_threshold = float(optimal_threshold)
        self.feature_names = list(feature_names)

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transaction DataFrame into engineered model features."""
        return self.feature_pipeline.transform(X)

    def predict_proba_from_transformed(self, X_transformed: pd.DataFrame) -> np.ndarray:
        """Return calibrated fraud probabilities for already-transformed features."""
        p_lgb = self.lgbm_model.predict_proba(X_transformed)[:, 1]
        p_cat = self.catboost_model.predict_proba(X_transformed)[:, 1]
        return self.lgbm_weight * p_lgb + (1.0 - self.lgbm_weight) * p_cat

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated fraud probabilities directly from raw transactions."""
        X_transformed = self.transform_features(X)
        return self.predict_proba_from_transformed(X_transformed)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return binary fraud predictions using ``optimal_threshold`` (or override)."""
        thresh = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(np.int32)


@dataclass
class ShapExplainerBundle:
    """Dual TreeExplainer bundle for fast exact ensemble SHAP attribution."""

    lgbm_explainer: shap.TreeExplainer
    catboost_explainer: shap.TreeExplainer
    lgbm_weight: float
    feature_names: List[str]
