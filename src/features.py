"""Domain-specific feature engineering pipeline with strict zero-leakage guarantees.

All entity-level aggregations, frequency encodings, missingness indicators,
imputations, and categorical encodings are encapsulated in scikit-learn
transformers so that statistics are learned exclusively from training folds.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    RAW_BINARY_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    TIMESTAMP_COL,
)


class EntityAggregationAndFrequencyTransformer(BaseEstimator, TransformerMixin):
    """Fit entity-level priors and frequency maps on training data, then engineer features.

    Learns smoothed per-card and per-merchant statistics plus category/entity
    frequencies exclusively during ``fit()``. Unknown cards or merchants observed
    during validation, test, or live inference gracefully fall back to global priors.
    """

    def __init__(self, smoothing_weight: float = 5.0) -> None:
        self.smoothing_weight = smoothing_weight

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "EntityAggregationAndFrequencyTransformer":
        """Learn entity statistics and frequency distributions from training data."""
        df = self._ensure_dataframe(X)
        n_rows = max(len(df), 1)

        amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        self.global_mean_amount_: float = float(amounts.mean())
        self.global_std_amount_: float = float(max(amounts.std(ddof=0), 1.0))

        # Smoothed per-card historical spend stats (fitted strictly on X_train)
        card_stats = (
            df.assign(_amt=amounts)
            .groupby("card_id", observed=True)["_amt"]
            .agg(["count", "mean", "std"])
        )
        card_stats["std"] = card_stats["std"].fillna(self.global_std_amount_)
        w = self.smoothing_weight
        smoothed_card_mean = (
            card_stats["count"] * card_stats["mean"] + w * self.global_mean_amount_
        ) / (card_stats["count"] + w)
        smoothed_card_std = (
            card_stats["count"] * card_stats["std"] + w * self.global_std_amount_
        ) / (card_stats["count"] + w)

        self.card_mean_map_: Dict[str, float] = smoothed_card_mean.to_dict()
        self.card_std_map_: Dict[str, float] = smoothed_card_std.to_dict()
        self.card_freq_map_: Dict[str, float] = (card_stats["count"] / n_rows).to_dict()

        # Smoothed per-merchant stats
        merch_stats = (
            df.assign(_amt=amounts)
            .groupby("merchant_id", observed=True)["_amt"]
            .agg(["count", "mean"])
        )
        smoothed_merch_mean = (
            merch_stats["count"] * merch_stats["mean"] + w * self.global_mean_amount_
        ) / (merch_stats["count"] + w)
        self.merchant_mean_map_: Dict[str, float] = smoothed_merch_mean.to_dict()
        self.merchant_freq_map_: Dict[str, float] = (merch_stats["count"] / n_rows).to_dict()

        # Frequency encoding for merchant_category
        cat_counts = df["merchant_category"].fillna("unknown").astype(str).value_counts()
        self.category_freq_map_: Dict[str, float] = (cat_counts / n_rows).to_dict()
        self.default_freq_: float = 1.0 / (n_rows + 1.0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute domain, velocity, ratio, interaction, and cyclical features."""
        df = self._ensure_dataframe(X).copy()

        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(self.global_mean_amount_)
        card_limit = pd.to_numeric(df["card_limit"], errors="coerce").fillna(5000.0).clip(lower=1.0)
        distance = pd.to_numeric(df["distance_from_home_km"], errors="coerce")
        device_trust = pd.to_numeric(df["device_trust_score"], errors="coerce")
        card_age = pd.to_numeric(df["card_age_days"], errors="coerce")
        tx_1h = pd.to_numeric(df["tx_count_last_1h"], errors="coerce").fillna(0.0).clip(lower=0.0)
        tx_24h = pd.to_numeric(df["tx_count_last_24h"], errors="coerce").fillna(0.0).clip(lower=0.0)
        amt_24h = pd.to_numeric(df["amount_sum_last_24h"], errors="coerce").fillna(0.0).clip(lower=0.0)
        failed_pins = (
            pd.to_numeric(df["failed_pin_attempts_24h"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        is_intl = pd.to_numeric(df["is_international"], errors="coerce").fillna(0.0)
        is_cp = pd.to_numeric(df["is_card_present"], errors="coerce").fillna(1.0)

        # Missingness flags computed prior to imputation
        distance_missing = distance.isna().astype(np.float64)
        device_trust_missing = device_trust.isna().astype(np.float64)
        card_age_missing = card_age.isna().astype(np.float64)

        # Entity lookups with unseen-entity fallback
        card_ids = df["card_id"].fillna("UNKNOWN").astype(str)
        merch_ids = df["merchant_id"].fillna("UNKNOWN").astype(str)
        merch_cats = df["merchant_category"].fillna("unknown").astype(str)

        card_mean = card_ids.map(self.card_mean_map_).fillna(self.global_mean_amount_).astype(np.float64)
        card_std = card_ids.map(self.card_std_map_).fillna(self.global_std_amount_).astype(np.float64)
        merch_mean = (
            merch_ids.map(self.merchant_mean_map_).fillna(self.global_mean_amount_).astype(np.float64)
        )

        card_freq = card_ids.map(self.card_freq_map_).fillna(self.default_freq_).astype(np.float64)
        merch_freq = merch_ids.map(self.merchant_freq_map_).fillna(self.default_freq_).astype(np.float64)
        cat_freq = merch_cats.map(self.category_freq_map_).fillna(self.default_freq_).astype(np.float64)

        # Velocity, ratio, and spend-deviation features
        avg_24h = np.where(tx_24h > 0, amt_24h / np.maximum(tx_24h, 1.0), card_mean)
        amount_to_limit_ratio = amount / (card_limit + 1.0)
        amount_to_24h_avg_ratio = amount / (avg_24h + 1.0)
        velocity_burst_ratio = tx_1h / ((tx_24h / 24.0) + 0.1)
        spend_24h_to_limit_ratio = (amt_24h + amount) / (card_limit + 1.0)
        amount_zscore_vs_card = (amount - card_mean) / (card_std + 1.0)
        amount_to_card_mean_ratio = amount / (card_mean + 1.0)

        # Cyclical temporal features
        ts = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce").fillna(43_200.0)
        hour_of_day = (ts % 86_400.0) / 3_600.0
        day_of_week = ((ts // 86_400.0) % 7.0).astype(np.int32)
        hour_sin = np.sin(2.0 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour_of_day / 24.0)
        is_night_tx = ((hour_of_day >= 0.0) & (hour_of_day < 5.0)).astype(np.float64)
        is_weekend = (day_of_week >= 5).astype(np.float64)

        # Risk interaction terms
        cnp_intl_interaction = (1.0 - is_cp) * is_intl
        trust_filled = device_trust.fillna(0.8)
        log_amount = np.log1p(amount.clip(lower=0.0))
        low_trust_high_amt = (1.0 - trust_filled) * log_amount
        log_distance = np.log1p(distance.clip(lower=0.0))

        out = pd.DataFrame(
            {
                "amount": amount.astype(np.float64),
                "log_amount": log_amount.astype(np.float64),
                "card_limit": card_limit.astype(np.float64),
                "distance_from_home_km": distance.astype(np.float64),
                "log_distance_km": log_distance.astype(np.float64),
                "device_trust_score": device_trust.astype(np.float64),
                "tx_count_last_1h": tx_1h.astype(np.float64),
                "tx_count_last_24h": tx_24h.astype(np.float64),
                "amount_sum_last_24h": amt_24h.astype(np.float64),
                "card_age_days": card_age.astype(np.float64),
                "failed_pin_attempts_24h": failed_pins.astype(np.float64),
                "is_international": is_intl.astype(np.float64),
                "is_card_present": is_cp.astype(np.float64),
                "amount_to_limit_ratio": amount_to_limit_ratio.astype(np.float64),
                "amount_to_24h_avg_ratio": amount_to_24h_avg_ratio.astype(np.float64),
                "velocity_burst_ratio": velocity_burst_ratio.astype(np.float64),
                "spend_24h_to_limit_ratio": spend_24h_to_limit_ratio.astype(np.float64),
                "card_mean_amount": card_mean,
                "card_std_amount": card_std,
                "amount_zscore_vs_card": amount_zscore_vs_card.astype(np.float64),
                "amount_to_card_mean_ratio": amount_to_card_mean_ratio.astype(np.float64),
                "merchant_mean_amount": merch_mean,
                "card_tx_freq": card_freq,
                "merchant_tx_freq": merch_freq,
                "category_tx_freq": cat_freq,
                "hour_sin": hour_sin.astype(np.float64),
                "hour_cos": hour_cos.astype(np.float64),
                "is_night_tx": is_night_tx,
                "is_weekend": is_weekend,
                "cnp_intl_interaction": cnp_intl_interaction.astype(np.float64),
                "low_trust_high_amt": low_trust_high_amt.astype(np.float64),
                "distance_missing": distance_missing,
                "device_trust_missing": device_trust_missing,
                "card_age_missing": card_age_missing,
                "merchant_category": merch_cats,
            },
            index=df.index,
        )
        return out[ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES]

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame | Dict[str, object]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        return pd.DataFrame(X)


def get_engineered_feature_names() -> List[str]:
    """Return ordered list of output feature names produced by the engineered pipeline."""
    return list(ENGINEERED_NUMERIC_FEATURES) + list(ENGINEERED_CATEGORICAL_FEATURES)


def build_baseline_preprocessor() -> ColumnTransformer:
    """Build a minimal raw-feature preprocessor for the Stage 1 LogisticRegression baseline."""
    numeric_cols = list(RAW_NUMERIC_FEATURES) + list(RAW_BINARY_FEATURES)
    categorical_cols = list(RAW_CATEGORICAL_FEATURES)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """Build the full zero-leakage feature engineering + imputation/encoding pipeline."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, ENGINEERED_NUMERIC_FEATURES),
            ("cat", cat_pipe, ENGINEERED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_transformer.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("feature_engineering", EntityAggregationAndFrequencyTransformer()),
            ("preprocess", column_transformer),
        ]
    )
