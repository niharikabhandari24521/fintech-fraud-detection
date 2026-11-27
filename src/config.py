"""Central configuration for the FinTech Fraud Detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PATH: Path = DATA_DIR / "transactions.csv"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
EXPLAINER_PATH: Path = ARTIFACTS_DIR / "shap_explainer.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
README_PATH: Path = PROJECT_ROOT / "README.md"

RANDOM_SEED: int = 42

# Chronological split fractions (train -> val -> test) to prevent temporal leakage
TRAIN_FRACTION: float = 0.65
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.20

TARGET_COL: str = "is_fraud"
TIMESTAMP_COL: str = "transaction_time"
ID_COL: str = "transaction_id"

MERCHANT_CATEGORIES: List[str] = [
    "grocery",
    "dining",
    "gas_station",
    "online_retail",
    "electronics",
    "travel",
    "entertainment",
    "wire_transfer",
    "crypto_exchange",
]

RAW_NUMERIC_FEATURES: List[str] = [
    "amount",
    "card_limit",
    "distance_from_home_km",
    "device_trust_score",
    "tx_count_last_1h",
    "tx_count_last_24h",
    "amount_sum_last_24h",
    "card_age_days",
    "failed_pin_attempts_24h",
]

RAW_BINARY_FEATURES: List[str] = [
    "is_international",
    "is_card_present",
]

RAW_CATEGORICAL_FEATURES: List[str] = [
    "merchant_category",
]

ENTITY_COLS: List[str] = [
    "card_id",
    "merchant_id",
]

# Engineered numeric features produced by FeatureEngineeringTransformer
ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "amount",
    "log_amount",
    "card_limit",
    "distance_from_home_km",
    "log_distance_km",
    "device_trust_score",
    "tx_count_last_1h",
    "tx_count_last_24h",
    "amount_sum_last_24h",
    "card_age_days",
    "failed_pin_attempts_24h",
    "is_international",
    "is_card_present",
    "amount_to_limit_ratio",
    "amount_to_24h_avg_ratio",
    "velocity_burst_ratio",
    "spend_24h_to_limit_ratio",
    "card_mean_amount",
    "card_std_amount",
    "amount_zscore_vs_card",
    "amount_to_card_mean_ratio",
    "merchant_mean_amount",
    "card_tx_freq",
    "merchant_tx_freq",
    "category_tx_freq",
    "hour_sin",
    "hour_cos",
    "is_night_tx",
    "is_weekend",
    "cnp_intl_interaction",
    "low_trust_high_amt",
    "distance_missing",
    "device_trust_missing",
    "card_age_missing",
]

ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
    "merchant_category",
]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "amount": "Transaction Amount ($)",
    "log_amount": "Log Transaction Amount",
    "card_limit": "Card Credit Limit ($)",
    "distance_from_home_km": "Distance from Home (km)",
    "log_distance_km": "Log Distance from Home",
    "device_trust_score": "Device Trust Score",
    "tx_count_last_1h": "Tx Count (Last 1h)",
    "tx_count_last_24h": "Tx Count (Last 24h)",
    "amount_sum_last_24h": "Cumulative 24h Spend ($)",
    "card_age_days": "Card Tenure (Days)",
    "failed_pin_attempts_24h": "Failed PIN Attempts (24h)",
    "is_international": "International Transaction",
    "is_card_present": "Card Present (Chip/Swipe)",
    "amount_to_limit_ratio": "Amount / Card Limit Ratio",
    "amount_to_24h_avg_ratio": "Amount vs. 24h Avg Spend",
    "velocity_burst_ratio": "1h vs. 24h Velocity Burst",
    "spend_24h_to_limit_ratio": "24h Utilization Ratio",
    "card_mean_amount": "Card Historical Avg ($)",
    "card_std_amount": "Card Historical Std ($)",
    "amount_zscore_vs_card": "Amount Z-Score vs. Card History",
    "amount_to_card_mean_ratio": "Amount / Card Mean Ratio",
    "merchant_mean_amount": "Merchant Avg Ticket ($)",
    "card_tx_freq": "Card Activity Frequency",
    "merchant_tx_freq": "Merchant Popularity",
    "category_tx_freq": "Merchant Category Frequency",
    "hour_sin": "Time of Day (Sin)",
    "hour_cos": "Time of Day (Cos)",
    "is_night_tx": "Late-Night Window (00:00-05:00)",
    "is_weekend": "Weekend Indicator",
    "cnp_intl_interaction": "Card-Not-Present x International",
    "low_trust_high_amt": "Low Device Trust x High Amount",
    "distance_missing": "Missing Location Flag",
    "device_trust_missing": "Missing Device Score Flag",
    "card_age_missing": "Missing Card Age Flag",
    "merchant_category": "Merchant Category",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and optimization settings for model training."""

    random_seed: int = RANDOM_SEED
    n_optuna_trials: int = 20
    cv_folds: int = 3
    default_lgbm_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "num_leaves": 20,
            "max_depth": 5,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    default_catboost_params: Dict[str, object] = field(
        default_factory=lambda: {
            "iterations": 350,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "verbose": 0,
            "allow_writing_files": False,
        }
    )
