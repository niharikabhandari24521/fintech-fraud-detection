"""Synthetic realistic FinTech credit card transaction generator.

Generates a chronologically ordered imbalanced transaction dataset (~1.5% fraud)
with realistic cardholder baselines, merchant profiles, velocity bursts,
device/geospatial anomalies, and realistic missingness patterns.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, MERCHANT_CATEGORIES, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fetch_data")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _compute_rolling_velocity(
    timestamps: np.ndarray,
    card_indices: np.ndarray,
    amounts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly causal 1h/24h transaction counts and 24h cumulative spend per card.

    Assumes inputs are sorted ascending by `timestamps`.
    """
    n = len(timestamps)
    tx_1h = np.zeros(n, dtype=np.int32)
    tx_24h = np.zeros(n, dtype=np.int32)
    amt_24h = np.zeros(n, dtype=np.float64)

    # Group row indices by card for fast O(N) two-pointer causal sliding windows
    order = np.argsort(card_indices, kind="mergesort")
    sorted_cards = card_indices[order]
    split_points = np.flatnonzero(np.diff(sorted_cards)) + 1
    groups = np.split(order, split_points)

    for grp in groups:
        left_1h = 0
        left_24h = 0
        running_amt_24h = 0.0
        for pos, idx in enumerate(grp):
            t_now = timestamps[idx]
            while left_24h < pos and (t_now - timestamps[grp[left_24h]]) > 86400.0:
                running_amt_24h -= amounts[grp[left_24h]]
                left_24h += 1
            while left_1h < pos and (t_now - timestamps[grp[left_1h]]) > 3600.0:
                left_1h += 1

            tx_1h[idx] = pos - left_1h
            tx_24h[idx] = pos - left_24h
            amt_24h[idx] = max(0.0, running_amt_24h)
            running_amt_24h += amounts[idx]

    return tx_1h, tx_24h, amt_24h


def generate_transactions(
    n_rows: int = 50_000,
    target_fraud_rate: float = 0.015,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a realistic, chronologically ordered transaction dataset.

    Args:
        n_rows: Total number of transactions to simulate.
        target_fraud_rate: Desired prevalence of the positive fraud class (~1.5%).
        seed: Random seed for reproducibility.

    Returns:
        Chronologically sorted DataFrame matching the production transaction schema.
    """
    rng = np.random.default_rng(seed)

    n_cards = 1_500
    n_merchants = 400
    horizon_seconds = 30 * 86_400  # 30 days of continuous transaction traffic

    # Cardholder latent profiles
    card_limits = rng.choice(
        [1_500.0, 3_000.0, 5_000.0, 10_000.0, 20_000.0],
        size=n_cards,
        p=[0.20, 0.30, 0.25, 0.15, 0.10],
    )
    card_base_spend = rng.lognormal(mean=3.6, sigma=0.55, size=n_cards)
    card_ages = rng.uniform(30.0, 2_500.0, size=n_cards)

    # Merchant latent profiles
    cat_probs = [0.22, 0.18, 0.12, 0.18, 0.10, 0.07, 0.07, 0.03, 0.03]
    merchant_cats = rng.choice(MERCHANT_CATEGORIES, size=n_merchants, p=cat_probs)
    cat_risk_map = {
        "grocery": -0.7,
        "dining": -0.5,
        "gas_station": -0.4,
        "entertainment": -0.1,
        "online_retail": 0.45,
        "electronics": 0.65,
        "travel": 0.55,
        "wire_transfer": 1.15,
        "crypto_exchange": 1.35,
    }
    cat_amount_mult = {
        "grocery": 0.75,
        "dining": 0.65,
        "gas_station": 0.60,
        "entertainment": 0.85,
        "online_retail": 1.15,
        "electronics": 2.40,
        "travel": 2.80,
        "wire_transfer": 3.20,
        "crypto_exchange": 2.90,
    }
    merchant_base_risk = np.array([cat_risk_map[c] for c in merchant_cats]) + rng.normal(
        0.0, 0.25, size=n_merchants
    )
    merchant_amt_mult = np.array([cat_amount_mult[c] for c in merchant_cats])

    # Generate mostly legitimate baseline traffic + realistic fraud bursts
    n_burst_events = int(n_rows * 0.012)
    n_regular = n_rows - n_burst_events

    reg_timestamps = rng.uniform(0.0, horizon_seconds, size=n_regular)
    # Daytime peak weighting (8am-10pm more common)
    hour_of_day = (reg_timestamps % 86_400) / 3_600.0
    night_mask = (hour_of_day >= 0.0) & (hour_of_day < 5.0)
    # Shift half of random night timestamps to daytime for realistic circadian traffic
    shift_mask = night_mask & (rng.random(n_regular) < 0.65)
    reg_timestamps[shift_mask] += rng.uniform(7.0, 14.0, size=shift_mask.sum()) * 3_600.0
    reg_timestamps = np.clip(reg_timestamps, 0.0, horizon_seconds - 1.0)

    reg_cards = rng.integers(0, n_cards, size=n_regular)
    reg_merchants = rng.integers(0, n_merchants, size=n_regular)

    # Inject realistic card-testing / velocity burst clusters (3-5 rapid txs within minutes)
    burst_cards = []
    burst_merchants = []
    burst_timestamps = []
    burst_flag = []
    remaining_burst = n_burst_events
    while remaining_burst > 0:
        cluster_size = min(remaining_burst, int(rng.integers(3, 6)))
        c_idx = int(rng.integers(0, n_cards))
        t_start = float(rng.uniform(3_600.0, horizon_seconds - 1_800.0))
        high_risk_merchants = np.where(
            np.isin(
                merchant_cats,
                ["online_retail", "electronics", "crypto_exchange", "wire_transfer"],
            )
        )[0]
        m_pool = high_risk_merchants if len(high_risk_merchants) > 0 else np.arange(n_merchants)
        for j in range(cluster_size):
            burst_cards.append(c_idx)
            burst_merchants.append(int(rng.choice(m_pool)))
            burst_timestamps.append(t_start + j * float(rng.uniform(35.0, 240.0)))
            burst_flag.append(1)
        remaining_burst -= cluster_size

    timestamps = np.concatenate([reg_timestamps, np.array(burst_timestamps)])
    card_indices = np.concatenate([reg_cards, np.array(burst_cards, dtype=np.int64)])
    merchant_indices = np.concatenate([reg_merchants, np.array(burst_merchants, dtype=np.int64)])
    is_burst_cluster = np.concatenate(
        [np.zeros(n_regular, dtype=np.int32), np.array(burst_flag, dtype=np.int32)]
    )

    # Chronological sort
    sort_idx = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[sort_idx]
    card_indices = card_indices[sort_idx]
    merchant_indices = merchant_indices[sort_idx]
    is_burst_cluster = is_burst_cluster[sort_idx]

    # Sample transaction attributes
    m_cat = merchant_cats[merchant_indices]
    c_limit = card_limits[card_indices]
    c_base = card_base_spend[card_indices]
    c_age = card_ages[card_indices]

    # Card-present & international probabilities depend on merchant category
    cnp_cats = np.isin(m_cat, ["online_retail", "travel", "wire_transfer", "crypto_exchange"])
    is_card_present = np.where(
        cnp_cats,
        rng.binomial(1, 0.12, size=n_rows),
        rng.binomial(1, 0.85, size=n_rows),
    )
    # Fraud burst clusters are predominantly card-not-present
    is_card_present = np.where(
        is_burst_cluster == 1,
        rng.binomial(1, 0.10, size=n_rows),
        is_card_present,
    )

    is_international = np.where(
        is_burst_cluster == 1,
        rng.binomial(1, 0.65, size=n_rows),
        rng.binomial(1, 0.08, size=n_rows),
    )

    # Transaction amounts tied to card baseline + merchant multiplier, capped reasonably near limit
    raw_amt = (
        c_base
        * merchant_amt_mult[merchant_indices]
        * rng.lognormal(mean=0.0, sigma=0.55, size=n_rows)
    )
    # Burst attacks often escalate or probe with larger tickets
    raw_amt = np.where(
        is_burst_cluster == 1,
        raw_amt * rng.uniform(1.8, 4.2, size=n_rows),
        raw_amt,
    )
    amount = np.round(np.clip(raw_amt, 1.50, c_limit * 1.15), 2)

    # Distance from home (higher for travel & international)
    dist_scale = np.where(is_international == 1, 1_800.0, 14.0)
    dist_scale = np.where(m_cat == "travel", dist_scale * 2.0, dist_scale)
    distance_km = np.round(rng.exponential(scale=dist_scale), 2)

    # Device trust score: normally high (0.75-0.98), degrades in suspicious contexts
    device_trust = np.clip(
        rng.beta(a=8.0, b=1.8, size=n_rows)
        - 0.28 * is_burst_cluster
        - 0.08 * is_international,
        0.02,
        0.99,
    )
    device_trust = np.round(device_trust, 4)

    # Failed PIN attempts in last 24h
    failed_pins = np.where(
        is_burst_cluster == 1,
        rng.poisson(lam=1.6, size=n_rows),
        rng.poisson(lam=0.06, size=n_rows),
    )
    failed_pins = np.clip(failed_pins, 0, 8)

    # Causal rolling velocity features
    tx_1h, tx_24h, amt_24h = _compute_rolling_velocity(timestamps, card_indices, amount)

    # Latent fraud log-odds combining realistic non-linear FinTech risk regimes
    hours = (timestamps % 86_400) / 3_600.0
    is_night = ((hours >= 0.0) & (hours < 5.0)).astype(float)
    amt_ratio_card = amount / (c_base + 1.0)
    amt_ratio_limit = amount / (c_limit + 1.0)
    cnp_intl = (1.0 - is_card_present) * is_international
    vel_burst = tx_1h / ((tx_24h / 24.0) + 0.1)

    # Realistic non-linear fraud rings (ATO, card-testing bursts, geo-jumps, night digital)
    # Note: signals intentionally rely on multi-feature interactions and card-relative
    # deviations so linear models on raw features struggle compared to FE + GBDTs.
    ato_regime = (
        (amt_ratio_card > 2.8) & (device_trust < 0.55) & (is_card_present == 0)
    ).astype(float)
    card_testing_regime = (
        (tx_1h >= 2) & (vel_burst > 2.5) & (is_card_present == 0)
    ).astype(float)
    geo_jump_regime = ((distance_km > 350.0) & (tx_24h >= 2) & (is_card_present == 0)).astype(
        float
    )
    pin_util_regime = ((amt_ratio_limit > 0.35) & (failed_pins >= 1)).astype(float)
    night_digital_regime = (
        (is_night == 1.0)
        & (is_card_present == 0)
        & np.isin(m_cat, ["crypto_exchange", "wire_transfer", "electronics", "online_retail"])
    ).astype(float)
    new_card_high_spend = ((c_age < 180.0) & (amt_ratio_limit > 0.45)).astype(float)
    trusted_repeat_discount = (
        (device_trust > 0.80) & (is_card_present == 1) & (tx_1h == 0)
    ).astype(float)

    risk_score = (
        2.60 * ato_regime
        + 2.40 * card_testing_regime
        + 1.95 * geo_jump_regime
        + 2.15 * pin_util_regime
        + 1.75 * night_digital_regime
        + 1.50 * new_card_high_spend
        + 0.95 * cnp_intl
        + 0.40 * np.log1p(amt_ratio_card)
        - 1.15 * device_trust
        - 1.40 * trusted_repeat_discount
        + 0.35 * merchant_base_risk[merchant_indices]
        + 1.20 * is_burst_cluster
        + rng.normal(0.0, 0.30, size=n_rows)
    )

    # Calibrate intercept via binary search so realized fraud rate matches ~target_fraud_rate
    low, high = -15.0, 5.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        mean_p = float(_sigmoid(risk_score + mid).mean())
        if mean_p < target_fraud_rate:
            low = mid
        else:
            high = mid
    intercept = 0.5 * (low + high)
    fraud_prob = _sigmoid(risk_score + intercept)
    is_fraud = rng.binomial(1, fraud_prob).astype(np.int32)

    # Inject realistic MCAR/MAR missingness in selected raw fields
    dist_series = distance_km.astype(float)
    dist_series[rng.random(n_rows) < 0.03] = np.nan

    trust_series = device_trust.astype(float)
    trust_series[rng.random(n_rows) < 0.025] = np.nan

    age_series = np.round(c_age, 1).astype(float)
    age_series[rng.random(n_rows) < 0.02] = np.nan

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX_{i:07d}" for i in range(n_rows)],
            "transaction_time": np.round(timestamps, 2),
            "card_id": [f"CARD_{c:04d}" for c in card_indices],
            "merchant_id": [f"MERCH_{m:04d}" for m in merchant_indices],
            "merchant_category": m_cat,
            "amount": amount,
            "card_limit": c_limit,
            "distance_from_home_km": dist_series,
            "is_international": is_international.astype(np.int32),
            "is_card_present": is_card_present.astype(np.int32),
            "device_trust_score": trust_series,
            "tx_count_last_1h": tx_1h,
            "tx_count_last_24h": tx_24h,
            "amount_sum_last_24h": np.round(amt_24h, 2),
            "card_age_days": age_series,
            "failed_pin_attempts_24h": failed_pins.astype(np.int32),
            "is_fraud": is_fraud,
        }
    )

    return df


def main() -> None:
    """CLI entrypoint to generate and persist the dataset."""
    parser = argparse.ArgumentParser(
        description="Generate realistic imbalanced FinTech credit card fraud dataset."
    )
    parser.add_argument("--n-rows", type=int, default=50_000, help="Number of transactions.")
    parser.add_argument(
        "--fraud-rate", type=float, default=0.015, help="Target positive fraud rate."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="Output CSV path.")
    args = parser.parse_args()

    logger.info(
        "Generating %d synthetic transactions (target fraud rate=%.2f%%, seed=%d)...",
        args.n_rows,
        args.fraud_rate * 100.0,
        args.seed,
    )
    df = generate_transactions(
        n_rows=args.n_rows,
        target_fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    fraud_count = int(df["is_fraud"].sum())
    fraud_pct = 100.0 * fraud_count / len(df)
    logger.info(
        "Saved %d rows to %s | Fraud cases: %d (%.3f%%)",
        len(df),
        args.output,
        fraud_count,
        fraud_pct,
    )


if __name__ == "__main__":
    main()
