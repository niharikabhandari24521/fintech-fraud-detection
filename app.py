"""Streamlit Community Cloud web application for SentinelPay Fraud Detection."""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MERCHANT_CATEGORIES, METRICS_PATH, MODEL_PATH
from src.predict import get_predictor

st.set_page_config(
    page_title="SentinelPay | Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESETS: Dict[str, Dict[str, Any]] = {
    "Normal Everyday Purchase": {
        "amount": 54.80,
        "card_limit": 5_000.0,
        "merchant_category": "grocery",
        "hour_of_day": 14,
        "is_weekend": False,
        "distance_from_home_km": 4.2,
        "is_international": False,
        "is_card_present": True,
        "device_trust_score": 0.93,
        "tx_count_last_1h": 0,
        "tx_count_last_24h": 2,
        "amount_sum_last_24h": 88.50,
        "card_age_days": 940,
        "failed_pin_attempts_24h": 0,
        "card_id": "CARD_0120",
        "merchant_id": "MERCH_0018",
    },
    "Suspicious Velocity Burst": {
        "amount": 640.00,
        "card_limit": 3_000.0,
        "merchant_category": "electronics",
        "hour_of_day": 2,
        "is_weekend": True,
        "distance_from_home_km": 185.0,
        "is_international": False,
        "is_card_present": False,
        "device_trust_score": 0.42,
        "tx_count_last_1h": 4,
        "tx_count_last_24h": 7,
        "amount_sum_last_24h": 1_350.0,
        "card_age_days": 210,
        "failed_pin_attempts_24h": 1,
        "card_id": "CARD_0042",
        "merchant_id": "MERCH_0112",
    },
    "High-Risk Account Takeover (CNP + Intl)": {
        "amount": 2_850.00,
        "card_limit": 3_000.0,
        "merchant_category": "crypto_exchange",
        "hour_of_day": 3,
        "is_weekend": False,
        "distance_from_home_km": 3_400.0,
        "is_international": True,
        "is_card_present": False,
        "device_trust_score": 0.14,
        "tx_count_last_1h": 5,
        "tx_count_last_24h": 6,
        "amount_sum_last_24h": 1_200.0,
        "card_age_days": 95,
        "failed_pin_attempts_24h": 3,
        "card_id": "CARD_0012",
        "merchant_id": "MERCH_0045",
    },
}


@st.cache_resource
def load_cached_predictor():
    """Cache loaded ensemble and TreeSHAP explainers across Streamlit reruns."""
    return get_predictor()


@st.cache_data
def load_metrics() -> Dict[str, Any]:
    """Load training progression metrics from artifacts/metrics.json."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _build_risk_gauge(probability: float, threshold: float) -> go.Figure:
    """Create a Plotly risk gauge indicator with dynamic threshold marker."""
    prob_pct = probability * 100.0
    thresh_pct = threshold * 100.0

    bar_color = "#dc2626" if probability >= threshold else "#0d9488"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            number={"suffix": "%", "valueformat": ".1f"},
            delta={
                "reference": thresh_pct,
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#0d9488"},
                "valueformat": ".1f",
                "suffix": "% vs threshold",
            },
            title={"text": "Calibrated Fraud Risk Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": bar_color, "thickness": 0.30},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, thresh_pct * 0.5], "color": "#ecfdf5"},
                    {"range": [thresh_pct * 0.5, thresh_pct], "color": "#fefce8"},
                    {"range": [thresh_pct, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.85,
                    "value": thresh_pct,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=55, b=10))
    return fig


def _build_shap_bar_chart(reasons: list[dict[str, Any]], title: str) -> go.Figure:
    """Render horizontal bar chart of top SHAP drivers for a specific prediction."""
    df_shap = pd.DataFrame(reasons).iloc[::-1].reset_index(drop=True)
    colors = [
        "#dc2626" if val >= 0 else "#0d9488" for val in df_shap["shap_value"]
    ]
    labels = [
        f"{row['display_name']} ({row['feature_value']})"
        for _, row in df_shap.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            x=df_shap["shap_value"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in df_shap["shap_value"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Log-Odds Contribution (+ increases fraud risk, - decreases)",
        yaxis_title="",
        height=310,
        margin=dict(l=10, r=40, t=45, b=40),
    )
    return fig


def main() -> None:
    """Render the SentinelPay Streamlit application."""
    st.title("🛡️ SentinelPay — Real-Time Credit Card Fraud Detection")
    st.caption(
        "Optuna-Tuned LightGBM + CatBoost Calibrated Ensemble | Zero-Leakage Feature Pipeline | Exact TreeSHAP Explainability"
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model artifact `artifacts/model.joblib` was not found. "
            "Run `python data/fetch_data.py` and `python src/train.py` first."
        )
        st.stop()

    predictor = load_cached_predictor()
    metrics_data = load_metrics()

    # Sidebar: Preset Scenario Selector & Transaction Controls
    st.sidebar.header("Transaction Simulator")
    preset_name = st.sidebar.selectbox(
        "Load Scenario Preset",
        options=list(PRESETS.keys()),
        index=0,
    )
    preset = PRESETS[preset_name]

    with st.sidebar.form("transaction_inputs"):
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0,
            max_value=50_000.0,
            value=float(preset["amount"]),
            step=10.0,
        )
        card_limit = st.number_input(
            "Card Credit Limit ($)",
            min_value=500.0,
            max_value=100_000.0,
            value=float(preset["card_limit"]),
            step=500.0,
        )
        merchant_category = st.selectbox(
            "Merchant Category",
            options=MERCHANT_CATEGORIES,
            index=MERCHANT_CATEGORIES.index(preset["merchant_category"]),
        )
        device_trust_score = st.slider(
            "Device Trust Score",
            min_value=0.01,
            max_value=0.99,
            value=float(preset["device_trust_score"]),
            step=0.01,
        )
        distance_from_home_km = st.slider(
            "Distance from Home (km)",
            min_value=0.0,
            max_value=5_000.0,
            value=float(preset["distance_from_home_km"]),
            step=5.0,
        )
        tx_count_last_1h = st.slider(
            "Transactions in Last 1h",
            min_value=0,
            max_value=15,
            value=int(preset["tx_count_last_1h"]),
        )
        tx_count_last_24h = st.slider(
            "Transactions in Last 24h",
            min_value=0,
            max_value=40,
            value=max(int(preset["tx_count_last_24h"]), int(preset["tx_count_last_1h"])),
        )
        amount_sum_last_24h = st.number_input(
            "Cumulative 24h Spend ($)",
            min_value=0.0,
            max_value=100_000.0,
            value=float(preset["amount_sum_last_24h"]),
            step=50.0,
        )
        failed_pin_attempts_24h = st.slider(
            "Failed PIN/CVV Attempts (24h)",
            min_value=0,
            max_value=8,
            value=int(preset["failed_pin_attempts_24h"]),
        )
        hour_of_day = st.slider(
            "Hour of Day (0-23)",
            min_value=0,
            max_value=23,
            value=int(preset["hour_of_day"]),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            is_international = st.checkbox(
                "International", value=bool(preset["is_international"])
            )
            is_weekend = st.checkbox("Weekend", value=bool(preset["is_weekend"]))
        with col_b:
            is_card_present = st.checkbox(
                "Card Present", value=bool(preset["is_card_present"])
            )
        card_age_days = st.slider(
            "Card Age (Days)",
            min_value=1,
            max_value=2_500,
            value=int(preset["card_age_days"]),
        )
        st.form_submit_button("Score Transaction", width="stretch")

    # Convert UI inputs to schema timestamp (days 0..4 weekday, days 5..6 weekend)
    day_offset = 5 if is_weekend else 2
    transaction_time = float(day_offset * 86_400 + hour_of_day * 3_600)

    payload = {
        "transaction_id": "TX_STREAMLIT",
        "transaction_time": transaction_time,
        "card_id": preset["card_id"],
        "merchant_id": preset["merchant_id"],
        "merchant_category": merchant_category,
        "amount": float(amount),
        "card_limit": float(card_limit),
        "distance_from_home_km": float(distance_from_home_km),
        "is_international": int(is_international),
        "is_card_present": int(is_card_present),
        "device_trust_score": float(device_trust_score),
        "tx_count_last_1h": int(tx_count_last_1h),
        "tx_count_last_24h": int(max(tx_count_last_24h, tx_count_last_1h)),
        "amount_sum_last_24h": float(amount_sum_last_24h),
        "card_age_days": float(card_age_days),
        "failed_pin_attempts_24h": int(failed_pin_attempts_24h),
    }

    tab_scoring, tab_perf = st.tabs(
        ["🔍 Live Transaction Scoring", "📊 Model Performance & Iteration"]
    )

    with tab_scoring:
        result = predictor.predict(payload, top_k=3)
        prob = result["fraud_probability"]
        thresh = result["threshold_used"]
        is_fraud = result["is_fraud"]
        tier = result["risk_tier"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision", "🚨 FLAG / BLOCK" if is_fraud else "✅ APPROVE")
        c2.metric("Fraud Probability", f"{prob * 100:.2f}%")
        c3.metric("Decision Threshold", f"{thresh * 100:.2f}%")
        c4.metric("Risk Tier", tier)

        col_left, col_right = st.columns([1.0, 1.25])
        with col_left:
            st.plotly_chart(
                _build_risk_gauge(prob, thresh),
                width="stretch",
            )
            snap = result["engineered_snapshot"]
            st.markdown("**Engineered Risk Signals (Computed Zero-Leakage):**")
            s1, s2 = st.columns(2)
            s1.metric("Amount / Limit Ratio", f"{snap['amount_to_limit_ratio']:.2f}x")
            s1.metric("Velocity Burst Ratio", f"{snap['velocity_burst_ratio']:.2f}x")
            s2.metric("Spend Z-Score vs Card", f"{snap['amount_zscore_vs_card']:+.2f}σ")
            s2.metric(
                "CNP × International",
                "Yes (High Risk)" if snap["cnp_intl_interaction"] == 1 else "No",
            )

        with col_right:
            st.plotly_chart(
                _build_shap_bar_chart(
                    result["top_reasons"],
                    title="Top 3 SHAP Drivers Behind This Prediction",
                ),
                width="stretch",
            )
            reasons_df = pd.DataFrame(result["top_reasons"])[
                ["display_name", "feature_value", "shap_value", "direction"]
            ]
            reasons_df.columns = ["Feature", "Input Value", "SHAP Impact", "Effect"]
            st.dataframe(reasons_df, hide_index=True, width="stretch")

        with st.expander("View Top 10 Feature Contributions (Full Local Breakdown)"):
            st.plotly_chart(
                _build_shap_bar_chart(
                    result["all_top_10_reasons"],
                    title="Top 10 Local TreeSHAP Attributions",
                ),
                width="stretch",
            )

    with tab_perf:
        if not metrics_data:
            st.warning("No `artifacts/metrics.json` found. Run `python src/train.py` first.")
        else:
            stages = metrics_data["stages"]
            opt = metrics_data["optimization"]
            ds = metrics_data["dataset"]

            s1 = stages["stage_1_baseline_logreg"]
            s2 = stages["stage_2_fe_lgbm"]
            s3 = stages["stage_3_tuned_ensemble"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Final PR-AUC",
                f"{s3['pr_auc']:.4f}",
                delta=f"+{opt['pr_auc_lift_vs_baseline_pct']:.1f}% vs baseline",
            )
            k2.metric(
                "Final F1-Score",
                f"{s3['f1']:.4f}",
                delta=f"+{opt['f1_lift_vs_baseline_pct']:.1f}% vs baseline",
            )
            k3.metric("Final Recall", f"{s3['recall']:.4f}")
            k4.metric(
                "Test Fraud Prevalence",
                f"{ds['overall_fraud_rate'] * 100:.2f}%",
                help=f"Chronological holdout test set: {ds['test_rows']:,} txs ({ds['test_fraud_count']} frauds)",
            )

            prog_df = pd.DataFrame(
                [
                    {
                        "Stage": "1. Baseline (LogReg)",
                        "PR-AUC": s1["pr_auc"],
                        "F1-Score": s1["f1"],
                        "Recall": s1["recall"],
                        "Precision": s1["precision"],
                        "ROC-AUC": s1["roc_auc"],
                    },
                    {
                        "Stage": "2. FE + LightGBM",
                        "PR-AUC": s2["pr_auc"],
                        "F1-Score": s2["f1"],
                        "Recall": s2["recall"],
                        "Precision": s2["precision"],
                        "ROC-AUC": s2["roc_auc"],
                    },
                    {
                        "Stage": "3. Tuned LGBM+CatBoost Ensemble",
                        "PR-AUC": s3["pr_auc"],
                        "F1-Score": s3["f1"],
                        "Recall": s3["recall"],
                        "Precision": s3["precision"],
                        "ROC-AUC": s3["roc_auc"],
                    },
                ]
            )
            st.subheader("Model Iteration Progression (Chronological Hold-Out Test Set)")
            st.dataframe(prog_df, hide_index=True, width="stretch")

            melted = prog_df.melt(
                id_vars="Stage",
                value_vars=["PR-AUC", "F1-Score", "Recall", "Precision"],
                var_name="Metric",
                value_name="Score",
            )
            fig_prog = px.bar(
                melted,
                x="Metric",
                y="Score",
                color="Stage",
                barmode="group",
                text_auto=".3f",
                title="Progression Across Imbalanced Evaluation Metrics",
            )
            fig_prog.update_layout(yaxis_range=[0, 1.05], height=380)
            st.plotly_chart(fig_prog, width="stretch")

            col_pr, col_cm = st.columns(2)
            with col_pr:
                fig_pr = go.Figure()
                for label, st_dict, color in [
                    ("1. Baseline LogReg", s1, "#94a3b8"),
                    ("2. FE + LightGBM", s2, "#2563eb"),
                    ("3. Tuned Ensemble", s3, "#dc2626"),
                ]:
                    fig_pr.add_trace(
                        go.Scatter(
                            x=st_dict["pr_curve"]["recall"],
                            y=st_dict["pr_curve"]["precision"],
                            mode="lines",
                            name=f"{label} (PR-AUC={st_dict['pr_auc']:.3f})",
                            line=dict(width=2.5, color=color),
                        )
                    )
                fig_pr.update_layout(
                    title="Precision-Recall Curve Progression",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=370,
                )
                st.plotly_chart(fig_pr, width="stretch")

            with col_cm:
                cm = s3["confusion_matrix"]
                cm_z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
                fig_cm = px.imshow(
                    cm_z,
                    text_auto=True,
                    x=["Predicted Legit (0)", "Predicted Fraud (1)"],
                    y=["Actual Legit (0)", "Actual Fraud (1)"],
                    color_continuous_scale="Blues",
                    title=f"Final Ensemble Confusion Matrix (Threshold={s3['threshold']:.3f})",
                )
                fig_cm.update_layout(height=370)
                st.plotly_chart(fig_cm, width="stretch")

            top_global = metrics_data.get("top_global_shap_features", [])
            if top_global:
                gdf = pd.DataFrame(top_global).iloc[::-1]
                fig_g = px.bar(
                    gdf,
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Global Top 15 Features by Mean |SHAP Value|",
                    labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
                )
                fig_g.update_layout(height=420)
                st.plotly_chart(fig_g, width="stretch")


if __name__ == "__main__":
    main()
