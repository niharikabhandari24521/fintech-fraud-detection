"""End-to-end training progression for FinTech Credit Card Fraud Detection.

Strict 3-stage progression evaluated on a chronological hold-out test split:
1. Baseline LogisticRegression on minimal imputed/scaled raw features.
2. Feature-Engineered LightGBM (default hyperparameters) on domain/velocity/entity features.
3. Optuna-tuned LightGBM + CatBoost Platt-calibrated ensemble with validation
   threshold optimization for F1 / PR-AUC.

Outputs:
- ``artifacts/model.joblib``: Serialized preprocessor + calibrated ensemble + threshold.
- ``artifacts/shap_explainer.joblib``: Dual TreeExplainer bundle for instant local SHAP.
- ``artifacts/metrics.json``: Comprehensive progression metrics, curves, & confusion matrices.
- Auto-updates the comparison table inside ``README.md`` if marker tags are present.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    EXPLAINER_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    README_PATH,
    TARGET_COL,
    TEST_FRACTION,
    TIMESTAMP_COL,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TrainingConfig,
)
from src.features import (
    build_baseline_preprocessor,
    build_feature_pipeline,
    get_engineered_feature_names,
)
from src.models import (
    FraudDetectionEnsemble,
    PlattCalibratedModel,
    ShapExplainerBundle,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transaction data chronologically into train, validation, and test sets."""
    df_sorted = df.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-6

    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute imbalanced classification metrics, confusion matrix, and sampled curve points."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)

    # Downsample curves to <= 80 points for compact metrics.json storage
    pr_idx = np.linspace(0, len(prec_curve) - 1, min(80, len(prec_curve)), dtype=int)
    roc_idx = np.linspace(0, len(fpr_curve) - 1, min(80, len(fpr_curve)), dtype=int)

    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "brier_score": round(brier, 5),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "pr_curve": {
            "recall": [round(float(x), 4) for x in rec_curve[pr_idx]],
            "precision": [round(float(x), 4) for x in prec_curve[pr_idx]],
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_curve[roc_idx]],
            "tpr": [round(float(x), 4) for x in tpr_curve[roc_idx]],
        },
    }


def _fit_platt_calibrated_model(
    model_factory: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 3,
    seed: int = RANDOM_SEED,
) -> PlattCalibratedModel:
    """Fit out-of-fold Platt scaling calibrator plus full-train base estimator."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X_train), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_train, y_train):
        fold_model = model_factory()
        fold_model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = np.clip(
            fold_model.predict_proba(X_train.iloc[va_idx])[:, 1], 1e-6, 1.0 - 1e-6
        )

    oof_logits = np.log(oof_probs / (1.0 - oof_probs)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=seed)
    calibrator.fit(oof_logits, y_train)

    full_model = model_factory()
    full_model.fit(X_train, y_train)

    return PlattCalibratedModel(estimator=full_model, calibrator=calibrator)


def _tune_lightgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float,
    cfg: TrainingConfig,
) -> Dict[str, Any]:
    """Optimize LightGBM hyperparameters via Optuna maximizing CV PR-AUC."""
    skf = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 180, 480),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 48),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 25, 120),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 12.0),
            "subsample_freq": 1,
            "random_state": cfg.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        cv_scores: List[float] = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            clf = LGBMClassifier(**params)
            clf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            probs = clf.predict_proba(X_train.iloc[va_idx])[:, 1]
            cv_scores.append(float(average_precision_score(y_train[va_idx], probs)))
        return float(np.mean(cv_scores))

    sampler = optuna.samplers.TPESampler(seed=cfg.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update(
        {
            "subsample_freq": 1,
            "random_state": cfg.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    logger.info(
        "Optuna finished %d trials | Best 3-fold CV PR-AUC: %.4f",
        cfg.n_optuna_trials,
        study.best_value,
    )
    return best_params


def _optimize_ensemble_and_threshold(
    p_lgb_val: np.ndarray,
    p_cat_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, float, float]:
    """Select optimal ensemble blend weight (PR-AUC) and decision threshold (F1) on validation."""
    best_weight = 0.5
    best_val_pr_auc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * p_lgb_val + (1.0 - w) * p_cat_val
        score = float(average_precision_score(y_val, blend))
        if score > best_val_pr_auc:
            best_val_pr_auc = score
            best_weight = float(round(w, 2))

    val_blend = best_weight * p_lgb_val + (1.0 - best_weight) * p_cat_val

    best_threshold = 0.5
    best_val_f1 = -1.0
    for thresh in np.linspace(0.05, 0.90, 86):
        preds = (val_blend >= thresh).astype(np.int32)
        score = float(f1_score(y_val, preds, zero_division=0))
        if score > best_val_f1:
            best_val_f1 = score
            best_threshold = float(round(thresh, 4))

    logger.info(
        "Validation optimization | lgbm_weight=%.2f (val PR-AUC=%.4f) | threshold=%.4f (val F1=%.4f)",
        best_weight,
        best_val_pr_auc,
        best_threshold,
        best_val_f1,
    )
    return best_weight, best_threshold, best_val_pr_auc


def _sync_readme_metrics_table(metrics_payload: Dict[str, Any]) -> None:
    """Update the metrics comparison table in README.md if markers exist."""
    if not README_PATH.exists():
        return

    text = README_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- METRICS_TABLE_START -->"
    end_marker = "<!-- METRICS_TABLE_END -->"
    if start_marker not in text or end_marker not in text:
        return

    s1 = metrics_payload["stages"]["stage_1_baseline_logreg"]
    s2 = metrics_payload["stages"]["stage_2_fe_lgbm"]
    s3 = metrics_payload["stages"]["stage_3_tuned_ensemble"]

    table = "\n".join(
        [
            start_marker,
            "| Iteration Stage | Model & Pipeline | PR-AUC | F1-Score | Recall | Precision | ROC-AUC |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **1. Baseline** | LogisticRegression (Raw + Scaled) | {s1['pr_auc']:.4f} | {s1['f1']:.4f} | {s1['recall']:.4f} | {s1['precision']:.4f} | {s1['roc_auc']:.4f} |",
            f"| **2. Feature Engineering** | LightGBM (Domain + Velocity + Entity Priors) | {s2['pr_auc']:.4f} | {s2['f1']:.4f} | {s2['recall']:.4f} | {s2['precision']:.4f} | {s2['roc_auc']:.4f} |",
            f"| **3. Tuned Ensemble** | Optuna LGBM + CatBoost (Calibrated + Threshold `{s3['threshold']:.2f}`) | **{s3['pr_auc']:.4f}** | **{s3['f1']:.4f}** | **{s3['recall']:.4f}** | **{s3['precision']:.4f}** | **{s3['roc_auc']:.4f}** |",
            end_marker,
        ]
    )

    updated = re.sub(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        table,
        text,
        flags=re.DOTALL,
    )
    README_PATH.write_text(updated, encoding="utf-8")
    logger.info("Synchronized metrics table in %s", README_PATH)


def run_training(cfg: TrainingConfig | None = None) -> Dict[str, Any]:
    """Execute the full 3-stage training, calibration, SHAP export, and evaluation workflow."""
    cfg = cfg or TrainingConfig()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python data/fetch_data.py` first."
        )

    logger.info("Loading dataset from %s...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    train_df, val_df, test_df = chronological_split(df)

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.int32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int32)

    X_train_raw = train_df.drop(columns=[TARGET_COL])
    X_val_raw = val_df.drop(columns=[TARGET_COL])
    X_test_raw = test_df.drop(columns=[TARGET_COL])

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    logger.info(
        "Chronological split | Train=%d (%.2f%% fraud) | Val=%d (%.2f%% fraud) | Test=%d (%.2f%% fraud)",
        len(train_df),
        100.0 * y_train.mean(),
        len(val_df),
        100.0 * y_val.mean(),
        len(test_df),
        100.0 * y_test.mean(),
    )

    # ------------------------------------------------------------------
    # Stage 1: Baseline LogisticRegression on minimal raw features
    # ------------------------------------------------------------------
    logger.info("Stage 1/3: Training Baseline LogisticRegression...")
    baseline_pipe = Pipeline(
        steps=[
            ("preprocess", build_baseline_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=cfg.random_seed,
                ),
            ),
        ]
    )
    baseline_pipe.fit(X_train_raw, y_train)
    p_test_s1 = baseline_pipe.predict_proba(X_test_raw)[:, 1]
    s1_metrics = evaluate_predictions(y_test, p_test_s1, threshold=0.50)
    logger.info(
        "Stage 1 Test | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f",
        s1_metrics["pr_auc"],
        s1_metrics["f1"],
        s1_metrics["recall"],
        s1_metrics["precision"],
    )

    # ------------------------------------------------------------------
    # Stage 2: Feature-Engineered LightGBM (default hyperparameters)
    # ------------------------------------------------------------------
    logger.info("Stage 2/3: Fitting zero-leakage feature pipeline & default LightGBM...")
    fe_pipeline = build_feature_pipeline()
    X_train_fe = fe_pipeline.fit_transform(X_train_raw, y_train)
    X_val_fe = fe_pipeline.transform(X_val_raw)
    X_test_fe = fe_pipeline.transform(X_test_raw)
    feature_names = get_engineered_feature_names()

    s2_params = dict(cfg.default_lgbm_params)
    s2_params["scale_pos_weight"] = min(scale_pos_weight, 8.0)
    lgbm_default = LGBMClassifier(**s2_params)
    lgbm_default.fit(X_train_fe, y_train)
    p_test_s2 = lgbm_default.predict_proba(X_test_fe)[:, 1]
    s2_metrics = evaluate_predictions(y_test, p_test_s2, threshold=0.50)
    logger.info(
        "Stage 2 Test | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f",
        s2_metrics["pr_auc"],
        s2_metrics["f1"],
        s2_metrics["recall"],
        s2_metrics["precision"],
    )

    # ------------------------------------------------------------------
    # Stage 3: Optuna-Tuned LightGBM + CatBoost Calibrated Ensemble
    # ------------------------------------------------------------------
    logger.info("Stage 3/3: Running Optuna hyperparameter optimization for LightGBM...")
    best_lgb_params = _tune_lightgbm_with_optuna(
        X_train=X_train_fe,
        y_train=y_train,
        scale_pos_weight=scale_pos_weight,
        cfg=cfg,
    )

    logger.info("Fitting Platt-calibrated Optuna LightGBM and CatBoost models...")
    calibrated_lgbm = _fit_platt_calibrated_model(
        model_factory=lambda: LGBMClassifier(**best_lgb_params),
        X_train=X_train_fe,
        y_train=y_train,
        cv_folds=cfg.cv_folds,
        seed=cfg.random_seed,
    )

    cat_params = dict(cfg.default_catboost_params)
    cat_params["scale_pos_weight"] = 4.0
    calibrated_cat = _fit_platt_calibrated_model(
        model_factory=lambda: CatBoostClassifier(**cat_params),
        X_train=X_train_fe,
        y_train=y_train,
        cv_folds=cfg.cv_folds,
        seed=cfg.random_seed,
    )

    p_lgb_val = calibrated_lgbm.predict_proba(X_val_fe)[:, 1]
    p_cat_val = calibrated_cat.predict_proba(X_val_fe)[:, 1]
    lgbm_weight, optimal_threshold, val_pr_auc = _optimize_ensemble_and_threshold(
        p_lgb_val, p_cat_val, y_val
    )

    ensemble = FraudDetectionEnsemble(
        feature_pipeline=fe_pipeline,
        lgbm_model=calibrated_lgbm,
        catboost_model=calibrated_cat,
        lgbm_weight=lgbm_weight,
        optimal_threshold=optimal_threshold,
        feature_names=feature_names,
    )

    p_test_s3 = ensemble.predict_proba_from_transformed(X_test_fe)
    s3_metrics = evaluate_predictions(y_test, p_test_s3, threshold=optimal_threshold)
    logger.info(
        "Stage 3 Test | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f | Threshold=%.4f",
        s3_metrics["pr_auc"],
        s3_metrics["f1"],
        s3_metrics["recall"],
        s3_metrics["precision"],
        optimal_threshold,
    )

    # ------------------------------------------------------------------
    # SHAP Explainer & Global Feature Importance
    # ------------------------------------------------------------------
    logger.info("Building dual TreeExplainer bundle for local & global SHAP...")
    lgb_explainer = shap.TreeExplainer(calibrated_lgbm.estimator)
    cat_explainer = shap.TreeExplainer(calibrated_cat.estimator)
    shap_bundle = ShapExplainerBundle(
        lgbm_explainer=lgb_explainer,
        catboost_explainer=cat_explainer,
        lgbm_weight=lgbm_weight,
        feature_names=feature_names,
    )

    # Compute global feature importance on a stratified test sample (500 rows)
    sample_idx = np.linspace(0, len(X_test_fe) - 1, min(500, len(X_test_fe)), dtype=int)
    X_sample = X_test_fe.iloc[sample_idx]
    lgb_shap = lgb_explainer.shap_values(X_sample)
    if isinstance(lgb_shap, list):
        lgb_shap = lgb_shap[1]
    cat_shap = cat_explainer.shap_values(X_sample)
    if isinstance(cat_shap, list):
        cat_shap = cat_shap[1]
    blended_abs_shap = np.mean(
        np.abs(lgbm_weight * np.asarray(lgb_shap) + (1.0 - lgbm_weight) * np.asarray(cat_shap)),
        axis=0,
    )
    top_global_features = [
        {"feature": feat, "mean_abs_shap": round(float(val), 5)}
        for feat, val in sorted(
            zip(feature_names, blended_abs_shap), key=lambda x: x[1], reverse=True
        )[:15]
    ]

    # Persist artifacts
    joblib.dump(ensemble, MODEL_PATH, compress=3)
    joblib.dump(shap_bundle, EXPLAINER_PATH, compress=3)
    logger.info("Saved model artifact to %s (%.2f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)
    logger.info(
        "Saved SHAP artifact to %s (%.2f MB)", EXPLAINER_PATH, EXPLAINER_PATH.stat().st_size / 1e6
    )

    # Compute relative improvements vs Baseline
    pr_auc_lift_pct = (
        100.0 * (s3_metrics["pr_auc"] - s1_metrics["pr_auc"]) / max(s1_metrics["pr_auc"], 1e-6)
    )
    f1_lift_pct = 100.0 * (s3_metrics["f1"] - s1_metrics["f1"]) / max(s1_metrics["f1"], 1e-6)

    metrics_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "overall_fraud_rate": round(float(df[TARGET_COL].mean()), 5),
            "test_fraud_count": int(y_test.sum()),
            "n_engineered_features": len(feature_names),
        },
        "optimization": {
            "n_optuna_trials": cfg.n_optuna_trials,
            "best_lgbm_params": best_lgb_params,
            "lgbm_ensemble_weight": lgbm_weight,
            "catboost_ensemble_weight": round(1.0 - lgbm_weight, 2),
            "val_pr_auc": round(val_pr_auc, 4),
            "optimal_threshold": optimal_threshold,
            "pr_auc_lift_vs_baseline_pct": round(pr_auc_lift_pct, 2),
            "f1_lift_vs_baseline_pct": round(f1_lift_pct, 2),
        },
        "stages": {
            "stage_1_baseline_logreg": s1_metrics,
            "stage_2_fe_lgbm": s2_metrics,
            "stage_3_tuned_ensemble": s3_metrics,
        },
        "top_global_shap_features": top_global_features,
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Wrote metrics report to %s", METRICS_PATH)

    _sync_readme_metrics_table(metrics_payload)
    return metrics_payload


if __name__ == "__main__":
    run_training()
