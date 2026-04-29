"""
Retrain XGBoost with distribution-shift-robust preprocessing.

Two changes vs ``xgb_tuning.py``:
  1. ``QuantileTransformer(output_distribution='normal')`` instead of
     ``StandardScaler``. Maps train values to ranks → standard normal, so
     test inputs that fall outside the train range get clipped to the
     extreme quantiles instead of producing 5–10σ outliers. This was the
     root cause of every rolling feature having ~0 SHAP importance:
     ``f1/f2_roll_recent_knockdowns_norm`` had train zero-rate 0.74% but
     test zero-rate 44.66% (cold-start fighters), and other rolling
     features had test std up to 10× train std.
  2. ``scale_pos_weight=1`` (was 2.71). Probability calibration is now
     handled post-hoc by ``models/calibrate_xgb.py``; baking class-weight
     bias into ``predict_proba`` just inflates outputs and forces the
     calibrator to undo it.

Run:  python -m models.xgb_retrain_robust
Then: python -m models.calibrate_xgb xgb_tuned_12feat_robust.pkl
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
from scipy.stats import uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from config import MODELS_DIR
from models.data_loader import get_canonical_splits

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RANDOM_STATE = 42
DST_PIPELINE = CHECKPOINT_DIR / "xgb_tuned_12feat_robust.pkl"
DST_METRICS = CHECKPOINT_DIR / "xgb_tuned_12feat_robust_metrics.json"


def _param_distributions() -> dict:
    return {
        "xgb__max_depth": [2, 3, 4],
        "xgb__min_child_weight": [5, 10, 15, 20, 30],
        "xgb__gamma": uniform(0.0, 2.0),
        "xgb__reg_alpha": uniform(0.1, 10.0),
        "xgb__reg_lambda": uniform(1.0, 20.0),
        "xgb__subsample": uniform(0.5, 0.3),
        "xgb__colsample_bytree": uniform(0.4, 0.4),
        "xgb__colsample_bylevel": uniform(0.5, 0.5),
        "xgb__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "xgb__n_estimators": [200, 300, 500, 800],
        "xgb__scale_pos_weight": [1.0],  # post-hoc calibration handles class bias
        "xgb__random_state": [RANDOM_STATE],
        "xgb__n_jobs": [1],
    }


def _build_pipeline(n_train: int) -> Pipeline:
    # n_quantiles capped at sample size; 100 is fine for 676 train rows.
    qt = QuantileTransformer(
        n_quantiles=min(100, n_train),
        output_distribution="normal",
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        verbosity=0,
    )
    return Pipeline([("scaler", qt), ("xgb", xgb)])


def main(db_path: str = "data/ufc_matchmaker.db", n_iter: int = 200) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path)
    sc = splits["scaler"]
    X_train_raw = sc.inverse_transform(splits["X_train"]).astype(np.float64)
    X_val_raw = sc.inverse_transform(splits["X_val"]).astype(np.float64)
    X_test_raw = sc.inverse_transform(splits["X_test"]).astype(np.float64)
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]

    n_train = X_train_raw.shape[0]
    pipe = _build_pipeline(n_train)
    cv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=_param_distributions(),
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print(f"Training on {n_train} rows × {X_train_raw.shape[1]} features "
          f"(QuantileTransformer + XGB, scale_pos_weight=1, n_iter={n_iter})")
    search.fit(X_train_raw, y_train)

    best_pipe = search.best_estimator_
    best_cv = float(search.best_score_)
    train_auc = float(roc_auc_score(y_train, best_pipe.predict_proba(X_train_raw)[:, 1]))
    val_auc = float(roc_auc_score(y_val, best_pipe.predict_proba(X_val_raw)[:, 1]))
    test_auc = float(roc_auc_score(y_test, best_pipe.predict_proba(X_test_raw)[:, 1]))
    gap = train_auc - best_cv

    print()
    print("=" * 62)
    print(f"  Mean CV AUC (train):        {best_cv:.4f}")
    print(f"  Train AUC (best estimator): {train_auc:.4f}   (gap={gap:+.4f})")
    print(f"  Val AUC:                    {val_auc:.4f}")
    print(f"  Test AUC (peek):            {test_auc:.4f}")
    print("=" * 62)
    print("\nBest params:")
    for k in sorted(search.best_params_):
        v = search.best_params_[k]
        if isinstance(v, (np.floating, np.integer)):
            v = float(v) if isinstance(v, np.floating) else int(v)
        print(f"  {k}: {v!r}")

    joblib.dump(best_pipe, DST_PIPELINE)
    print(f"\nSaved: {DST_PIPELINE}")

    metrics = {
        "best_cv_auc": best_cv,
        "train_auc": train_auc,
        "train_cv_gap": gap,
        "val_auc": val_auc,
        "test_auc_peek": test_auc,
        "n_iter": n_iter,
        "scaler": "QuantileTransformer(output_distribution='normal')",
        "scale_pos_weight": 1.0,
    }
    DST_METRICS.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
