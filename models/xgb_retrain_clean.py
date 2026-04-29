"""
Retrain XGBoost on the distribution-stable 5-feature subset.

The new ``SELECTED_FEATURES`` in ``pipeline_config.py`` (``RFECV_OPTIMAL_CLEAN``)
contains only features whose train and test distributions match (test_std/train_std
≤ 1.5). All eight rolling features and the two rolling-derived cross-features
from the original 12-feature subset are gone.

Pipeline: ``StandardScaler + XGBClassifier`` with ``scale_pos_weight=1`` (the
calibrator in ``calibrate_xgb.py`` handles class bias post-hoc — baking it in
just inflates outputs and forces the calibrator to undo it).

Run:  python -m models.xgb_retrain_clean
Then: python -m models.calibrate_xgb xgb_tuned_clean.pkl
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import MODELS_DIR
from models.data_loader import get_canonical_splits

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RANDOM_STATE = 42
DST_PIPELINE = CHECKPOINT_DIR / "xgb_tuned_clean.pkl"
DST_METRICS = CHECKPOINT_DIR / "xgb_tuned_clean_metrics.json"


def _param_distributions() -> dict:
    return {
        "xgb__max_depth": [2, 3, 4],
        "xgb__min_child_weight": [5, 10, 15, 20, 30],
        "xgb__gamma": uniform(0.0, 2.0),
        "xgb__reg_alpha": uniform(0.1, 10.0),
        "xgb__reg_lambda": uniform(1.0, 20.0),
        "xgb__subsample": uniform(0.5, 0.3),
        "xgb__colsample_bytree": uniform(0.4, 0.6),
        "xgb__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "xgb__n_estimators": [200, 300, 500, 800],
        "xgb__scale_pos_weight": [1.0],
        "xgb__random_state": [RANDOM_STATE],
        "xgb__n_jobs": [1],
    }


def main(db_path: str = "data/ufc_matchmaker.db", n_iter: int = 200) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path)
    sc = splits["scaler"]
    X_train = sc.inverse_transform(splits["X_train"]).astype(np.float64)
    X_val = sc.inverse_transform(splits["X_val"]).astype(np.float64)
    X_test = sc.inverse_transform(splits["X_test"]).astype(np.float64)
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]

    n_train, n_feat = X_train.shape
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(objective="binary:logistic", eval_metric="auc", verbosity=0)),
    ])
    cv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipe, param_distributions=_param_distributions(),
        n_iter=n_iter, cv=cv, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1, refit=True,
    )

    print(f"Training on {n_train} rows × {n_feat} features "
          f"(StandardScaler + XGB, scale_pos_weight=1, n_iter={n_iter})")
    print(f"Features: {splits['feature_names']}")
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    best_cv = float(search.best_score_)
    train_auc = float(roc_auc_score(y_train, best_pipe.predict_proba(X_train)[:, 1]))
    val_auc = float(roc_auc_score(y_val, best_pipe.predict_proba(X_val)[:, 1]))
    test_auc = float(roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1]))

    print()
    print("=" * 62)
    print(f"  Mean CV AUC (train):        {best_cv:.4f}")
    print(f"  Train AUC (best estimator): {train_auc:.4f}   (gap={train_auc - best_cv:+.4f})")
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
        "train_cv_gap": train_auc - best_cv,
        "val_auc": val_auc,
        "test_auc_peek": test_auc,
        "n_iter": n_iter,
        "n_features": n_feat,
        "feature_names": list(splits["feature_names"]),
        "scale_pos_weight": 1.0,
    }
    DST_METRICS.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
