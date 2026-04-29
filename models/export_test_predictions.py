"""
Export test-set probability predictions (y_proba) and labels (y_true) for the
four headline models on the 12 RFECV feature pipeline.

Models
------
1. LogisticRegression — fit on scaled train from ``get_canonical_splits()``.
2. RandomForest — ``RandomizedSearchCV`` (same search space as ``baselines.BaselineComparison``,
   ``n_iter=30``, ``TimeSeriesSplit(5)``), fit on scaled train.
3. XGBoost — loaded from ``models/checkpoints/xgb_tuned_12feat.pkl`` (Pipeline with internal
   ``StandardScaler``); **expects raw (unscaled) rows** → uses ``inverse_transform`` on test.
4. Neural network — loaded from ``models/checkpoints/nn_12feat.pt``; expects **scaled** test rows.

Outputs (under ``outputs/``)
----------------------------
- ``test_predictions_four_models.npz`` — arrays: ``y_true``, ``logreg_y_proba``, ``rf_y_proba``,
  ``xgb_y_proba``, ``nn_y_proba``, ``event_ids_test`` (optional)
- ``test_predictions_meta.json`` — short provenance / shapes

Run:  python -m models.export_test_predictions
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config import BASE_DIR
from models.baselines import RANDOM_STATE
from models.data_loader import get_canonical_splits
from models.nn_binary import load_binary_nn, predict_proba as nn_predict_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINTS = BASE_DIR / "models" / "checkpoints"


def _raw_from_splits(splits: dict) -> tuple[np.ndarray, np.ndarray]:
    sc = splits["scaler"]
    X_train_raw = sc.inverse_transform(splits["X_train"]).astype(np.float64)
    X_test_raw = sc.inverse_transform(splits["X_test"]).astype(np.float64)
    return X_train_raw, X_test_raw


def main(db_path: str = "data/ufc_matchmaker.db") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path)
    X_tr_s = np.asarray(splits["X_train"], dtype=np.float64)
    X_te_s = np.asarray(splits["X_test"], dtype=np.float64)
    y_tr = splits["y_train"].astype(np.int64)
    y_te = splits["y_test"].astype(np.int64)
    X_tr_raw, X_te_raw = _raw_from_splits(splits)
    event_ids = splits.get("event_ids_test")
    if event_ids is not None:
        event_ids = np.asarray(event_ids, dtype=np.int32)

    feature_names = splits.get("feature_names", [])

    # --- 1. LogReg (scaled) ---
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    logreg.fit(X_tr_s, y_tr)
    logreg_proba = logreg.predict_proba(X_te_s)[:, 1].astype(np.float64)

    # --- 2. Random Forest (scaled), tuned like baselines ---
    logger.info("Tuning RandomForest for export (n_iter=30)...")
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_distributions = {
        "n_estimators": [200, 500, 1000],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=5),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_tr_s, y_tr)
    rf = rf_search.best_estimator_
    rf_proba = rf.predict_proba(X_te_s)[:, 1].astype(np.float64)

    # --- 3. XGB (checkpoint; raw input) ---
    xgb_path = CHECKPOINTS / "xgb_tuned_12feat.pkl"
    if not xgb_path.is_file():
        raise FileNotFoundError(f"Missing {xgb_path} — run python -m models.xgb_tuning")
    xgb_pipe = joblib.load(xgb_path)
    xgb_proba = xgb_pipe.predict_proba(X_te_raw)[:, 1].astype(np.float64)

    # --- 4. NN checkpoint (scaled) ---
    nn_path = CHECKPOINTS / "nn_12feat.pt"
    if not nn_path.is_file():
        raise FileNotFoundError(f"Missing {nn_path} — run python -m models.nn_binary")
    nn_model = load_binary_nn(str(nn_path))
    nn_proba = nn_predict_proba(nn_model, X_te_s).astype(np.float64)

    npz_path = OUTPUT_DIR / "test_predictions_four_models.npz"
    save_kw: dict = {
        "y_true": y_te,
        "logreg_y_proba": logreg_proba,
        "rf_y_proba": rf_proba,
        "xgb_y_proba": xgb_proba,
        "nn_y_proba": nn_proba,
    }
    if event_ids is not None:
        save_kw["event_ids_test"] = event_ids
    np.savez_compressed(npz_path, **save_kw)

    meta = {
        "npz": str(npz_path),
        "n_test": int(len(y_te)),
        "feature_dim": int(X_te_s.shape[1]),
        "feature_names": list(feature_names),
        "y_true": "is_bonus_fight, test split from get_canonical_splits (temporal)",
        "random_forest_best_params": rf_search.best_params_,
        "random_forest_cv_auc_mean": float(rf_search.best_score_),
        "models": {
            "logreg": "LogisticRegression fit on scaled X_train; proba from scaled X_test",
            "random_forest": "Tuned RF (30 RandomizedSearch × TimeSeriesSplit-5) on scaled train",
            "xgboost": str(xgb_path.name) + " — Pipeline(StandardScaler, XGB); proba from raw X_test",
            "neural_network": str(nn_path.name) + " — scaled X_test",
        },
    }
    meta_path = OUTPUT_DIR / "test_predictions_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {npz_path}")
    print(f"Saved {meta_path}")
    print(f"  shapes: y_true {y_te.shape}, each y_proba {logreg_proba.shape}")
    return npz_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
