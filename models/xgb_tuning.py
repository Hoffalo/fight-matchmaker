"""
Hyperparameter search for XGBoost on the small-n bonus pipeline.

- **12-D RFECV path** (``run()``): raw feature columns after augment + inverse-scale
  so ``StandardScaler`` refits each CV fold.
- **PCA path** (``run_pca_xgb_tuning()``): loads ``pca_transformer.pkl`` from
  ``pca_pipeline`` (best LogReg CV component count), transforms scaled 115-D train
  rows, then ``Pipeline(StandardScaler, XGB)`` on PCA coordinates. Saves
  ``xgb_pca_tuned.pkl`` only if CV AUC beats the 12-feat XGB reference (~0.5832).

Run:  python -m models.xgb_tuning          # 12-feat RFECV subset
      python -m models.xgb_tuning pca      # PCA features
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import MODELS_DIR, SCALE_POS_WEIGHT
from models.data_loader import get_canonical_splits

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RANDOM_STATE = 42
LOGREG_CV_REF = 0.5945  # reference from feature_selection RFECV_opt curve
XGB_12_FEAT_CV_REF = 0.5832  # tuned XGB on 12 RFECV features (user-reported)
GAP_12_FEAT_XGB_REF = 0.069  # train − CV gap on 12-feat XGB (user-reported)
PCA_TRANSFORMER_PATH = CHECKPOINT_DIR / "pca_transformer.pkl"
PCA_META_PATH = CHECKPOINT_DIR / "pca_best_meta.json"
XGB_PCA_CHECKPOINT = CHECKPOINT_DIR / "xgb_pca_tuned.pkl"


def _xgb_param_distributions() -> dict:
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
        "xgb__scale_pos_weight": [float(SCALE_POS_WEIGHT)],
        "xgb__random_state": [RANDOM_STATE],
        "xgb__n_jobs": [1],
    }


def _xgb_pca_param_distributions() -> dict:
    """Search space for XGBoost on PCA features (denser, decorrelated inputs)."""
    return {
        "xgb__max_depth": [2, 3, 4, 5],
        "xgb__min_child_weight": [3, 5, 10, 15],
        "xgb__gamma": uniform(0.0, 2.0),
        "xgb__reg_alpha": uniform(0.1, 10.0),
        "xgb__reg_lambda": uniform(1.0, 20.0),
        "xgb__subsample": uniform(0.5, 0.4),
        "xgb__colsample_bytree": uniform(0.4, 0.6),
        "xgb__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "xgb__n_estimators": [200, 300, 500, 800],
        "xgb__scale_pos_weight": [float(SCALE_POS_WEIGHT)],
        "xgb__random_state": [RANDOM_STATE],
        "xgb__n_jobs": [1],
    }


def _build_xgb_pca_search() -> RandomizedSearchCV:
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        verbosity=0,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("xgb", base),
        ]
    )
    cv = TimeSeriesSplit(n_splits=5)
    return RandomizedSearchCV(
        pipe,
        param_distributions=_xgb_pca_param_distributions(),
        n_iter=200,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def _build_xgb_search() -> RandomizedSearchCV:
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        verbosity=0,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("xgb", base),
        ]
    )
    cv = TimeSeriesSplit(n_splits=5)
    return RandomizedSearchCV(
        pipe,
        param_distributions=_xgb_param_distributions(),
        n_iter=200,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def _raw_train_val(splits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert canonical scaling to recover raw features for per-fold scaling in CV."""
    sc = splits["scaler"]
    X_train_raw = sc.inverse_transform(splits["X_train"])
    X_val_raw = sc.inverse_transform(splits["X_val"])
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    return X_train_raw.astype(np.float64), y_train, X_val_raw.astype(np.float64), y_val


def _json_friendly_params(p: dict) -> dict:
    out = {}
    for key, v in p.items():
        if isinstance(v, (np.floating, float)):
            out[key] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[key] = int(v)
        else:
            out[key] = v
    return out


def _print_top_configs(search: RandomizedSearchCV, k: int = 10) -> None:
    results = search.cv_results_
    scores = np.asarray(results["mean_test_score"])
    order = np.argsort(-scores)[:k]
    print(f"\nTop {k} configurations (mean CV ROC-AUC):")
    for rank, idx in enumerate(order, 1):
        p = _json_friendly_params(results["params"][idx])
        print(f"  #{rank}  score={scores[idx]:.4f}  params={p}")


def run(
    db_path: str = "data/ufc_matchmaker.db",
    n_iter: int = 200,
) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path)
    X_train_raw, y_train, X_val_raw, y_val = _raw_train_val(splits)
    n_feat = X_train_raw.shape[1]
    logger.info("Tuning XGBoost: train=%s feats=%d", X_train_raw.shape, n_feat)

    search = _build_xgb_search()
    search.n_iter = n_iter
    search.fit(X_train_raw, y_train)

    best_cv = float(search.best_score_)
    print("\n--- RandomizedSearchCV (Pipeline: scaler + XGB) ---")
    print("Best params (full dict):")
    for k_sorted in sorted(search.best_params_.keys()):
        v = search.best_params_[k_sorted]
        if isinstance(v, (np.floating, np.integer)):
            v = float(v) if isinstance(v, np.floating) else int(v)
        print(f"  {k_sorted}: {v!r}")
    print(f"Best CV ROC-AUC (mean): {best_cv:.4f}")

    _print_top_configs(search)

    best_pipe = search.best_estimator_
    train_proba = best_pipe.predict_proba(X_train_raw)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_proba))
    gap = train_auc - best_cv
    print("\n--- Generalization check (best RandomizedSearch model) ---")
    print(f"Train AUC:     {train_auc:.4f}")
    print(f"Mean CV AUC:   {best_cv:.4f}")
    print(f"Gap (train - CV): {gap:+.4f}  ", end="")
    if gap > 0.05:
        print("(> 0.05 — likely still overfitting)")
    else:
        print("(≤ 0.05)")
    if train_auc > best_cv + 0.05:
        print("Warning: train_auc exceeds mean CV by >0.05 — still overfitting.")

    # LogReg same CV / per-fold scaler
    log_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    cv = TimeSeriesSplit(n_splits=5)
    log_cv_scores = cross_val_score(
        log_pipe, X_train_raw, y_train, cv=cv, scoring="roc_auc", n_jobs=1
    )
    log_cv_mean = float(np.mean(log_cv_scores))

    # Early stopping (single train/val split; scaler on full train)
    scaler_es = StandardScaler().fit(X_train_raw)
    X_tr_es = scaler_es.transform(X_train_raw)
    X_va_es = scaler_es.transform(X_val_raw)

    best_xgb = best_pipe.named_steps["xgb"]
    xgb_early = clone(best_xgb)
    xgb_early.set_params(n_estimators=2000, early_stopping_rounds=50)
    xgb_early.fit(
        X_tr_es,
        y_train,
        eval_set=[(X_va_es, y_val)],
        verbose=False,
    )
    val_proba_early = xgb_early.predict_proba(X_va_es)[:, 1]
    val_auc_early = float(roc_auc_score(y_val, val_proba_early))
    best_it = getattr(xgb_early, "best_iteration", None)
    if best_it is None:
        best_it = getattr(xgb_early, "best_iteration_", None)
    print(f"\nEarly stopping: best_iteration ≈ {best_it}")

    print("\nModel comparison on %d features:" % n_feat)
    print(f"  LogReg (balanced, ref):  CV AUC ≈ {LOGREG_CV_REF:.4f}")
    print(f"  LogReg (balanced, same CV): mean CV AUC = {log_cv_mean:.4f}")
    print(f"  XGBoost (tuned search):    mean CV AUC = {best_cv:.4f}")
    print(f"  XGBoost (early stop):      Val AUC     = {val_auc_early:.4f}")

    xgb_path = CHECKPOINT_DIR / "xgb_tuned_12feat.pkl"
    scaler_path = CHECKPOINT_DIR / "scaler_12feat.pkl"
    joblib.dump(best_pipe, xgb_path)
    joblib.dump(best_pipe.named_steps["scaler"], scaler_path)
    print(f"\nSaved pipeline → {xgb_path}")
    print(f"Saved scaler   → {scaler_path}")

    metrics_path = CHECKPOINT_DIR / "xgb_tuned_metrics.json"
    metrics_path.write_text(
        json.dumps({"best_cv_auc": best_cv, "val_auc_early_stop": val_auc_early}, indent=2)
    )

    return {
        "best_cv_auc": best_cv,
        "train_auc": train_auc,
        "train_cv_gap": gap,
        "logreg_cv_mean": log_cv_mean,
        "val_auc_early_stop": val_auc_early,
        "best_params": dict(search.best_params_),
        "best_iteration": best_it,
        "checkpoint_pipeline": str(xgb_path),
        "checkpoint_scaler": str(scaler_path),
    }


def run_pca_xgb_tuning(
    db_path: str = "data/ufc_matchmaker.db",
    n_iter: int = 200,
    beat_threshold: float = XGB_12_FEAT_CV_REF,
) -> dict:
    """
    Tune XGBoost on PCA-transformed training data (best ``pca_transformer.pkl``).
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not PCA_TRANSFORMER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PCA_TRANSFORMER_PATH}. Run: python -m models.pca_pipeline"
        )

    pca = joblib.load(PCA_TRANSFORMER_PATH)
    n_components = int(pca.components_.shape[0])

    meta_n: int | None = None
    if PCA_META_PATH.is_file():
        try:
            meta = json.loads(PCA_META_PATH.read_text())
            meta_n = int(meta.get("best_n_components", n_components))
        except (json.JSONDecodeError, TypeError, ValueError):
            meta_n = None
    if meta_n is not None and meta_n != n_components:
        logger.warning(
            "PCA meta best_n=%s vs transformer n=%s — using transformer.",
            meta_n,
            n_components,
        )

    splits = get_canonical_splits(db_path, subset_features=False)
    X_train = np.asarray(splits["X_train"], dtype=np.float64)
    y_train = splits["y_train"]
    X_train_pca = pca.transform(X_train)

    logger.info(
        "PCA XGB tuning: train=%s PCA_dim=%d (scaled 115-D → PCA)",
        X_train_pca.shape,
        n_components,
    )

    search = _build_xgb_pca_search()
    search.n_iter = n_iter
    search.fit(X_train_pca, y_train)

    best_cv = float(search.best_score_)
    best_pipe = search.best_estimator_
    train_proba = best_pipe.predict_proba(X_train_pca)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_proba))
    gap = train_auc - best_cv

    print("\n--- RandomizedSearchCV (PCA features: Pipeline scaler + XGB) ---")
    print("Best params (full dict):")
    for k_sorted in sorted(search.best_params_.keys()):
        v = search.best_params_[k_sorted]
        if isinstance(v, (np.floating, np.integer)):
            v = float(v) if isinstance(v, np.floating) else int(v)
        print(f"  {k_sorted}: {v!r}")
    print(f"Best CV ROC-AUC (mean): {best_cv:.4f}")

    _print_top_configs(search, k=10)

    print("\n--- Generalization (PCA + XGB best estimator) ---")
    print(f"Train AUC:       {train_auc:.4f}")
    print(f"Mean CV AUC:     {best_cv:.4f}")
    print(f"Gap (train − CV): {gap:+.4f}  ", end="")
    if gap > 0.05:
        print("(> 0.05)")
    else:
        print("(≤ 0.05)")
    if gap < 0.05:
        print(
            "Gap under 0.05 — less overfitting than the ~0.069 12-feat XGB reference; "
            "dense/decorrelated PCA inputs often stabilize trees."
        )
    elif gap < GAP_12_FEAT_XGB_REF:
        print(
            f"Gap below 12-feat XGB reference (~{GAP_12_FEAT_XGB_REF:.3f}) but still above 0.05."
        )
    else:
        print(
            f"Gap still large vs targets (0.05 and ~{GAP_12_FEAT_XGB_REF:.3f} 12-feat ref)."
        )

    cv = TimeSeriesSplit(n_splits=5)
    log_pca = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    log_pca_scores = cross_val_score(
        log_pca, X_train_pca, y_train, cv=cv, scoring="roc_auc", n_jobs=1,
    )
    log_pca_mean = float(np.mean(log_pca_scores))
    log_pca_std = float(np.std(log_pca_scores))

    print("\n" + "=" * 62)
    print(f"{'Feature Set':<26} {'Model':<10} {'CV AUC':<12}")
    print(f"{'─' * 26} {'─' * 10} {'─' * 12}")
    print(f"{'12-feat RFECV':<26} {'LogReg':<10} {LOGREG_CV_REF:<12.4f}")
    print(f"{'12-feat RFECV':<26} {'XGBoost':<10} {XGB_12_FEAT_CV_REF:<12.4f}")
    print(
        f"{f'PCA-{n_components} components':<26} {'LogReg':<10} "
        f"{log_pca_mean:.4f} ± {log_pca_std:.4f}"
    )
    print(f"{f'PCA-{n_components} components':<26} {'XGBoost':<10} {best_cv:<12.4f}")
    print("=" * 62)

    saved_path: str | None = None
    if best_cv > beat_threshold:
        joblib.dump(best_pipe, XGB_PCA_CHECKPOINT)
        saved_path = str(XGB_PCA_CHECKPOINT)
        print(
            f"\nSaved best PCA+XGB pipeline (CV {best_cv:.4f} > {beat_threshold:.4f}) → {saved_path}"
        )
        (CHECKPOINT_DIR / "xgb_pca_tuned_metrics.json").write_text(
            json.dumps(
                {
                    "best_cv_auc": best_cv,
                    "train_auc": train_auc,
                    "train_cv_gap": gap,
                    "n_pca_components": n_components,
                    "beat_threshold": beat_threshold,
                    "logreg_pca_cv_mean": log_pca_mean,
                },
                indent=2,
            )
        )
    else:
        print(
            f"\nPCA XGB CV AUC {best_cv:.4f} did not exceed threshold {beat_threshold:.4f} "
            f"(12-feat XGB ref) — not saving {XGB_PCA_CHECKPOINT.name}"
        )

    return {
        "best_cv_auc": best_cv,
        "train_auc": train_auc,
        "train_cv_gap": gap,
        "n_pca_components": n_components,
        "logreg_pca_cv_mean": log_pca_mean,
        "logreg_pca_cv_std": log_pca_std,
        "saved_checkpoint": saved_path,
        "best_params": dict(search.best_params_),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("pca", "--pca"):
        run_pca_xgb_tuning()
    else:
        run()
