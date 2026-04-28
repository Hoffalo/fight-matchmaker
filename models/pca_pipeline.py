# EXPERIMENTAL: PCA-based feature pipeline. Results showed RFECV features outperformed PCA.
# Kept for reference and presentation plots.
"""
PCA on the full 115-dim scaled matchup vector, then LogReg CV over component counts.

- Loads ``get_canonical_splits(..., subset_features=False)`` so StandardScaler is fit
  on raw 115-dim training rows only.
- PCA is fit on training data; for CV, uses ``Pipeline(PCA, LogisticRegression)``
  so each TimeSeriesSplit fold refits PCA on the fold train slice (no PCA leakage).
- Writes plots under ``outputs/plots/`` and checkpoints under ``models/checkpoints/``.

Run:  python -m models.pca_pipeline
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from config import BASE_DIR, MODELS_DIR
from models.data_loader import get_canonical_splits

logger = logging.getLogger(__name__)

OUTPUT_PLOTS = BASE_DIR / "outputs" / "plots"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
PCA_COMPONENT_GRID = (10, 15, 20, 25, 30, 40)
RANDOM_STATE = 42
RFECV12_LOGREG_REF = 0.5945
FULL115_LOGREG_REF = 0.5950


def _variance_threshold_n(cumvar: np.ndarray, target: float) -> int:
    """Smallest n such that cumulative variance >= target (1-indexed count)."""
    if len(cumvar) == 0:
        return 0
    idx = int(np.searchsorted(cumvar, target, side="left"))
    return min(idx + 1, len(cumvar))


def run(
    db_path: str = "data/ufc_matchmaker.db",
    component_grid: tuple[int, ...] = PCA_COMPONENT_GRID,
) -> dict:
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path, subset_features=False)
    X_train = np.asarray(splits["X_train"], dtype=np.float64)
    X_val = np.asarray(splits["X_val"], dtype=np.float64)
    X_test = np.asarray(splits["X_test"], dtype=np.float64)
    y_train = splits["y_train"].astype(np.int32)
    y_val = splits["y_val"].astype(np.int32)
    scaler_115 = splits["scaler"]
    n_feat = X_train.shape[1]

    print(f"\nPCA pipeline: train {X_train.shape}, features={n_feat} (scaled, train-fitted scaler)")

    # ── Full PCA for variance profile (train only) ───────────────────────
    n_pc_max = min(X_train.shape[0], X_train.shape[1])
    pca_full = PCA(n_components=n_pc_max, random_state=RANDOM_STATE, svd_solver="full")
    pca_full.fit(X_train)
    evr = np.asarray(pca_full.explained_variance_ratio_, dtype=np.float64)
    cumvar = np.cumsum(evr)

    print("\nCumulative variance thresholds (on training fit):")
    for target in (0.80, 0.85, 0.90, 0.95, 0.99):
        n = _variance_threshold_n(cumvar, target)
        actual = cumvar[n - 1] if n > 0 else 0.0
        print(f"  {target * 100:.0f}% variance: {n} components (cum = {100 * actual:.2f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "b-", label="Cumulative explained variance")
    ax.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    ax.set_xlabel("Number of components", fontsize=14)
    ax.set_ylabel("Cumulative explained variance", fontsize=14)
    ax.set_title(f"PCA — cumulative explained variance ({n_feat} features)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_ev = OUTPUT_PLOTS / "pca_explained_variance.png"
    fig.savefig(p_ev, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {p_ev}")

    # ── PCA transforms at fixed counts (fit on full train; for deploy / plots) ─
    results: dict[int, dict] = {}
    for n_components in component_grid:
        nc = min(n_components, n_pc_max)
        pca = PCA(n_components=nc, random_state=RANDOM_STATE, svd_solver="full")
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        ve = float(np.sum(pca.explained_variance_ratio_))
        results[n_components] = {
            "X_train": X_train_pca,
            "X_val": X_val_pca,
            "X_test": X_test_pca,
            "pca": pca,
            "variance_explained": ve,
        }
        print(f"  {n_components} components: {100 * ve:.1f}% variance retained")

    # ── LogReg CV: Pipeline refits PCA per fold ───────────────────────────
    cv = TimeSeriesSplit(n_splits=5)

    print("\nLogReg CV AUC by PCA component count (PCA refit per CV fold):")
    print(f"  {'Components':<12} {'Variance':>10} {'CV AUC':>18}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 18}")

    auc_by_n: dict[int, float] = {}
    std_by_n: dict[int, float] = {}

    for n_comp in component_grid:
        nc = min(n_comp, n_pc_max)
        pipe = Pipeline(
            [
                ("pca", PCA(n_components=nc, random_state=RANDOM_STATE, svd_solver="full")),
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
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1,
        )
        mean_auc, std_auc = float(np.mean(scores)), float(np.std(scores))
        auc_by_n[n_comp] = mean_auc
        std_by_n[n_comp] = std_auc
        ve = results[n_comp]["variance_explained"]
        print(f"  {n_comp:<12} {100 * ve:>9.1f}% {mean_auc:>10.4f} ± {std_auc:.4f}")

    pipe115 = Pipeline(
        [
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
    scores_115 = cross_val_score(
        pipe115, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1,
    )
    print(
        f"\n  Full {n_feat}-feat LogReg (same CV):     {np.mean(scores_115):.4f} ± {np.std(scores_115):.4f}"
    )
    print(f"  Full {n_feat}-feat LogReg (ref table):   {FULL115_LOGREG_REF:.4f}")
    print(f"  RFECV 12-feat LogReg (ref table):       {RFECV12_LOGREG_REF:.4f}")

    component_counts = list(component_grid)
    auc_scores = [auc_by_n[n] for n in component_counts]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(component_counts, auc_scores, "bo-", markersize=8)
    ax2.axhline(y=RFECV12_LOGREG_REF, color="r", linestyle="--", label="RFECV 12-feat baseline (ref)")
    ax2.set_xlabel("PCA components", fontsize=14)
    ax2.set_ylabel("CV ROC-AUC", fontsize=14)
    ax2.set_title("Model performance vs PCA components (LogReg)", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p_auc = OUTPUT_PLOTS / "pca_components_vs_auc.png"
    fig2.savefig(p_auc, dpi=300)
    plt.close(fig2)
    print(f"\nSaved: {p_auc}")

    best_n = max(auc_by_n, key=lambda k: auc_by_n[k])
    best_pca = results[best_n]["pca"]
    joblib.dump(best_pca, CHECKPOINT_DIR / "pca_transformer.pkl")
    joblib.dump(scaler_115, CHECKPOINT_DIR / "scaler_115dim.pkl")
    meta_path = CHECKPOINT_DIR / "pca_best_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "best_n_components": int(best_n),
                "best_logreg_cv_auc": float(auc_by_n[best_n]),
                "variance_explained_full_train": float(
                    results[best_n]["variance_explained"]
                ),
            },
            indent=2,
        )
    )
    print(f"\nBest component count (CV mean AUC): {best_n}  (AUC={auc_by_n[best_n]:.4f})")
    print(f"Saved: {CHECKPOINT_DIR / 'pca_transformer.pkl'}")
    print(f"Saved: {CHECKPOINT_DIR / 'scaler_115dim.pkl'}")
    print(f"Saved: {meta_path}")

    return {
        "cumvar": cumvar,
        "results": results,
        "auc_by_n": auc_by_n,
        "std_by_n": std_by_n,
        "best_n": best_n,
        "best_cv_auc": auc_by_n[best_n],
        "plots": {"explained_variance": str(p_ev), "components_vs_auc": str(p_auc)},
        "full_115_cv_mean": float(np.mean(scores_115)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
