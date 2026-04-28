"""
models/feature_selection.py
Feature selection on the full 115-dim matchup vector (career + cross + odds + context + rolling).

Training slice: ~338 unique fights (≈26.9% positive before augmentation). Compares mutual
information, RFECV (LogReg), L1 LogReg, and XGBoost importance (``scale_pos_weight`` from
``config.SCALE_POS_WEIGHT``, ≈ 2.71).

Subset names can be copied into ``models.pipeline_config.SELECTED_FEATURES`` for
``get_canonical_splits()`` (recommended over the full 115 columns for small-n).

Run:  OMP_NUM_THREADS=1 python -m models.feature_selection
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import BASE_DIR, FEATURE_DIM, SCALE_POS_WEIGHT
from data.db import Database
from models.data_splits import augment_pair, build_raw_pairs, temporal_split_raw
from models.feature_engineering import ALL_FEATURE_NAMES, build_full_matchup_vector

logger = logging.getLogger(__name__)

OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
TOP_K = 30


def _ensure_xy_scaler():
    """Load DB, temporal split, augment with 115-dim vectors; scale train-only."""
    db = Database()
    raw = build_raw_pairs(db)
    raw_train, raw_val, _ = temporal_split_raw(raw)
    X_tr, y_tr, _ = augment_pair(raw_train, vector_fn=build_full_matchup_vector)
    X_va, y_va, _ = augment_pair(raw_val, vector_fn=build_full_matchup_vector)

    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    names = list(ALL_FEATURE_NAMES)
    return (
        X_tr_s.astype(np.float32),
        y_tr.astype(np.int32),
        X_va_s.astype(np.float32),
        y_va.astype(np.int32),
        names,
    )


def _mi_top30_from_scores(mi: np.ndarray, feature_names: list[str]) -> list[str]:
    order = np.argsort(-mi)
    top_idx = order[:TOP_K]
    return [feature_names[i] for i in top_idx]


def _plot_mi_horizontal(mi: np.ndarray, feature_names: list[str], path: Path) -> None:
    order = np.argsort(mi)
    top = order[-TOP_K:]
    vals = mi[top]
    labels = [feature_names[i] for i in top]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(vals)), vals, color="steelblue")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mutual information")
    ax.set_title(f"Top {TOP_K} features by mutual information ({FEATURE_DIM}-dim vector)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _rfecv_run(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> tuple[int, list[str], list[str]]:
    est = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    cv = TimeSeriesSplit(n_splits=5)
    rfecv = RFECV(
        estimator=est,
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=5,
        n_jobs=1,
    )
    rfecv.fit(X, y)
    n_opt = int(rfecv.n_features_)
    selected_mask = rfecv.support_
    selected_names = [feature_names[i] for i in np.where(selected_mask)[0]]
    rank = rfecv.ranking_
    best_order = np.argsort(rank)
    top30_idx = best_order[:TOP_K]
    top30_names = [feature_names[i] for i in top30_idx]
    return n_opt, selected_names, top30_names


def _l1_top30(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[list[str], list[str], float]:
    """
    L1 LogReg. Starts at C=0.01 as requested; increases C if all coefficients vanish on scaled data.
    Returns (surviving feature names, top-30 by |coef|, C used).
    """
    surviving: list[str] = []
    clf: LogisticRegression | None = None
    c_used = 0.01
    for C in (0.01, 0.03, 0.05, 0.1, 0.3, 1.0):
        c_used = C
        clf = LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            class_weight="balanced",
            max_iter=8000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        clf.fit(X, y)
        coef = clf.coef_.ravel()
        nz = np.abs(coef) > 1e-8
        if nz.any():
            surviving = [feature_names[i] for i in np.where(nz)[0]]
            break
    assert clf is not None
    coef = clf.coef_.ravel()
    order = np.argsort(-np.abs(coef))
    top30 = [feature_names[i] for i in order[:TOP_K]]
    return surviving, top30, float(c_used)


def _xgb_top30(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    clf = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    clf.fit(X, y)
    imp = clf.feature_importances_
    order = np.argsort(-imp)
    top_idx = order[:TOP_K]
    return imp, [feature_names[i] for i in top_idx]


def _plot_xgb_importance(imp: np.ndarray, feature_names: list[str], path: Path) -> None:
    order = np.argsort(imp)
    top = order[-TOP_K:]
    vals = imp[top]
    labels = [feature_names[i] for i in top]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(vals)), vals, color="darkorange")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title(f"Top {TOP_K} features by XGBoost importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _cv_auc_logreg(X: np.ndarray, y: np.ndarray) -> float:
    est = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    cv = TimeSeriesSplit(n_splits=5)
    m = cross_val_score(est, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(np.mean(m))


def _cv_auc_xgb(X: np.ndarray, y: np.ndarray) -> float:
    est = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    cv = TimeSeriesSplit(n_splits=5)
    m = cross_val_score(est, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(np.mean(m))


def _subset_cols(X: np.ndarray, feature_names: list[str], keep: list[str]) -> np.ndarray:
    idx = [feature_names.index(n) for n in keep]
    return X[:, idx]


def _plot_feature_count_vs_auc(
    points: list[tuple[int, float, float, str]],
    path: Path,
) -> None:
    xs = [p[0] for p in points]
    lr = [p[1] for p in points]
    xg = [p[2] for p in points]
    labels = [p[3] for p in points]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, lr, "o-", label="LogReg CV AUC", color="steelblue")
    ax.plot(xs, xg, "s-", label="XGBoost CV AUC", color="darkorange")
    for x, v, lb in zip(xs, lr, labels):
        ax.annotate(lb, (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Mean ROC-AUC (TimeSeriesSplit 5)")
    ax.set_title("Feature count vs CV AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _consensus_sets(
    top_mi: list[str],
    top_rfe: list[str],
    top_l1: list[str],
    top_xgb: list[str],
) -> list[tuple[str, int]]:
    votes: Counter[str] = Counter()
    for group in (top_mi, top_rfe, top_l1, top_xgb):
        for name in group:
            votes[name] += 1
    consensus = [(n, c) for n, c in votes.items() if c >= 2]
    consensus.sort(key=lambda t: (-t[1], t[0]))
    return consensus


def run() -> dict:
    import os

    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_va, y_va, names = _ensure_xy_scaler()
    assert X_tr.shape[1] == FEATURE_DIM == len(names)

    n_tr = len(y_tr)
    pos_tr = int(y_tr.sum())
    print(
        f"Train augmented: n={n_tr}, pos={pos_tr} ({100.0 * pos_tr / max(n_tr, 1):.1f}%)  "
        f"(unique fights ~338 → scale_pos_weight={SCALE_POS_WEIGHT:.3f})"
    )

    mi = mutual_info_classif(
        X_tr, y_tr, random_state=RANDOM_STATE, discrete_features=False
    )
    top_mi_names = _mi_top30_from_scores(mi, names)
    _plot_mi_horizontal(mi, names, OUTPUTS_DIR / "mutual_information_ranking.png")

    n_opt, selected_rfe, top_rfe_names = _rfecv_run(X_tr, y_tr, names)
    print(f"Optimal feature count: {n_opt}, Selected features: {selected_rfe}")

    surviving_l1, top_l1_names, l1_c = _l1_top30(X_tr, y_tr, names)
    print(
        f"L1 (saga, C tuned from 0.01; used C={l1_c}) surviving ({len(surviving_l1)}): {surviving_l1}"
    )

    imp, top_xgb_names = _xgb_top30(X_tr, y_tr, names)
    _plot_xgb_importance(imp, names, OUTPUTS_DIR / "xgb_feature_importance.png")

    consensus = _consensus_sets(top_mi_names, top_rfe_names, top_l1_names, top_xgb_names)
    consensus_names = [n for n, _ in consensus]
    print("\nConsensus (≥2 of 4 methods in top-30):")
    for n, c in consensus:
        print(f"  {c}/4  {n}")

    auc_full_lr = _cv_auc_logreg(X_tr, y_tr)
    auc_full_xgb = _cv_auc_xgb(X_tr, y_tr)

    auc_red_lr = float("nan")
    auc_red_xgb = float("nan")
    va_full = float("nan")
    va_red = float("nan")

    if consensus_names:
        Xc_tr = _subset_cols(X_tr, names, consensus_names)
        auc_red_lr = _cv_auc_logreg(Xc_tr, y_tr)
        auc_red_xgb = _cv_auc_xgb(Xc_tr, y_tr)
        lr_f = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ).fit(X_tr, y_tr)
        lr_r = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ).fit(Xc_tr, y_tr)
        va_full = float(roc_auc_score(y_va, lr_f.predict_proba(X_va)[:, 1]))
        va_red = float(
            roc_auc_score(
                y_va,
                lr_r.predict_proba(_subset_cols(X_va, names, consensus_names))[:, 1],
            )
        )

    print(f'\n115 features: AUC = {auc_full_lr:.4f}  (LogReg CV; ref held-out AUC ~0.606)')
    if consensus_names:
        print(f"Reduced ({len(consensus_names)} features): AUC = {auc_red_lr:.4f}")
        print(f"Delta: {auc_red_lr - auc_full_lr:+.4f}")
        print(
            f"(XGBoost CV — 115: {auc_full_xgb:.4f}, reduced: {auc_red_xgb:.4f}; "
            f"val LogReg — 115: {va_full:.4f}, reduced: {va_red:.4f})"
        )

    mi_order = list(np.argsort(-mi))
    cross_names_only = [names[i] for i in range(48, 72)]

    curve_points: list[tuple[int, float, float, str]] = []
    specs = [
        (10, [names[i] for i in mi_order[:10]], "top10_MI"),
        (20, [names[i] for i in mi_order[:20]], "top20_MI"),
        (30, [names[i] for i in mi_order[:30]], "top30_MI"),
        (24, cross_names_only, "cross_only"),
        (n_opt, list(dict.fromkeys(selected_rfe)), "RFECV_opt"),
        (115, list(names), "all115"),
    ]
    seen_n: set[int] = set()
    for n_label, keep, tag in specs:
        if n_label in seen_n:
            continue
        seen_n.add(n_label)
        Xm = _subset_cols(X_tr, names, keep)
        lr_m = _cv_auc_logreg(Xm, y_tr)
        xgb_m = _cv_auc_xgb(Xm, y_tr)
        curve_points.append((n_label, lr_m, xgb_m, tag))
        print(f"  [{tag}] n={n_label}: LogReg CV AUC={lr_m:.4f}, XGB CV AUC={xgb_m:.4f}")

    curve_points.sort(key=lambda p: p[0])
    _plot_feature_count_vs_auc(curve_points, OUTPUTS_DIR / "feature_count_vs_auc.png")

    X_cross = _subset_cols(X_tr, names, cross_names_only)
    auc_cross_lr = _cv_auc_logreg(X_cross, y_tr)
    auc_cross_xgb = _cv_auc_xgb(X_cross, y_tr)
    print(
        f"\nCross-features ONLY (indices 48–71, 24 feats): "
        f"LogReg CV AUC={auc_cross_lr:.4f}, XGB CV AUC={auc_cross_xgb:.4f}"
    )
    print(f"  vs full 115 LogReg CV={auc_full_lr:.4f}")

    candidates: list[tuple[str, list[str]]] = [
        ("consensus", consensus_names),
        ("top10_mi", [names[i] for i in mi_order[:10]]),
        ("top20_mi", [names[i] for i in mi_order[:20]]),
        ("top30_mi", [names[i] for i in mi_order[:30]]),
    ]
    best_name, best_list, best_auc = "all115", list(names), auc_full_lr
    for label, feats in candidates:
        if not feats:
            continue
        a = _cv_auc_logreg(_subset_cols(X_tr, names, feats), y_tr)
        if a > best_auc:
            best_auc, best_name, best_list = a, label, feats

    if best_list and best_name != "all115":
        print(
            "\nEnable subset in training — set in models/pipeline_config.py, e.g.:\n"
            f"  SELECTED_FEATURES = {best_list!r}\n"
            "(Sync RECOMMENDED_SUBSET_MI_TOP30 / add a 115-dim RECOMMENDED_* copy.)"
        )
    else:
        print("\nBest CV AUC was full 115 features; leave SELECTED_FEATURES = None.")

    return {
        "consensus": consensus_names,
        "best_subset_name": best_name,
        "best_subset_auc_cv": best_auc,
        "auc_full_115_lr_cv": auc_full_lr,
        "cross_only_lr_cv": auc_cross_lr,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()