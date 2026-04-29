"""
Re-run RFECV feature selection on the distribution-stable feature subspace.

The current ``SELECTED_FEATURES`` (12-D RFECV-optimal) includes rolling features
that look stable in training (98.8% are constant fallback values — see
``rolling_features.report_history_depth``) but explode at test time, so SHAP
shows them at 0 and the model has nothing to learn from.

Approach: data-driven filter. Compute std on train and test (after the canonical
StandardScaler fit on train). If a feature's ``test_std / train_std`` ratio is
above ``MAX_STD_RATIO``, drop it — that's a feature that doesn't generalize.
This catches both ``*_roll_*`` and the rolling-derived non-roll features
(``recent_form``, ``fight_frequency``, ``variance_clash``, …) in one pass.
Then run RFECV on the surviving features.

Run:  OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 python -m models.feature_selection_clean
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from models.data_loader import get_canonical_splits
from models.feature_engineering import ALL_FEATURE_NAMES

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
MAX_STD_RATIO = 1.5  # drop features whose test std exceeds 1.5x train std


def main(db_path: str = "data/ufc_matchmaker.db") -> None:
    splits = get_canonical_splits(db_path, subset_features=False)  # all 115
    X_train_all = splits["X_train"]
    y_train = splits["y_train"]
    X_val_all = splits["X_val"]
    y_val = splits["y_val"]
    X_test_all = splits["X_test"]
    y_test = splits["y_test"]
    names = splits["feature_names"]

    # Data-driven cleaning: drop features with test/train std ratio above threshold.
    # Train std is ~1.0 by construction (StandardScaler fit on train); the ratio
    # therefore reduces to test_std for almost all features.
    train_std = X_train_all.std(axis=0)
    test_std = X_test_all.std(axis=0)
    ratios = test_std / np.maximum(train_std, 1e-6)
    keep_mask = ratios <= MAX_STD_RATIO
    clean_idx = [i for i, k in enumerate(keep_mask) if k]
    clean_names = [names[i] for i in clean_idx]
    dropped = [(names[i], float(ratios[i])) for i, k in enumerate(keep_mask) if not k]
    dropped.sort(key=lambda x: -x[1])
    print(f"Filtering: {len(names)} → {len(clean_names)} clean features "
          f"(dropped {len(dropped)} with test_std/train_std > {MAX_STD_RATIO})")
    if dropped:
        print(f"\nTop 10 dropped (highest std ratio):")
        for n, r in dropped[:10]:
            print(f"  {r:>6.2f}× {n}")

    X_train = X_train_all[:, clean_idx]
    X_val = X_val_all[:, clean_idx]
    X_test = X_test_all[:, clean_idx]

    cv = TimeSeriesSplit(n_splits=5)
    est = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    rfecv = RFECV(estimator=est, step=1, cv=cv, scoring="roc_auc",
                  min_features_to_select=2, n_jobs=1)
    print(f"\nRunning RFECV on {X_train.shape} (LogReg, TimeSeriesSplit-5)...")
    rfecv.fit(X_train, y_train)

    n_opt = int(rfecv.n_features_)
    selected = [clean_names[i] for i in np.where(rfecv.support_)[0]]
    rank = rfecv.ranking_

    print()
    print("=" * 60)
    print(f"  RFECV optimal n: {n_opt}")
    print(f"  CV AUC at optimum: {float(rfecv.cv_results_['mean_test_score'].max()):.4f}")
    print("=" * 60)
    print(f"\nSelected features (n={n_opt}):")
    for f in selected:
        print(f"  {f}")

    # Sanity-check: how does this subset do vs the current 12-feature subset?
    # Train a fresh LogReg on the new subset and compare val/test AUC.
    sel_idx = [clean_names.index(f) for f in selected]
    X_tr_sel = X_train[:, sel_idx]
    X_va_sel = X_val[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]

    est_eval = LogisticRegression(
        class_weight="balanced", max_iter=5000,
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    est_eval.fit(X_tr_sel, y_train)
    val_auc = roc_auc_score(y_val, est_eval.predict_proba(X_va_sel)[:, 1])
    test_auc = roc_auc_score(y_test, est_eval.predict_proba(X_te_sel)[:, 1])

    print()
    print("=" * 60)
    print(f"  LogReg on new subset:")
    print(f"    Val AUC:  {val_auc:.4f}")
    print(f"    Test AUC: {test_auc:.4f}")
    print("=" * 60)

    print("\n# Paste this into models/pipeline_config.py:")
    print("RFECV_OPTIMAL_CLEAN: tuple[str, ...] = (")
    for f in selected:
        print(f'    "{f}",')
    print(")")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
