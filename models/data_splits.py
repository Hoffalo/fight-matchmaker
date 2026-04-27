"""
models/data_splits.py
Temporal train / val / test split for the UFC fight quality classifier.

M1 — temporal split.
M2 — leak-safe ordering: split raw fights by date FIRST, then augment
     (A, B) / (B, A) within each split so both orderings of a given fight
     can never land in different splits.

Actual database date ranges (corrected from the original plan doc):
  train : Jan 2025 – Aug 2025   (event_date < val_cutoff)
  val   : Sep 2025 – Dec 2025   (val_cutoff <= event_date < test_cutoff)
  test  : Jan 2026 – present    (event_date >= test_cutoff)

Usage
-----
    from data.db import Database
    from models.data_splits import build_raw_pairs, temporal_split

    raw = build_raw_pairs(Database())
    X_tr, y_tr, X_va, y_va, X_te, y_te, m_tr, m_va, m_te = temporal_split(raw)
"""
import importlib.util as _ilu
import logging
import os as _os
import sys as _sys

import numpy as np

logger = logging.getLogger(__name__)

# Default cutoffs matching the actual database date ranges.
VAL_CUTOFF  = "2025-09-01"
TEST_CUTOFF = "2026-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# Sibling import that bypasses models/__init__.py
# (lets this module load in environments without torch installed yet)
# ─────────────────────────────────────────────────────────────────────────────

def _load_sibling(name: str):
    cache_key = f"_data_splits_sibling_{name}"
    if cache_key in _sys.modules:
        return _sys.modules[cache_key]
    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = _ilu.spec_from_file_location(cache_key, _os.path.join(here, f"{name}.py"))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[cache_key] = mod
    spec.loader.exec_module(mod)
    return mod


build_matchup_vector = _load_sibling("feature_engineering").build_matchup_vector


# ─────────────────────────────────────────────────────────────────────────────
# Raw pair loader  (one row per fight — no augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_pairs(db) -> dict:
    """
    Load one row per fight from the DB. No (A, B) / (B, A) augmentation here —
    that happens AFTER splitting (see temporal_split).

    Returns
    -------
    dict with:
      f1         : list of fighter dicts, length M (= number of fights)
      f2         : list of fighter dicts, length M
      y          : float32 array (M,) — bonus label
      fight_id   : int64 array (M,)  — DB primary key (M2 step 1)
      event_date : np.ndarray of ISO strings (M,)
    """
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            """
            SELECT f.id           AS fight_id,
                   f.fighter1_id,
                   f.fighter2_id,
                   f.is_bonus_fight,
                   e.date         AS event_date
            FROM fights f
            LEFT JOIN events e ON f.event_id = e.id
            WHERE f.fighter1_id IS NOT NULL AND f.fighter2_id IS NOT NULL
            ORDER BY e.date ASC
            """
        ).fetchall()]
        fighters = {r["id"]: dict(r) for r in conn.execute(
            "SELECT * FROM fighters"
        ).fetchall()}

    f1_list, f2_list, y_list, fid_list, date_list = [], [], [], [], []
    skipped = 0
    for fight in rows:
        f1 = fighters.get(fight["fighter1_id"])
        f2 = fighters.get(fight["fighter2_id"])
        if f1 is None or f2 is None:
            skipped += 1
            continue
        f1_list.append(f1)
        f2_list.append(f2)
        y_list.append(float(fight["is_bonus_fight"] or 0))
        fid_list.append(fight["fight_id"])
        date_list.append(fight["event_date"] or "")

    if skipped:
        logger.warning("Skipped %d fights with missing fighter profiles", skipped)
    logger.info("Loaded %d raw fight pairs (one row per fight)", len(f1_list))

    return {
        "f1":         f1_list,
        "f2":         f2_list,
        "y":          np.array(y_list,  dtype=np.float32),
        "fight_id":   np.array(fid_list, dtype=np.int64),
        "event_date": np.array(date_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation  ((A, B) and (B, A) both share the same fight_id)
# ─────────────────────────────────────────────────────────────────────────────

def augment_pair(raw: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Expand one row per fight into two rows: (A, B) and (B, A).
    Both rows share the same fight_id and event_date — so any group-aware
    splitter that respects fight_id keeps them together.

    M2 step 2: both orderings share a fight_id.
    """
    X, y, fids, dates = [], [], [], []
    f1_arr, f2_arr = raw["f1"], raw["f2"]
    y_arr, fid_arr, date_arr = raw["y"], raw["fight_id"], raw["event_date"]

    for i in range(len(f1_arr)):
        f1, f2 = f1_arr[i], f2_arr[i]
        label  = float(y_arr[i])
        fid    = int(fid_arr[i])
        date   = date_arr[i]
        for vec in (build_matchup_vector(f1, f2), build_matchup_vector(f2, f1)):
            X.append(vec)
            y.append(label)
            fids.append(fid)
            dates.append(date)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        {
            "fight_id":   np.array(fids,  dtype=np.int64),
            "event_date": np.array(dates),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split  (split first, augment within each split — M2 step 4)
# ─────────────────────────────────────────────────────────────────────────────

def _select_raw(raw: dict, idx) -> dict:
    """Index into a raw dict, preserving the same shape."""
    idx = np.asarray(idx)
    return {
        "f1":         [raw["f1"][i] for i in idx],
        "f2":         [raw["f2"][i] for i in idx],
        "y":          raw["y"][idx],
        "fight_id":   raw["fight_id"][idx],
        "event_date": np.asarray(raw["event_date"])[idx],
    }


def temporal_split_raw(
    raw: dict,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> tuple[dict, dict, dict]:
    """
    Split raw fights by event_date — no augmentation. Use this when you
    want to do further splitting on the train portion (k-fold CV) without
    inflating fold sizes via augmentation.

    Returns (raw_train, raw_val, raw_test).
    """
    dates = np.asarray(raw["event_date"])

    # M1 step 3: sort by event_date.
    order = np.argsort(dates, kind="stable")
    raw_sorted = _select_raw(raw, order)
    dates_sorted = raw_sorted["event_date"]

    # Rows without a date go to train.
    train_mask = (dates_sorted < val_cutoff) | (dates_sorted == "")
    val_mask   = (dates_sorted >= val_cutoff) & (dates_sorted < test_cutoff)
    test_mask  = dates_sorted >= test_cutoff

    raw_train = _select_raw(raw_sorted, np.where(train_mask)[0])
    raw_val   = _select_raw(raw_sorted, np.where(val_mask)[0])
    raw_test  = _select_raw(raw_sorted, np.where(test_mask)[0])

    _assert_no_temporal_overlap(dates_sorted, train_mask, val_mask, test_mask)
    return raw_train, raw_val, raw_test


def temporal_split(
    raw: dict,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> tuple:
    """
    Split raw fights by event_date and augment WITHIN each split.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    meta_train, meta_val, meta_test

    Guarantees (asserted at runtime)
    --------------------------------
    - max(train_dates) < min(val_dates) < min(test_dates)   (M1 step 6)
    - no fight_id appears in more than one split            (M2 step 5)
    """
    raw_train, raw_val, raw_test = temporal_split_raw(raw, val_cutoff, test_cutoff)

    # M2 step 4: augment WITHIN each split, never across.
    X_tr, y_tr, meta_tr = augment_pair(raw_train)
    X_va, y_va, meta_va = augment_pair(raw_val)
    X_te, y_te, meta_te = augment_pair(raw_test)

    _print_split_sizes(
        raw_train["y"], raw_val["y"], raw_test["y"],
        y_tr, y_va, y_te,
        val_cutoff, test_cutoff,
    )
    assert_no_fight_id_leakage(meta_tr, meta_va, meta_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te, meta_tr, meta_va, meta_te


# ─────────────────────────────────────────────────────────────────────────────
# Verification helpers
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_fight_id_leakage(meta_train: dict, meta_val: dict, meta_test: dict) -> None:
    """
    M2 step 5: verify no fight_id appears in more than one split.

    Raises AssertionError listing the leaking fight_ids if any.
    """
    train_ids = set(meta_train["fight_id"].tolist())
    val_ids   = set(meta_val["fight_id"].tolist())
    test_ids  = set(meta_test["fight_id"].tolist())

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    assert not tv, f"Leakage: {len(tv)} fight_ids in train ∩ val: {sorted(tv)[:5]}"
    assert not tt, f"Leakage: {len(tt)} fight_ids in train ∩ test: {sorted(tt)[:5]}"
    assert not vt, f"Leakage: {len(vt)} fight_ids in val ∩ test: {sorted(vt)[:5]}"

    logger.info(
        "No fight_id leakage  (train=%d, val=%d, test=%d unique fights)",
        len(train_ids), len(val_ids), len(test_ids),
    )


def _print_split_sizes(
    raw_y_tr: np.ndarray, raw_y_va: np.ndarray, raw_y_te: np.ndarray,
    y_tr: np.ndarray,     y_va: np.ndarray,     y_te: np.ndarray,
    val_cutoff: str, test_cutoff: str,
) -> None:
    def _stats(raw_y, aug_y):
        n_fights = len(raw_y)
        n_rows   = len(aug_y)
        pos      = int(raw_y.sum())
        pct      = f"{100.0 * pos / max(n_fights, 1):.1f}%"
        return n_fights, n_rows, pos, pct

    tr_f, tr_r, tr_p, tr_pc = _stats(raw_y_tr, y_tr)
    va_f, va_r, va_p, va_pc = _stats(raw_y_va, y_va)
    te_f, te_r, te_p, te_pc = _stats(raw_y_te, y_te)

    msg = (
        f"\n── Temporal Split  (split → augment) ───────────────────────────────\n"
        f"  Train  : {tr_f:>4} fights  → {tr_r:>4} rows  |  bonus: {tr_p} ({tr_pc})  [< {val_cutoff}]\n"
        f"  Val    : {va_f:>4} fights  → {va_r:>4} rows  |  bonus: {va_p} ({va_pc})  [{val_cutoff} – {test_cutoff})\n"
        f"  Test   : {te_f:>4} fights  → {te_r:>4} rows  |  bonus: {te_p} ({te_pc})  [>= {test_cutoff}]\n"
        f"────────────────────────────────────────────────────────────────────\n"
    )
    print(msg)
    logger.info(msg)


def _assert_no_temporal_overlap(
    dates: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """M1 step 6: max(train_dates) < min(val_dates) < min(test_dates)."""
    train_dates = dates[train_mask & (dates != "")]
    val_dates   = dates[val_mask   & (dates != "")]
    test_dates  = dates[test_mask  & (dates != "")]

    if len(train_dates) > 0 and len(val_dates) > 0:
        assert max(train_dates) < min(val_dates), (
            f"Temporal overlap: max(train)={max(train_dates)} >= min(val)={min(val_dates)}"
        )

    if len(val_dates) > 0 and len(test_dates) > 0:
        assert max(val_dates) < min(test_dates), (
            f"Temporal overlap: max(val)={max(val_dates)} >= min(test)={min(test_dates)}"
        )

    logger.info(
        "No temporal overlap  (max(train)=%s  min(val)=%s  min(test)=%s)",
        max(train_dates) if len(train_dates) else "N/A",
        min(val_dates)   if len(val_dates)   else "N/A",
        min(test_dates)  if len(test_dates)  else "N/A",
    )


# ─────────────────────────────────────────────────────────────────────────────
# M3 — k-fold cross-validation on the raw (one-row-per-fight) train set
# ─────────────────────────────────────────────────────────────────────────────

# How many positives a val fold needs before we trust TimeSeriesSplit.
# Below this, we fall back to StratifiedKFold (M3 step 5).
_MIN_MINORITY_PER_FOLD = 2


def kfold_indices(
    raw: dict,
    n_splits: int = 5,
    strategy: str = "auto",
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """
    Generate k-fold (train_idx, val_idx) pairs over RAW fights.

    Parameters
    ----------
    raw          : output of build_raw_pairs() OR a raw split (e.g. raw_train
                   from temporal_split_raw). Indices are positional into raw.
    n_splits     : number of folds (default 5, per M3 step 1).
    strategy     : "auto" | "timeseries" | "stratified"
                   "auto" tries TimeSeriesSplit; if any val fold has fewer
                   than _MIN_MINORITY_PER_FOLD positives, falls back to
                   StratifiedKFold (M3 step 5).
    random_state : 42 (sprint reproducibility convention).

    Returns
    -------
    folds          : list of (train_idx, val_idx) numpy arrays
    strategy_used  : "timeseries" or "stratified"
    """
    from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold  # lazy

    n = len(raw["y"])
    if n < n_splits + 1:
        raise ValueError(f"Not enough samples ({n}) for {n_splits}-fold CV")

    y = raw["y"]
    indices = np.arange(n)

    if strategy in ("auto", "timeseries"):
        # TimeSeriesSplit assumes data is already sorted by time. The raw
        # dict from build_raw_pairs / temporal_split_raw is sorted by date.
        ts = TimeSeriesSplit(n_splits=n_splits)
        folds = [(tr, va) for tr, va in ts.split(indices)]

        ok = all(int(y[va].sum()) >= _MIN_MINORITY_PER_FOLD for _, va in folds)
        if ok or strategy == "timeseries":
            if not ok:
                logger.warning(
                    "TimeSeriesSplit folds have <%d positives but strategy "
                    "was forced to 'timeseries'. Metrics may be unstable.",
                    _MIN_MINORITY_PER_FOLD,
                )
            return folds, "timeseries"
        logger.info(
            "TimeSeriesSplit produced folds with too-few positives — "
            "falling back to StratifiedKFold (M3 step 5)."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [(tr, va) for tr, va in skf.split(indices, y)]
    return folds, "stratified"


def kfold_split(
    raw: dict,
    n_splits: int = 5,
    strategy: str = "auto",
    random_state: int = 42,
):
    """
    Generator yielding per-fold augmented arrays.

    Yields
    ------
    fold_i, X_train, y_train, X_val, y_val, meta_train, meta_val

    Augmentation happens INSIDE each fold (M2 leakage fix carried into k-fold).
    Asserts no fight_id leakage within each fold.
    """
    folds, strategy_used = kfold_indices(raw, n_splits, strategy, random_state)
    logger.info("k-fold strategy=%s  n_splits=%d", strategy_used, len(folds))

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        raw_tr = _select_raw(raw, train_idx)
        raw_va = _select_raw(raw, val_idx)

        X_tr, y_tr, m_tr = augment_pair(raw_tr)
        X_va, y_va, m_va = augment_pair(raw_va)

        # Per-fold leakage check (M2 step 5 applied to CV).
        train_ids = set(m_tr["fight_id"].tolist())
        val_ids   = set(m_va["fight_id"].tolist())
        leak = train_ids & val_ids
        assert not leak, f"Fold {fold_i}: {len(leak)} fight_ids leaked across train/val"

        yield fold_i, X_tr, y_tr, X_va, y_va, m_tr, m_va


def cv_score_sklearn(
    estimator_factory,
    raw: dict,
    n_splits: int = 5,
    strategy: str = "auto",
    random_state: int = 42,
    metrics: tuple = ("f1", "accuracy", "roc_auc"),
) -> dict:
    """
    Run k-fold CV with a fresh sklearn estimator per fold.

    Fits a StandardScaler on each fold's train portion only (no leakage).
    Reports mean ± std across folds (M3 step 4).

    Parameters
    ----------
    estimator_factory : callable() -> sklearn estimator. Called once per fold
                        so each fold gets a fresh model.
    raw               : raw dict (typically raw_train from temporal_split_raw).
    metrics           : subset of {"f1", "accuracy", "roc_auc"}.

    Returns
    -------
    dict mapping each metric name to {"mean": float, "std": float, "folds": list[float]}
    plus "_strategy" key recording which splitter was used.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    metric_fns = {
        "f1":       lambda y, p, _: f1_score(y, p, zero_division=0),
        "accuracy": lambda y, p, _: accuracy_score(y, p),
        "roc_auc":  lambda y, _, proba: roc_auc_score(y, proba) if proba is not None else float("nan"),
    }

    results: dict[str, list[float]] = {m: [] for m in metrics}
    strategy_used = None

    folds, strategy_used = kfold_indices(raw, n_splits, strategy, random_state)
    logger.info("CV strategy=%s  n_splits=%d", strategy_used, len(folds))

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        raw_tr = _select_raw(raw, train_idx)
        raw_va = _select_raw(raw, val_idx)

        X_tr, y_tr, m_tr = augment_pair(raw_tr)
        X_va, y_va, m_va = augment_pair(raw_va)

        leak = set(m_tr["fight_id"].tolist()) & set(m_va["fight_id"].tolist())
        assert not leak, f"Fold {fold_i}: {len(leak)} fight_ids leaked across train/val"

        # Scaler fit on train only (doc reminder).
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf = estimator_factory()
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_va_s)
        y_proba = (
            clf.predict_proba(X_va_s)[:, 1]
            if hasattr(clf, "predict_proba") else None
        )

        for m in metrics:
            results[m].append(float(metric_fns[m](y_va, y_pred, y_proba)))

    summary: dict = {"_strategy": strategy_used, "_n_splits": len(folds)}
    for m, scores in results.items():
        if scores:
            summary[m] = {
                "mean":  float(np.mean(scores)),
                "std":   float(np.std(scores)),
                "folds": scores,
            }
    return summary


def format_cv_report(summary: dict, model_name: str = "model") -> str:
    """Format a cv_score_sklearn summary for logs / slides."""
    lines = [
        f"\n── CV report: {model_name}  "
        f"(strategy={summary['_strategy']}, k={summary['_n_splits']}) ──",
    ]
    for m, stats in summary.items():
        if m.startswith("_"):
            continue
        folds_str = ", ".join(f"{s:.3f}" for s in stats["folds"])
        lines.append(
            f"  {m:<10s}  {stats['mean']:.3f} ± {stats['std']:.3f}   "
            f"folds=[{folds_str}]"
        )
    lines.append("")
    return "\n".join(lines)
