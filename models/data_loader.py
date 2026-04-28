"""
models/data_loader.py
Canonical data loading for the UFC fight entertainment prediction pipeline.

Delegates to the split/augmentation infrastructure in data_splits.py (Mattheus's
M1/M2/M3 work) while using 115-dim feature vectors with matchup cross-features,
odds, context, and rolling fight_stats features (build_full_matchup_vector).

This module is the single entry point for classification models
(baselines.py, nn_binary.py).

Usage
-----
    from models.data_loader import load_real_data, get_canonical_splits

    # Default: RFECV subset from ``pipeline_config`` when set (e.g. 12-D):
    data = load_real_data()

    # Full 115-D scaled matrix (PCA / full-feat baselines); scaler fit on train only:
    data_full = get_canonical_splits(subset_features=False)
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import FEATURE_DIM
from models.feature_engineering import ALL_FEATURE_NAMES

assert len(ALL_FEATURE_NAMES) == FEATURE_DIM, (
    "ALL_FEATURE_NAMES length must match FEATURE_DIM in config.py"
)

logger = logging.getLogger(__name__)

N_FEATURES = FEATURE_DIM

# Match the actual DB date ranges (Jan 2025 – present).
# These mirror models/data_splits.py VAL_CUTOFF / TEST_CUTOFF.
TRAIN_CUTOFF = "2025-09-01"   # train: Jan 2025 – Aug 2025
VAL_CUTOFF   = "2026-01-01"   # val: Sep 2025 – Dec 2025; test: Jan 2026+


# ─────────────────────────────────────────────────────────────────────────────
# Core loader — uses partner's split infrastructure with 115-dim features
# ─────────────────────────────────────────────────────────────────────────────

def get_canonical_splits(
    db_path: str = "data/ufc_matchmaker.db",
    train_cutoff: str = TRAIN_CUTOFF,
    val_cutoff: str = VAL_CUTOFF,
    selected_features: list[str] | None = None,
    subset_features: bool = True,
) -> dict:
    """
    Canonical split pipeline using data_splits.py infrastructure + 115-dim features.

    Pipeline
    --------
    1. build_raw_pairs(db) → one row per fight, raw fighter dicts (+ rolling)
    2. temporal_split_raw() → split by date BEFORE augmentation
    3. augment_pair(vector_fn=build_full_matchup_vector) → 115-dim, both orderings
    4. assert_no_fight_id_leakage() → runtime safety check
    5. Optional column filter (``selected_features`` / ``pipeline_config``)
    6. Fit StandardScaler on train only

    Parameters
    ----------
    subset_features : bool, default True
        If True, may reduce columns using ``selected_features`` or
        ``pipeline_config.SELECTED_FEATURES`` (RFECV subset). If False, always keep
        all ``FEATURE_DIM`` (115) columns — use for PCA / full-feat baselines.
    selected_features : optional list of feature names (subset of ``ALL_FEATURE_NAMES``).
        If ``None``, uses non-empty ``models.pipeline_config.SELECTED_FEATURES`` when
        ``subset_features`` is True; if that is also unset/empty, all columns are kept.
        Ignored when ``subset_features`` is False (full vector).

    Returns
    -------
    dict with keys:
        X_train, y_train       — scaled training arrays (float32)
        X_val, y_val           — scaled validation arrays
        X_test, y_test         — scaled test arrays
        scaler                 — fitted StandardScaler
        feature_names          — list of names (length = number of columns kept)
        event_ids_test         — int array mapping test rows to events
        raw_train              — raw train dict (for k-fold CV downstream)
        summary                — dict with dataset statistics
    """
    from data.db import Database
    from models.data_splits import (
        build_raw_pairs,
        temporal_split_raw,
        augment_pair,
        assert_no_fight_id_leakage,
        build_full_matchup_vector,
    )

    db_path_resolved = str(Path(db_path).resolve()) if not Path(db_path).is_absolute() else db_path
    if not Path(db_path_resolved).exists():
        raise FileNotFoundError(
            f"Database not found at {db_path_resolved}. "
            "Run the data pipeline first: python main.py collect"
        )

    db = Database(db_path_resolved)

    effective_sel: list[str] | None = None
    if subset_features:
        if selected_features is not None and len(selected_features) > 0:
            effective_sel = list(selected_features)
        else:
            try:
                from models.pipeline_config import SELECTED_FEATURES as _pc_sel
                effective_sel = (
                    list(_pc_sel) if _pc_sel is not None and len(_pc_sel) > 0 else None
                )
            except ImportError:
                effective_sel = None
    elif selected_features is not None and len(selected_features) > 0:
        logger.warning(
            "subset_features=False: ignoring selected_features=%s (using all %d columns)",
            selected_features[:5],
            FEATURE_DIM,
        )

    # ── 1. Load raw pairs (one row per fight, no vectors yet) ────────────
    raw = build_raw_pairs(db)
    total_fights = len(raw["y"])

    if total_fights == 0:
        raise ValueError(
            "No fights with both fighter IDs found in DB. "
            "Run the data pipeline first: python main.py collect"
        )

    # ── 2. Temporal split on raw fights ──────────────────────────────────
    raw_train, raw_val, raw_test = temporal_split_raw(
        raw, val_cutoff=train_cutoff, test_cutoff=val_cutoff,
    )

    if len(raw_train["y"]) == 0:
        raise ValueError(f"No training fights before {train_cutoff}.")
    if len(raw_val["y"]) == 0:
        raise ValueError(f"No validation fights in [{train_cutoff}, {val_cutoff}).")
    if len(raw_test["y"]) == 0:
        raise ValueError(f"No test fights on/after {val_cutoff}.")

    # ── 3. Augment within each split (115-dim) ────────────────────────────
    vector_fn = build_full_matchup_vector

    X_train, y_train, meta_train = augment_pair(raw_train, vector_fn=vector_fn)
    X_val, y_val, meta_val       = augment_pair(raw_val, vector_fn=vector_fn)
    X_test, y_test, meta_test    = augment_pair(raw_test, vector_fn=vector_fn)

    # ── 4. Leakage assertion ─────────────────────────────────────────────
    assert_no_fight_id_leakage(meta_train, meta_val, meta_test)

    # ── 5. Handle NaNs ───────────────────────────────────────────────────
    nan_count = int(np.isnan(X_train).sum() + np.isnan(X_val).sum() + np.isnan(X_test).sum())
    if nan_count > 0:
        logger.warning("Found %d NaN values in features — replacing with 0.", nan_count)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val   = np.nan_to_num(X_val, nan=0.0)
        X_test  = np.nan_to_num(X_test, nan=0.0)

    feature_names_out = list(ALL_FEATURE_NAMES)
    if effective_sel is not None:
        missing = [n for n in effective_sel if n not in feature_names_out]
        if missing:
            raise ValueError(
                f"selected_features unknown or not in ALL_FEATURE_NAMES: {missing[:8]}..."
            )
        col_idx = [feature_names_out.index(n) for n in effective_sel]
        X_train = X_train[:, col_idx]
        X_val = X_val[:, col_idx]
        X_test = X_test[:, col_idx]
        feature_names_out = [feature_names_out[i] for i in col_idx]
        logger.info("Feature column filter: %d → %d columns", len(ALL_FEATURE_NAMES), len(feature_names_out))

    # ── 6. Fit scaler on train only ──────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # ── 7. Build event_ids for test set ──────────────────────────────────
    test_fight_ids = meta_test["fight_id"]
    unique_fights = np.unique(test_fight_ids)
    fight_to_event = {fid: i for i, fid in enumerate(unique_fights)}
    event_ids_test = np.array([fight_to_event[fid] for fid in test_fight_ids], dtype=np.int32)

    # ── 8. Summary ───────────────────────────────────────────────────────
    summary = {
        "total_unique_fights": total_fights,
        "feature_dim": len(feature_names_out),
        "feature_filter_active": effective_sel is not None,
        "train": {
            "fights": len(raw_train["y"]),
            "samples": len(y_train),
            "pos": int(y_train.sum()),
            "pos_pct": float(y_train.mean() * 100),
        },
        "val": {
            "fights": len(raw_val["y"]),
            "samples": len(y_val),
            "pos": int(y_val.sum()),
            "pos_pct": float(y_val.mean() * 100),
        },
        "test": {
            "fights": len(raw_test["y"]),
            "samples": len(y_test),
            "pos": int(y_test.sum()),
            "pos_pct": float(y_test.mean() * 100),
        },
        "nan_features_replaced": nan_count,
    }

    _print_summary(summary)

    return {
        "X_train": X_train_scaled, "y_train": y_train.astype(np.int32),
        "X_val": X_val_scaled, "y_val": y_val.astype(np.int32),
        "X_test": X_test_scaled, "y_test": y_test.astype(np.int32),
        "scaler": scaler,
        "feature_names": feature_names_out,
        "event_ids_test": event_ids_test,
        "raw_train": raw_train,
        "summary": summary,
    }


def load_real_data(
    db_path: str = "data/ufc_matchmaker.db",
    train_cutoff: str = TRAIN_CUTOFF,
    val_cutoff: str = VAL_CUTOFF,
    selected_features: list[str] | None = None,
    subset_features: bool = True,
) -> dict:
    """
    Load fight data from SQLite — delegates to get_canonical_splits().

    Backward-compatible entry point for baselines.py and nn_binary.py.
    """
    return get_canonical_splits(
        db_path=db_path,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        selected_features=selected_features,
        subset_features=subset_features,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict) -> None:
    """Print a clean data summary to stdout and logger."""
    lines = [
        "",
        "=" * 72,
        "  DATA SUMMARY — UFC Fight Entertainment Dataset (115-dim)",
        "=" * 72,
        f"  Total unique fights:  {summary['total_unique_fights']}",
        f"  Feature dimensions:   {summary['feature_dim']}",
        "",
    ]

    for split_name in ("train", "val", "test"):
        s = summary[split_name]
        lines.append(
            f"  {split_name.upper():<6}  "
            f"{s['fights']:>5} fights  ({s['samples']:>5} samples w/ augmentation)  "
            f"{s['pos']:>4} pos ({s['pos_pct']:.1f}%)"
        )

    lines.append("")

    if summary["nan_features_replaced"] > 0:
        lines.append(f"  WARNING: {summary['nan_features_replaced']} NaN values replaced with 0.")
    else:
        lines.append("  Data quality: no NaN values detected.")

    lines.append("=" * 72)

    output = "\n".join(lines)
    print(output)
    logger.info(
        "Data summary: %d fights, feature_dim=%d",
        summary["total_unique_fights"],
        summary["feature_dim"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/ufc_matchmaker.db"
    try:
        splits = get_canonical_splits(db_path)

        print(f"\nFeature dimension: {splits['X_train'].shape[1]}")
        print(f"Train: {splits['summary']['train']['fights']} unique → "
              f"{splits['summary']['train']['samples']} augmented")
        print(f"Val:   {splits['summary']['val']['fights']} unique → "
              f"{splits['summary']['val']['samples']} augmented")
        print(f"Test:  {splits['summary']['test']['fights']} unique → "
              f"{splits['summary']['test']['samples']} augmented")
        print(f"Train positive rate: {splits['y_train'].mean():.3f}")
        print(f"Val positive rate:   {splits['y_val'].mean():.3f}")
        print(f"Test positive rate:  {splits['y_test'].mean():.3f}")
        assert splits["X_train"].shape[1] == len(splits["feature_names"]), "Feature dim mismatch"
        print("\nAll checks passed.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("To generate a test DB, run: python main.py collect")
        sys.exit(1)
