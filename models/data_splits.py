"""
models/data_splits.py
Temporal train / val / test split for the UFC fight quality classifier.

Actual database date ranges (corrected from the original plan doc):
  train : Jan 2025 – Aug 2025   (event_date < val_cutoff)
  val   : Sep 2025 – Dec 2025   (val_cutoff <= event_date < test_cutoff)
  test  : Jan 2026 – present    (event_date >= test_cutoff)

Usage
-----
    from models.training import build_classification_dataset
    from models.data_splits import temporal_split
    from data.db import Database

    db = Database()
    X, y, meta = build_classification_dataset(db)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y, meta)
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default cutoffs matching the actual database date ranges.
VAL_CUTOFF  = "2025-09-01"
TEST_CUTOFF = "2026-01-01"


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into train / val / test by event_date.

    Parameters
    ----------
    X            : Feature matrix, shape (N, F). Produced by build_classification_dataset().
    y            : Binary label vector, shape (N,).
    meta         : Dict with "event_date" (ISO-string array, shape (N,)) and
                   "fight_id" (int array, shape (N,)).
                   Both are returned by build_classification_dataset().
    val_cutoff   : First date of the validation set (inclusive). ISO-8601 string.
                   Default: 2025-09-01.
    test_cutoff  : First date of the test set (inclusive). ISO-8601 string.
                   Default: 2026-01-01.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test  (six numpy arrays)

    Notes
    -----
    - Rows with a missing event_date are assigned to train (conservative — they
      can never leak into val/test this way).
    - Both (A, B) and (B, A) augmentation rows for a fight share the same
      event_date, so they land in the same split automatically.
    - An assertion checks max(train_dates) < min(val_dates) < min(test_dates).
    """
    dates = np.asarray(meta["event_date"])

    # Defensive: sort everything by date before splitting (M1 step 3).
    # build_classification_dataset() already orders by event_date ASC, but
    # this guarantees the invariant regardless of upstream changes.
    order = np.argsort(dates, kind="stable")
    X     = X[order]
    y     = y[order]
    dates = dates[order]

    # Rows without a date go to train.
    train_mask = (dates < val_cutoff) | (dates == "")
    val_mask   = (dates >= val_cutoff) & (dates < test_cutoff)
    test_mask  = dates >= test_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    _print_split_sizes(y_train, y_val, y_test, val_cutoff, test_cutoff)
    _assert_no_temporal_overlap(dates, train_mask, val_mask, test_mask)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_split_sizes(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_cutoff: str,
    test_cutoff: str,
) -> None:
    def _stats(arr):
        n = len(arr)
        pos = int(arr.sum())
        pct = f"{100.0 * pos / max(n, 1):.1f}%"
        return n, pos, pct

    tr_n, tr_pos, tr_pct = _stats(y_train)
    va_n, va_pos, va_pct = _stats(y_val)
    te_n, te_pos, te_pct = _stats(y_test)

    msg = (
        f"\n── Temporal Split ──────────────────────────────────────────\n"
        f"  Train  : {tr_n:>5} samples  |  bonus: {tr_pos} ({tr_pct})  [< {val_cutoff}]\n"
        f"  Val    : {va_n:>5} samples  |  bonus: {va_pos} ({va_pct})  [{val_cutoff} – {test_cutoff})\n"
        f"  Test   : {te_n:>5} samples  |  bonus: {te_pos} ({te_pct})  [>= {test_cutoff}]\n"
        f"────────────────────────────────────────────────────────────\n"
    )
    print(msg)
    logger.info(msg)


def _assert_no_temporal_overlap(
    dates: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """Raise AssertionError if any two splits share a date."""
    train_dates = dates[train_mask & (dates != "")]
    val_dates   = dates[val_mask   & (dates != "")]
    test_dates  = dates[test_mask  & (dates != "")]

    if len(train_dates) > 0 and len(val_dates) > 0:
        assert max(train_dates) < min(val_dates), (
            f"Temporal overlap detected: "
            f"max(train_dates)={max(train_dates)} >= min(val_dates)={min(val_dates)}"
        )

    if len(val_dates) > 0 and len(test_dates) > 0:
        assert max(val_dates) < min(test_dates), (
            f"Temporal overlap detected: "
            f"max(val_dates)={max(val_dates)} >= min(test_dates)={min(test_dates)}"
        )

    logger.info(
        "No temporal overlap — max(train)=%s  min(val)=%s  min(test)=%s",
        max(train_dates) if len(train_dates) else "N/A",
        min(val_dates)   if len(val_dates)   else "N/A",
        min(test_dates)  if len(test_dates)  else "N/A",
    )
