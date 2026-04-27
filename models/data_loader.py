"""
models/data_loader.py
Canonical data loading for the UFC fight entertainment prediction pipeline.

Reads real fight data from the SQLite DB, builds 72-dim feature vectors,
applies a temporal train/val/test split, and performs symmetric augmentation
AFTER splitting so both orderings of each fight stay in the same fold.

Schema assumptions (from data/db.py):
  - fights table:    id, fighter1_id, fighter2_id, event_id, is_bonus_fight
  - events table:    id, date
  - fighters table:  all career stats consumed by feature_engineering.py
  - is_bonus_fight:  populated by scrapers/wikipedia_bonus_scraper.py
                     and db.refresh_bonus_labels()

Usage
-----
    from models.data_loader import load_real_data

    data = load_real_data("data/ufc_matchmaker.db")
    # data["X_train"], data["y_train"], ... data["scaler"], data["feature_names"]

    # Or pass straight into BaselineComparison:
    bc = BaselineComparison()
    bc.load_data(data)
"""
import logging
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from models.feature_engineering import (
    ALL_FEATURE_NAMES,
    build_full_matchup_vector,
    extract_fighter_features,
)

logger = logging.getLogger(__name__)

N_FEATURES = 72

# Match the actual DB date ranges (Jan 2025 – present).
# These mirror models/data_splits.py for consistency.
TRAIN_CUTOFF = "2025-09-01"   # train: Jan 2025 – Aug 2025
VAL_CUTOFF   = "2026-01-01"   # val: Sep 2025 – Dec 2025; test: Jan 2026+


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(
    db_path: str = "data/ufc_matchmaker.db",
    train_cutoff: str = TRAIN_CUTOFF,
    val_cutoff: str = VAL_CUTOFF,
) -> dict:
    """
    Load fight data from the real SQLite DB and prepare train/val/test splits.

    Pipeline
    --------
    1. Query all fights with is_bonus_fight labels + event dates
    2. Build 72-dim feature vectors (single ordering per fight)
    3. Temporal split by event date
    4. Symmetric augmentation WITHIN each split — both (A,B) and (B,A)
       orderings stay in the same fold, preventing data leakage
    5. Fit StandardScaler on training data only

    Default cutoffs match the actual DB date range (Jan 2025 – present):
      train : < 2025-09-01   (Jan – Aug 2025)
      val   : [2025-09-01, 2026-01-01)  (Sep – Dec 2025)
      test  : >= 2026-01-01  (Jan 2026 – present)

    Parameters
    ----------
    db_path      : path to the SQLite database
    train_cutoff : fights before this date → train (default 2025-09-01)
    val_cutoff   : fights in [train_cutoff, val_cutoff) → val;
                   fights on/after val_cutoff → test (default 2026-01-01)

    Returns
    -------
    dict with keys:
        X_train, y_train       — scaled training arrays
        X_val, y_val           — scaled validation arrays
        X_test, y_test         — scaled test arrays
        scaler                 — fitted StandardScaler
        feature_names          — list of 72 feature names
        event_ids_test         — int array mapping test rows to events
        summary                — dict with dataset statistics
    """
    db_path = str(Path(db_path).resolve()) if not Path(db_path).is_absolute() else db_path
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run the data pipeline first: python main.py collect"
        )

    logger.info("Loading fight data from %s ...", db_path)

    # ── 1. Query fights ──────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    fights = [dict(r) for r in conn.execute(
        """
        SELECT
            f.id            AS fight_id,
            f.fighter1_id,
            f.fighter2_id,
            f.is_bonus_fight,
            f.event_id,
            e.date          AS event_date
        FROM fights f
        LEFT JOIN events e ON f.event_id = e.id
        WHERE f.fighter1_id IS NOT NULL
          AND f.fighter2_id IS NOT NULL
          AND e.date IS NOT NULL
        ORDER BY e.date ASC
        """
    ).fetchall()]

    if not fights:
        conn.close()
        raise ValueError(
            "No usable fights found in DB (need fighter IDs and event dates). "
            "Run the data pipeline first."
        )

    # ── 2. Load fighter cache ────────────────────────────────────────────
    fighters: dict[int, dict] = {
        row["id"]: dict(row)
        for row in conn.execute("SELECT * FROM fighters").fetchall()
    }
    conn.close()

    # ── 3. Build vectors (SINGLE ordering per fight, pre-split) ──────────
    quality_issues = _QualityTracker()

    records: list[dict] = []
    for fight in fights:
        f1 = fighters.get(fight["fighter1_id"])
        f2 = fighters.get(fight["fighter2_id"])

        if f1 is None or f2 is None:
            quality_issues.record("missing_fighter_profile", fight["fight_id"])
            continue

        vec = build_full_matchup_vector(f1, f2)

        nan_count = int(np.isnan(vec).sum())
        if nan_count > 0:
            quality_issues.record("nan_features", fight["fight_id"], detail=f"{nan_count} NaNs")
            vec = np.nan_to_num(vec, nan=0.0)

        records.append({
            "fight_id": fight["fight_id"],
            "event_id": fight["event_id"],
            "event_date": fight["event_date"],
            "label": int(fight["is_bonus_fight"] or 0),
            "vec_ab": vec,
            "f1": f1,
            "f2": f2,
        })

    # ── 4. Temporal split (on unique fights, BEFORE augmentation) ────────
    train_recs = [r for r in records if r["event_date"] < train_cutoff]
    val_recs   = [r for r in records if train_cutoff <= r["event_date"] < val_cutoff]
    test_recs  = [r for r in records if r["event_date"] >= val_cutoff]

    if not train_recs:
        raise ValueError(f"No training fights before {train_cutoff}. Check date data.")
    if not val_recs:
        raise ValueError(f"No validation fights in [{train_cutoff}, {val_cutoff}). Check date data.")
    if not test_recs:
        raise ValueError(f"No test fights on/after {val_cutoff}. Check date data.")

    # ── 5. Symmetric augmentation WITHIN each split ──────────────────────
    def _augment(recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add (B,A) ordering for each fight. Returns X, y, event_ids."""
        X_rows, y_rows, eids = [], [], []
        for r in recs:
            vec_ba = build_full_matchup_vector(r["f2"], r["f1"])

            nan_count = int(np.isnan(vec_ba).sum())
            if nan_count > 0:
                vec_ba = np.nan_to_num(vec_ba, nan=0.0)

            X_rows.append(r["vec_ab"])
            X_rows.append(vec_ba)
            y_rows.extend([r["label"], r["label"]])
            eids.extend([r["event_id"], r["event_id"]])
        return (
            np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(eids, dtype=np.int32),
        )

    X_train, y_train, _ = _augment(train_recs)
    X_val, y_val, _     = _augment(val_recs)
    X_test, y_test, event_ids_test = _augment(test_recs)

    # ── 6. Feature-level NaN / constant checks ───────────────────────────
    for col_idx in range(N_FEATURES):
        col = X_train[:, col_idx]
        if np.all(col == col[0]):
            quality_issues.record(
                "constant_feature", ALL_FEATURE_NAMES[col_idx],
                detail=f"value={col[0]:.4f}",
            )

    # ── 7. Fit scaler on train only ──────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # ── 8. Summary ───────────────────────────────────────────────────────
    def _date_range(recs):
        dates = [r["event_date"] for r in recs]
        return min(dates), max(dates)

    summary = {
        "total_unique_fights": len(records),
        "train": {"fights": len(train_recs), "samples": len(y_train),
                  "pos": int(y_train.sum()), "pos_pct": float(y_train.mean() * 100),
                  "date_range": _date_range(train_recs)},
        "val":   {"fights": len(val_recs), "samples": len(y_val),
                  "pos": int(y_val.sum()), "pos_pct": float(y_val.mean() * 100),
                  "date_range": _date_range(val_recs)},
        "test":  {"fights": len(test_recs), "samples": len(y_test),
                  "pos": int(y_test.sum()), "pos_pct": float(y_test.mean() * 100),
                  "date_range": _date_range(test_recs)},
        "features": N_FEATURES,
        "quality_issues": quality_issues.issues,
    }

    _print_summary(summary, quality_issues)

    return {
        "X_train": X_train_scaled, "y_train": y_train,
        "X_val": X_val_scaled, "y_val": y_val,
        "X_test": X_test_scaled, "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(ALL_FEATURE_NAMES),
        "event_ids_test": event_ids_test,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality tracking
# ─────────────────────────────────────────────────────────────────────────────

class _QualityTracker:
    """Accumulates data quality warnings during loading."""

    def __init__(self) -> None:
        self.issues: dict[str, list] = {}

    def record(self, category: str, item, detail: str = "") -> None:
        self.issues.setdefault(category, []).append(
            {"item": item, "detail": detail} if detail else item
        )

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def count(self, category: str) -> int:
        return len(self.issues.get(category, []))


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, quality: _QualityTracker) -> None:
    """Print a clean data summary to stdout and logger."""
    lines = [
        "",
        "=" * 72,
        "  DATA SUMMARY — UFC Fight Entertainment Dataset",
        "=" * 72,
        f"  Total unique fights:  {summary['total_unique_fights']}",
        f"  Feature dimensions:   {summary['features']}",
        "",
    ]

    for split_name in ("train", "val", "test"):
        s = summary[split_name]
        d0, d1 = s["date_range"]
        lines.append(
            f"  {split_name.upper():<6}  "
            f"{s['fights']:>5} fights  ({s['samples']:>5} samples w/ augmentation)  "
            f"{s['pos']:>4} pos ({s['pos_pct']:.1f}%)  "
            f"  [{d0} → {d1}]"
        )

    lines.append("")

    if quality.has_issues:
        lines.append("  DATA QUALITY FLAGS:")
        for cat, items in quality.issues.items():
            lines.append(f"    - {cat}: {len(items)} occurrences")
            if cat == "nan_features":
                for entry in items[:5]:
                    lines.append(f"        fight {entry['item']}: {entry['detail']}")
                if len(items) > 5:
                    lines.append(f"        ... and {len(items) - 5} more")
            elif cat == "constant_feature":
                for entry in items[:10]:
                    lines.append(f"        {entry['item']}: {entry['detail']}")
            elif cat == "missing_fighter_profile":
                lines.append(f"        fight_ids: {items[:10]}{'...' if len(items) > 10 else ''}")
    else:
        lines.append("  Data quality: no issues detected.")

    lines.append("=" * 72)

    output = "\n".join(lines)
    print(output)
    logger.info(output)


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
        data = load_real_data(db_path)
        print(f"\nReturned arrays:")
        print(f"  X_train: {data['X_train'].shape}  y_train: {data['y_train'].shape}")
        print(f"  X_val:   {data['X_val'].shape}  y_val:   {data['y_val'].shape}")
        print(f"  X_test:  {data['X_test'].shape}  y_test:  {data['y_test'].shape}")
        print(f"  Scaler fitted: {data['scaler'] is not None}")
        print(f"  Feature names: {len(data['feature_names'])} names")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("To generate a test DB, run: python main.py collect")
        sys.exit(1)
