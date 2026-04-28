"""
Create a minimal SQLite DB so temporal splits (train / val / test) are non-empty.

Used for local PCA/tests without running the full scrape pipeline.

Usage (from repo root):
    python data/seed_minimal_splits_db.py

Writes config.DB_PATH (default data/ufc_matchmaker.db).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as script: repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_PATH
from data.db import Database

logger = logging.getLogger(__name__)

# Dates aligned with models/data_loader.TRAIN_CUTOFF / VAL_CUTOFF:
# train < 2025-09-01 | val [2025-09-01, 2026-01-01) | test >= 2026-01-01


def _fighter_row(
    name: str,
    fid: int,
    *,
    wins: int = 12,
    losses: int = 3,
    ranking: int | None = 10,
    wc: str = "Lightweight",
) -> dict:
    return {
        "id": fid,
        "name": name,
        "weight_class": wc,
        "ranking": ranking,
        "is_champion": 0,
        "wins_total": wins,
        "losses_total": losses,
        "losses_ko": 1,
        "losses_sub": 0,
        "height_cm": 178.0,
        "reach_cm": 182.0,
        "sig_strikes_pm": 5.5,
        "sig_strike_acc": 0.48,
        "sig_strikes_abs_pm": 4.0,
        "sig_strike_def": 0.58,
        "td_avg": 2.0,
        "td_acc": 0.42,
        "td_def": 0.72,
        "sub_avg": 0.8,
        "ctrl_time_avg": 90.0,
        "ko_rate": 0.35,
        "sub_rate": 0.2,
        "dec_rate": 0.45,
        "finish_rate": 0.55,
        "grapple_ratio": 0.35,
    }


def seed(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or DB_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(str(path))

    fighters_spec = [
        _fighter_row("Seed Fighter Alpha", 1, ranking=5),
        _fighter_row("Seed Fighter Bravo", 2, ranking=8),
        _fighter_row("Seed Fighter Charlie", 3, ranking=None),
        _fighter_row("Seed Fighter Delta", 4, ranking=12),
        _fighter_row("Seed Fighter Echo", 5, ranking=14),
        _fighter_row("Seed Fighter Foxtrot", 6, ranking=None),
    ]

    events_spec = [
        (1, "Seed Event Train", "2025-06-15"),
        (2, "Seed Event Val", "2025-10-15"),
        (3, "Seed Event Test", "2026-02-01"),
    ]

    # Train: six distinct pairings among 1–4 → 12 augmented rows (enough for PCA).
    # Val / test: one fight each so splits stay non-empty.
    fights_spec = [
        (1, 1, 2, 1, 0),
        (2, 1, 3, 1, 0),
        (3, 1, 4, 1, 0),
        (4, 2, 3, 1, 0),
        (5, 2, 4, 1, 0),
        (6, 3, 4, 1, 0),
        (7, 5, 6, 2, 1),
        (8, 1, 5, 3, 0),
    ]

    with db.connect() as conn:
        conn.execute("DELETE FROM fight_stats")
        conn.execute("DELETE FROM fight_bonuses")
        conn.execute("DELETE FROM fights")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM fighters")

        for f in fighters_spec:
            cols = ", ".join(f.keys())
            placeholders = ", ".join(["?"] * len(f))
            conn.execute(
                f"INSERT INTO fighters ({cols}) VALUES ({placeholders})",
                tuple(f.values()),
            )

        for eid, name, date in events_spec:
            conn.execute(
                "INSERT INTO events (id, name, date) VALUES (?, ?, ?)",
                (eid, name, date),
            )

        for fight_id, f1, f2, eid, bonus in fights_spec:
            conn.execute(
                """
                INSERT INTO fights (id, event_id, fighter1_id, fighter2_id,
                    weight_class, is_bonus_fight)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fight_id, eid, f1, f2, "Lightweight", bonus),
            )

    logger.info(
        "Seeded minimal DB at %s (8 fights: 6 train / 1 val / 1 test)",
        path,
    )
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = seed()
    print(f"OK — seeded {p}")
