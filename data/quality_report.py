"""
data/quality_report.py
Lorenzo's L3 — print and persist a data-quality report covering:
  - DB volumes (fighters, events, fights, bonuses)
  - is_bonus_fight class balance
  - Per-event bonus coverage and rate
  - Temporal coverage and gaps
  - Bonus-row dedupe sanity check
  - Per-event summary CSV at outputs/event_bonus_summary.csv
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

from data.db import Database
from config import BASE_DIR

logger = logging.getLogger(__name__)

OUTPUTS_DIR = BASE_DIR / "outputs"
SUMMARY_CSV = OUTPUTS_DIR / "event_bonus_summary.csv"


def report(db: Database) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with db.connect() as conn:
        # ── Volumes ─────────────────────────────────────────────────────────
        n_fighters = conn.execute("SELECT COUNT(*) AS n FROM fighters").fetchone()["n"]
        n_events   = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        n_fights   = conn.execute("SELECT COUNT(*) AS n FROM fights").fetchone()["n"]
        n_bonuses  = conn.execute("SELECT COUNT(*) AS n FROM fight_bonuses").fetchone()["n"]

        # ── Class balance ───────────────────────────────────────────────────
        n_pos = conn.execute(
            "SELECT COUNT(*) AS n FROM fights WHERE is_bonus_fight = 1"
        ).fetchone()["n"]
        n_neg = n_fights - n_pos

        # ── Bonus types ─────────────────────────────────────────────────────
        bonus_by_type = {
            r["bonus_type"]: r["n"]
            for r in conn.execute(
                "SELECT bonus_type, COUNT(*) AS n FROM fight_bonuses GROUP BY bonus_type"
            ).fetchall()
        }

        # ── Resolution rates ────────────────────────────────────────────────
        resolved_fighter = conn.execute(
            "SELECT COUNT(*) AS n FROM fight_bonuses WHERE fighter_id IS NOT NULL"
        ).fetchone()["n"]
        resolved_fight = conn.execute(
            "SELECT COUNT(*) AS n FROM fight_bonuses WHERE fight_id IS NOT NULL"
        ).fetchone()["n"]

        # ── Dedupe sanity: same fight + bonus_type + fighter must be unique
        dupes = conn.execute(
            """SELECT event_id, bonus_type, fighter_name, COUNT(*) AS c
               FROM fight_bonuses
               GROUP BY event_id, bonus_type, fighter_name
               HAVING c > 1"""
        ).fetchall()

        # ── Temporal coverage ───────────────────────────────────────────────
        date_range = conn.execute(
            "SELECT MIN(date) AS mn, MAX(date) AS mx FROM events WHERE date != ''"
        ).fetchone()
        empty_dates = conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE date IS NULL OR date = ''"
        ).fetchone()["n"]

        per_year = {
            r["y"]: r["n"]
            for r in conn.execute(
                "SELECT substr(date,1,4) AS y, COUNT(*) AS n "
                "FROM events GROUP BY y ORDER BY y"
            ).fetchall()
        }
        per_month_counts = list(conn.execute(
            "SELECT substr(date,1,7) AS m, COUNT(*) AS events "
            "FROM events WHERE date != '' GROUP BY m ORDER BY m"
        ).fetchall())

        # ── Per-event summary ───────────────────────────────────────────────
        per_event = list(conn.execute(
            """SELECT
                   e.id AS event_id,
                   e.date,
                   e.name,
                   COUNT(DISTINCT f.id) AS num_fights,
                   COUNT(DISTINCT CASE WHEN f.is_bonus_fight = 1 THEN f.id END) AS num_bonus_fights,
                   COUNT(DISTINCT CASE WHEN b.bonus_type='FOTN' THEN b.id END) AS fotn_rows,
                   COUNT(DISTINCT CASE WHEN b.bonus_type='POTN' THEN b.id END) AS potn_rows
               FROM events e
               LEFT JOIN fights f ON f.event_id = e.id
               LEFT JOIN fight_bonuses b ON b.event_id = e.id
               GROUP BY e.id
               ORDER BY e.date""".strip()
        ).fetchall())

    # ── Detect events that yielded no bonuses ──────────────────────────────
    events_no_bonuses = [dict(r)["name"] for r in per_event if r["fotn_rows"] == 0 and r["potn_rows"] == 0]

    # ── Write CSV ──────────────────────────────────────────────────────────
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["event_id", "date", "name", "num_fights", "num_bonus_fights",
             "fotn_rows", "potn_rows", "bonus_rate"]
        )
        for r in per_event:
            d = dict(r)
            rate = (d["num_bonus_fights"] / d["num_fights"]) if d["num_fights"] else 0.0
            writer.writerow([d["event_id"], d["date"], d["name"],
                             d["num_fights"], d["num_bonus_fights"],
                             d["fotn_rows"], d["potn_rows"], f"{rate:.3f}"])
    logger.info("Wrote %s", SUMMARY_CSV)

    # ── Print human-readable summary ────────────────────────────────────────
    print("=" * 70)
    print("UFC MATCHMAKER — DATA QUALITY REPORT (Lorenzo / L3)")
    print("=" * 70)
    print(f"Fighters:               {n_fighters:>6}")
    print(f"Events:                 {n_events:>6}")
    print(f"Fights:                 {n_fights:>6}")
    print(f"Bonus rows (raw):       {n_bonuses:>6}  "
          f"(FOTN={bonus_by_type.get('FOTN',0)}, "
          f"POTN={bonus_by_type.get('POTN',0)})")
    print()
    print("--- Bonus row resolution ---")
    print(f"  fighter_id resolved:  {resolved_fighter}/{n_bonuses}")
    print(f"  fight_id resolved:    {resolved_fight}/{n_bonuses}")
    print(f"  duplicate rows:       {len(dupes)}  (hard zero expected)")
    print()
    print("--- Class balance (training label is_bonus_fight) ---")
    pos_pct = 100 * n_pos / max(n_fights, 1)
    print(f"  positive (bonus):     {n_pos:>4}  ({pos_pct:.1f}%)")
    print(f"  negative:             {n_neg:>4}  ({100 - pos_pct:.1f}%)")
    print()
    print("--- Temporal coverage ---")
    print(f"  date range:           {date_range['mn']}  →  {date_range['mx']}")
    print(f"  events with no date:  {empty_dates}")
    print(f"  events per year:      {per_year}")
    print()
    print("--- Events with NO bonus rows (Wikipedia 404 or non-UFC) ---")
    for name in events_no_bonuses:
        print(f"    - {name}")
    print()
    print(f"Per-event summary written to: {SUMMARY_CSV}")
    print("=" * 70)

    return {
        "fighters": n_fighters,
        "events": n_events,
        "fights": n_fights,
        "bonus_rows": n_bonuses,
        "bonus_rows_by_type": bonus_by_type,
        "fighter_resolved": resolved_fighter,
        "fight_resolved": resolved_fight,
        "duplicate_rows": len(dupes),
        "positive_fights": n_pos,
        "positive_pct": round(pos_pct, 2),
        "date_min": date_range["mn"],
        "date_max": date_range["mx"],
        "events_with_empty_date": empty_dates,
        "events_no_bonuses": events_no_bonuses,
        "events_per_year": per_year,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report(Database())
