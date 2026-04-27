"""
data/import_labels.py
Mirror of export_labels.py — read data/labels/fight_bonuses.csv and apply it
to the local DB. Useful for teammates who don't want to scrape Wikipedia
themselves.

Strategy:
  1. Read each row (event_name, event_date, bonus_type, fighter_name, source).
  2. Resolve event_id by (name, date) match against the local events table.
     Skip rows whose event isn't in the local DB yet.
  3. INSERT OR IGNORE into fight_bonuses (event_id, bonus_type, fighter_name,
     source). The UNIQUE constraint dedupes against any prior runs.
  4. Reuse the bonus matcher to fill fighter_id and fight_id.
  5. Refresh the is_bonus_fight column on fights.

Run: python -m data.import_labels
"""
import csv
import logging

from data.db import Database
from data.export_labels import BONUSES_CSV
from scrapers.wikipedia_bonus_scraper import match_bonuses_to_fights

logger = logging.getLogger(__name__)


def import_labels(db: Database, csv_path=BONUSES_CSV) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m data.export_labels` "
            "or `python -m scrapers.wikipedia_bonus_scraper` first."
        )

    with db.connect() as conn:
        events_lookup: dict[tuple[str, str], int] = {
            (r["name"], r["date"]): r["id"]
            for r in conn.execute("SELECT id, name, date FROM events")
        }

    inserted = 0
    skipped_unknown_event = 0

    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        with db.connect() as conn:
            for row in reader:
                event_id = events_lookup.get((row["event_name"], row["event_date"]))
                if event_id is None:
                    skipped_unknown_event += 1
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO fight_bonuses "
                    "(event_id, bonus_type, fighter_name, source) "
                    "VALUES (?, ?, ?, ?)",
                    (event_id, row["bonus_type"],
                     row["fighter_name"], row.get("source") or "csv"),
                )
                if cur.rowcount > 0:
                    inserted += 1

    logger.info("Imported %d new bonus rows (%d skipped — event not in local DB)",
                inserted, skipped_unknown_event)

    match_summary = match_bonuses_to_fights(db)
    flagged = db.refresh_bonus_labels()

    return {
        "rows_inserted": inserted,
        "rows_skipped_unknown_event": skipped_unknown_event,
        "match_summary": match_summary,
        "fights_flagged_is_bonus_fight": flagged,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(import_labels(Database()))
