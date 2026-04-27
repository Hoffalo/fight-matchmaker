"""
data/export_labels.py
Export the FOTN/POTN ground-truth labels in a portable, natural-key format
so teammates can import them without re-scraping Wikipedia and without
relying on locally-assigned event/fighter/fight IDs.

The CSV uses (event_name, event_date, bonus_type, fighter_name) as the
natural key. To consume it, a teammate runs:

    from data.import_labels import import_labels  # see import side
    # OR re-run the scraper (also idempotent):
    python -m scrapers.wikipedia_bonus_scraper

Output: data/labels/fight_bonuses.csv
"""
import csv
import logging
from pathlib import Path

from data.db import Database
from config import BASE_DIR

logger = logging.getLogger(__name__)

LABELS_DIR = BASE_DIR / "data" / "labels"
BONUSES_CSV = LABELS_DIR / "fight_bonuses.csv"


def export(db: Database) -> dict:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with db.connect() as conn:
        rows = conn.execute(
            """SELECT
                   e.name  AS event_name,
                   e.date  AS event_date,
                   b.bonus_type,
                   b.fighter_name,
                   b.source
               FROM fight_bonuses b
               JOIN events e ON e.id = b.event_id
               ORDER BY e.date, e.name, b.bonus_type, b.fighter_name"""
        ).fetchall()

    with open(BONUSES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_name", "event_date", "bonus_type",
                         "fighter_name", "source"])
        for r in rows:
            writer.writerow([r["event_name"], r["event_date"],
                             r["bonus_type"], r["fighter_name"], r["source"]])

    logger.info("Wrote %d bonus rows to %s", len(rows), BONUSES_CSV)
    return {"rows": len(rows), "path": str(BONUSES_CSV)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(export(Database()))
