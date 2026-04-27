"""
data/backfill_event_dates.py
One-off backfill: events.date is empty in the existing DB because the original
scrape_all_events read the wrong cell. Hit the events list once, normalize each
date to ISO (YYYY-MM-DD), and update existing rows by ufcstats_url.

Run: python -m data.backfill_event_dates
"""
import logging
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from data.db import Database
from config import URLS

logger = logging.getLogger(__name__)


_MONTHS = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june",
     "july","august","september","october","november","december"], start=1)}


def _normalize_date(raw: str) -> str | None:
    """'April 18, 2026' -> '2026-04-18'. Returns None if unparseable."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        return datetime.strptime(raw, "%B %d, %Y").date().isoformat()
    except ValueError:
        m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", raw)
        if not m:
            return None
        month = _MONTHS.get(m.group(1).lower())
        if not month:
            return None
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"


def fetch_event_dates() -> dict[str, str]:
    """Fetch the UFCStats events-list page and return {ufcstats_url: iso_date}."""
    resp = requests.get(
        URLS["ufcstats_events"],
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    out: dict[str, str] = {}
    for row in soup.select("tr.b-statistics__table-row"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        link = cells[0].select_one("a")
        if not link:
            continue
        url = link.get("href", "")
        date_span = cells[0].select_one("span.b-statistics__date")
        raw_date = date_span.get_text(strip=True) if date_span else ""
        iso = _normalize_date(raw_date)
        if url and iso:
            out[url] = iso
    return out


def backfill(db: Database) -> dict:
    url_to_date = fetch_event_dates()
    logger.info("Pulled %d (url -> date) entries from UFCStats", len(url_to_date))

    updated = 0
    missing = 0
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, ufcstats_url, date FROM events"
        ).fetchall()
        for row in rows:
            url = row["ufcstats_url"]
            iso = url_to_date.get(url)
            if not iso:
                missing += 1
                continue
            if row["date"] == iso:
                continue
            conn.execute(
                "UPDATE events SET date=? WHERE id=?", (iso, row["id"])
            )
            updated += 1

    logger.info("Updated %d events; %d had no match in events-list page", updated, missing)
    return {"updated": updated, "missing": missing, "total_in_db": len(rows)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = Database()
    result = backfill(db)
    print(result)
