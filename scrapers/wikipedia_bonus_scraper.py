"""
scrapers/wikipedia_bonus_scraper.py
Scrape UFC Fight of the Night / Performance of the Night bonus winners from
Wikipedia event pages. Each event page has a "Bonus awards" section with a
bulleted list of recipients.

This populates the `fight_bonuses` table with raw (event_id, bonus_type,
fighter_name) rows. Resolving fighter_name -> fighter_id and the corresponding
fight_id is a separate matching step (see match_bonuses_to_fights).
"""
import logging
import re
import time
import unicodedata
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from data.db import Database

logger = logging.getLogger(__name__)

WIKI_BASE = "https://en.wikipedia.org/wiki/"
HEADERS = {
    "User-Agent": (
        "UFC-Matchmaker-Research/1.0 "
        "(educational; lorenzohoff2006@gmail.com)"
    )
}


# ── URL slug mapping ─────────────────────────────────────────────────────────

def event_name_to_wiki_slug(event_name: str) -> str:
    """
    Map a DB event name to a Wikipedia URL slug.

      "UFC 326: Holloway vs. Oliveira 2"     -> "UFC_326"
      "UFC Fight Night: Sterling vs. Zalal"  -> "UFC_Fight_Night:_Sterling_vs._Zalal"
      "UFC on ESPN: ..."                     -> full name with underscores
    """
    name = event_name.strip()
    m = re.match(r"^(UFC\s+\d+)(?::|$)", name)
    if m:
        return m.group(1).replace(" ", "_")
    return name.replace(" ", "_")


def wiki_url(event_name: str) -> str:
    return WIKI_BASE + quote(event_name_to_wiki_slug(event_name), safe=":_")


# ── Bonus list parsing ──────────────────────────────────────────────────────

_VS_SPLIT = re.compile(r"\s+vs\.?\s+", re.I)
_NAME_LIST_SPLIT = re.compile(r"\s*,\s*and\s+|\s+and\s+|\s*,\s*", re.I)


def parse_bonus_list_item(text: str) -> tuple[str | None, list[str]]:
    """
    Parse one <li> from the Bonus awards section.

      "Fight of the Night: A vs. B"            -> ("FOTN", ["A","B"])
      "Performance of the Night: X, Y, and Z"  -> ("POTN", ["X","Y","Z"])
      "Fight of the Night: No bonus awarded."  -> ("FOTN", [])
      "Submission of the Night: ..."           -> (None, [])  (legacy, ignored)
    """
    m = re.match(r"\s*([^:]+):\s*(.*)$", text, re.DOTALL)
    if not m:
        return None, []
    head = m.group(1).strip().lower()
    body = m.group(2).strip()

    if "fight of the night" in head:
        bonus_type = "FOTN"
    elif "performance of the night" in head:
        bonus_type = "POTN"
    else:
        return None, []

    if "no bonus" in body.lower() or "not awarded" in body.lower() or not body:
        return bonus_type, []

    body = re.sub(r"\[\s*\d+\s*\]", "", body)  # strip footnote refs like [1]

    if bonus_type == "FOTN":
        parts = _VS_SPLIT.split(body)
        if len(parts) == 2:
            return bonus_type, [_clean_name(parts[0]), _clean_name(parts[1])]
        # Fallback: occasionally a comma-separated pair
        return bonus_type, [_clean_name(p) for p in _NAME_LIST_SPLIT.split(body) if p.strip()]

    # POTN
    return bonus_type, [_clean_name(p) for p in _NAME_LIST_SPLIT.split(body) if p.strip()]


def _clean_name(s: str) -> str:
    return s.strip(" .,;:'\" ")


# ── Page fetch + section extraction ─────────────────────────────────────────

def fetch_event_bonuses(event_name: str) -> list[dict]:
    """
    Scrape one event's bonuses. Returns list of
    {bonus_type, fighter_name, source_url}.
    Empty list if the page or section is missing.
    """
    url = wiki_url(event_name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        logger.warning("Wikipedia fetch failed for %r: %s", event_name, e)
        return []
    if resp.status_code != 200:
        logger.info("Wikipedia returned %d for %r (%s)", resp.status_code, event_name, url)
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    span = soup.find("span", id=re.compile(r"^Bonus(_awards)?$", re.I))
    if span is None:
        h = soup.find(["h2", "h3", "h4"], string=re.compile("Bonus", re.I))
        span = h
    if span is None:
        logger.info("No 'Bonus awards' section in %r", event_name)
        return []

    ul = span.find_next("ul")
    if ul is None:
        return []

    out: list[dict] = []
    for li in ul.find_all("li", recursive=False):
        text = li.get_text(" ", strip=True)
        bonus_type, names = parse_bonus_list_item(text)
        if bonus_type is None:
            continue
        for n in names:
            if n:
                out.append(
                    {"bonus_type": bonus_type, "fighter_name": n, "source_url": url}
                )
    return out


# ── DB persistence ──────────────────────────────────────────────────────────

def insert_bonuses(db: Database, event_id: int, bonuses: list[dict]) -> int:
    """Insert bonuses for one event. Returns count of new rows inserted."""
    if not bonuses:
        return 0
    sql = (
        "INSERT OR IGNORE INTO fight_bonuses "
        "(event_id, bonus_type, fighter_name, source) VALUES (?, ?, ?, ?)"
    )
    inserted = 0
    with db.connect() as conn:
        for b in bonuses:
            cur = conn.execute(
                sql, (event_id, b["bonus_type"], b["fighter_name"], "wikipedia")
            )
            if cur.rowcount > 0:
                inserted += 1
    return inserted


def scrape_all_event_bonuses(db: Database, *, sleep_s: float = 0.4) -> dict:
    """
    For every event in the DB, fetch its Wikipedia bonuses and persist.
    Idempotent thanks to UNIQUE(event_id, bonus_type, fighter_name).
    """
    with db.connect() as conn:
        events = [
            dict(r)
            for r in conn.execute(
                "SELECT id, name, date FROM events ORDER BY date"
            ).fetchall()
        ]

    rows_inserted = 0
    events_with_bonuses = 0
    events_no_bonuses = 0
    failed_events: list[str] = []

    for ev in events:
        time.sleep(sleep_s)
        try:
            bonuses = fetch_event_bonuses(ev["name"])
        except Exception as e:
            logger.warning("Bonus scrape errored for %r: %s", ev["name"], e)
            failed_events.append(ev["name"])
            continue
        if not bonuses:
            events_no_bonuses += 1
            continue
        n = insert_bonuses(db, ev["id"], bonuses)
        rows_inserted += n
        events_with_bonuses += 1

    summary = {
        "events_scanned": len(events),
        "events_with_bonuses": events_with_bonuses,
        "events_no_bonuses": events_no_bonuses,
        "rows_inserted": rows_inserted,
        "failed_events": failed_events,
    }
    logger.info("Bonus scrape complete: %s", summary)
    return summary


# ── Name resolution + fight matching ────────────────────────────────────────

# Letters NFD doesn't decompose (standalone codepoints, not base+combining mark).
_LIGATURE_FOLD = str.maketrans({
    "ł": "l", "Ł": "L",
    "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D",
    "ß": "ss",
})


def _normalize_name(s: str) -> str:
    """Lowercase, fold ligatures, strip diacritics, hyphens, and punctuation."""
    s = s.translate(_LIGATURE_FOLD)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_fighter_lookup(db: Database) -> dict[str, int]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id, name FROM fighters").fetchall()
    lookup: dict[str, int] = {}
    for r in rows:
        key = _normalize_name(r["name"])
        if key and key not in lookup:
            lookup[key] = r["id"]
    return lookup


def match_bonuses_to_fights(db: Database) -> dict:
    """
    Fill fighter_id and fight_id on every fight_bonuses row.
    Returns counts: matched_fighter, matched_fight, unmatched_name, no_fight_in_event.
    """
    fighter_lookup = _build_fighter_lookup(db)
    matched_fighter = matched_fight = unmatched_name = no_fight = 0

    with db.connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, event_id, fighter_name, fighter_id, fight_id "
                "FROM fight_bonuses"
            ).fetchall()
        ]

        for row in rows:
            updates: dict = {}

            if row["fighter_id"] is None:
                key = _normalize_name(row["fighter_name"])
                fid = fighter_lookup.get(key)
                if fid is None:
                    # Try last-name + first-name swap or partial match
                    fid = _fuzzy_lookup(key, fighter_lookup)
                if fid is not None:
                    updates["fighter_id"] = fid
                    matched_fighter += 1
                else:
                    unmatched_name += 1
                    logger.debug(
                        "Unmatched bonus fighter name: %r", row["fighter_name"]
                    )
                fighter_id = fid
            else:
                fighter_id = row["fighter_id"]

            if row["fight_id"] is None and fighter_id is not None:
                fight = conn.execute(
                    "SELECT id FROM fights WHERE event_id=? "
                    "AND (fighter1_id=? OR fighter2_id=?) LIMIT 1",
                    (row["event_id"], fighter_id, fighter_id),
                ).fetchone()
                if fight is not None:
                    updates["fight_id"] = fight["id"]
                    matched_fight += 1
                else:
                    no_fight += 1

            if updates:
                cols = ", ".join(f"{k}=?" for k in updates)
                conn.execute(
                    f"UPDATE fight_bonuses SET {cols} WHERE id=?",
                    list(updates.values()) + [row["id"]],
                )

    summary = {
        "rows": len(rows),
        "matched_fighter": matched_fighter,
        "matched_fight": matched_fight,
        "unmatched_name": unmatched_name,
        "no_fight_in_event": no_fight,
    }
    logger.info("Bonus matching complete: %s", summary)
    return summary


def _fuzzy_lookup(norm_query: str, lookup: dict[str, int]) -> int | None:
    """
    Cheap fuzzy match for spelling variants. Tries (in order):
      1. Concatenated form ('rong zhu' -> 'rongzhu') if it exists.
      2. Token-set overlap >= 2 tokens (handles middle-name diffs).
      3. Last-token match, but only if it's globally unique (handles
         nickname/short-name first-name swaps like 'Bia' vs 'Beatriz').
    """
    if not norm_query:
        return None

    # 1. Concatenated form
    concat = norm_query.replace(" ", "")
    if concat and concat in lookup:
        return lookup[concat]

    q_tokens = norm_query.split()
    q_set = set(q_tokens)

    # 2. Token-set overlap >= 2 (only if query has >=2 tokens)
    if len(q_tokens) >= 2:
        best_id, best_score = None, 1
        for k, v in lookup.items():
            score = len(q_set & set(k.split()))
            if score > best_score:
                best_score = score
                best_id = v
        if best_id is not None:
            return best_id

    # 3. Last-name uniqueness fallback
    if q_tokens:
        last = q_tokens[-1]
        candidates = [v for k, v in lookup.items() if k.split() and k.split()[-1] == last]
        if len(candidates) == 1:
            return candidates[0]

    return None


# ── CLI entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    db = Database()
    print("Scraping bonuses from Wikipedia...")
    scrape_summary = scrape_all_event_bonuses(db)
    print(scrape_summary)
    print("\nMatching bonus rows to fights...")
    match_summary = match_bonuses_to_fights(db)
    print(match_summary)
    print("\nRefreshing fights.is_bonus_fight label column...")
    flagged = db.refresh_bonus_labels()
    print({"fights_flagged": flagged})
