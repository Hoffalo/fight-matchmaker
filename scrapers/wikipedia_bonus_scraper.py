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
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "UFC-Matchmaker-Research/1.0 "
        "(educational; lorenzohoff2006@gmail.com)"
    )
}

# Events known to lack a per-event Wikipedia page (regional/dev leagues, etc).
# Their fights are not reliable negatives for is_bonus_fight — see
# events_with_reliable_bonus_data() so training can filter them.
EXCLUDED_EVENT_NAMES: set[str] = {
    "UFC - Road to UFC 4.6",  # Asian developmental sub-league round, no per-round Wiki page
}

# UFCStats normalizes broadcast-tagged events to "UFC Fight Night", but
# Wikipedia preserves "UFC on ESPN" / "UFC on ABC" / "UFC on Fox" titles.
# When the direct slug 404s we fall back to Wikipedia search.
_VALID_EVENT_TITLE_RX = re.compile(
    r"^UFC (Fight Night|on \w+|\d+):", re.IGNORECASE
)


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


def _matchup_key(event_title: str) -> frozenset[str]:
    """
    Extract the {fighter_a, fighter_b} matchup tokens from a UFC event title,
    normalized for safe comparison across naming conventions:

      "UFC Fight Night: Whittaker vs. De Ridder"  -> {'whittaker', 'de ridder'}
      "UFC on ABC: Whittaker vs. de Ridder"       -> {'whittaker', 'de ridder'}
      "UFC Fight Night: Royval vs. Taira"         -> {'royval', 'taira'}

    Returns an empty set if the title has no `:` or `vs.` structure.
    """
    m = re.search(r":\s*(.+?)\s*$", event_title)
    if not m:
        return frozenset()
    parts = re.split(r"\s+vs\.?\s+", m.group(1), flags=re.I)
    if len(parts) != 2:
        return frozenset()
    out: set[str] = set()
    for p in parts:
        norm = re.sub(r"[^a-z\s]", "", p.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm:
            out.add(norm)
    return frozenset(out)


def _wiki_search_event_url(event_name: str) -> str | None:
    """
    Use Wikipedia's search API to find the canonical page when the direct
    slug fails (e.g. UFCStats says "UFC Fight Night: X vs. Y" but Wikipedia
    files the same event under "UFC on ESPN: X vs. Y").

    Walks all hits and returns the first whose title matches a UFC-event
    pattern AND has the same matchup as `event_name` — this rejects similar
    older events (e.g. "Royval vs. Taira" is not the same fight as
    "Taira vs. Park" even though both involve Taira).
    """
    try:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": event_name,
                "srlimit": 8,
                "format": "json",
            },
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Wikipedia search API failed for %r: %s", event_name, e)
        return None

    src_key = _matchup_key(event_name)
    hits = resp.json().get("query", {}).get("search", [])
    for hit in hits:
        title = hit.get("title", "")
        if not _VALID_EVENT_TITLE_RX.match(title):
            continue
        if src_key and _matchup_key(title) != src_key:
            continue
        return WIKI_BASE + quote(title.replace(" ", "_"), safe=":_")
    return None


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
        # Common case: a single matchup "A vs. B"
        parts = _VS_SPLIT.split(body)
        if len(parts) == 2:
            return bonus_type, [_clean_name(parts[0]), _clean_name(parts[1])]
        # Multi-FOTN case: "A vs. B and C vs. D" or "A vs. B, C vs. D"
        # — split on connectives to get matchup chunks, then split each by "vs."
        out: list[str] = []
        for chunk in _NAME_LIST_SPLIT.split(body):
            chunk = chunk.strip()
            if not chunk:
                continue
            sub = _VS_SPLIT.split(chunk)
            if len(sub) == 2:
                out.extend([_clean_name(sub[0]), _clean_name(sub[1])])
            else:
                out.append(_clean_name(chunk))
        return bonus_type, [n for n in out if n]

    # POTN
    return bonus_type, [_clean_name(p) for p in _NAME_LIST_SPLIT.split(body) if p.strip()]


def _clean_name(s: str) -> str:
    return s.strip(" .,;:'\" ")


# ── Page fetch + section extraction ─────────────────────────────────────────

def _fetch_page(url: str) -> requests.Response | None:
    try:
        return requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        logger.warning("Wikipedia fetch failed for %s: %s", url, e)
        return None


def fetch_event_bonuses(event_name: str) -> list[dict]:
    """
    Scrape one event's bonuses. Tries the direct slug first; on 404 falls
    back to Wikipedia search to handle UFCStats↔Wikipedia naming drift
    ("UFC Fight Night" ↔ "UFC on ESPN" / "UFC on ABC", lowercase 'de'/'van'
    surnames, etc.).
    """
    url = wiki_url(event_name)
    resp = _fetch_page(url)

    # Fallback: search API for naming-drift cases
    if resp is None or resp.status_code != 200:
        fallback = _wiki_search_event_url(event_name)
        if fallback and fallback != url:
            logger.info(
                "Direct slug 404 for %r; trying search-API URL %s",
                event_name, fallback,
            )
            url = fallback
            resp = _fetch_page(url)

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "ERR"
        logger.info("Wikipedia returned %s for %r (%s)", code, event_name, url)
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
    events_excluded = 0
    failed_events: list[str] = []

    for ev in events:
        if ev["name"] in EXCLUDED_EVENT_NAMES:
            logger.info("Skipping excluded event: %r", ev["name"])
            events_excluded += 1
            continue
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
        "events_excluded": events_excluded,
        "rows_inserted": rows_inserted,
        "failed_events": failed_events,
    }
    logger.info("Bonus scrape complete: %s", summary)
    return summary


def events_with_reliable_bonus_data(db: Database) -> set[int]:
    """
    Return event IDs whose bonus labels are trustworthy:
      - The event has at least one row in fight_bonuses, AND
      - The event is not in EXCLUDED_EVENT_NAMES.

    Use this to filter the training set per the audit's Strategy D — fights
    from events with no scraped bonuses (Wikipedia gaps) are not reliable
    negatives and should be dropped, not labeled is_bonus_fight=0.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT e.id
               FROM events e
               JOIN fight_bonuses b ON b.event_id = e.id
               WHERE e.name NOT IN ({})""".format(
                ",".join("?" * len(EXCLUDED_EVENT_NAMES))
            ) if EXCLUDED_EVENT_NAMES else
            """SELECT DISTINCT e.id FROM events e
               JOIN fight_bonuses b ON b.event_id = e.id""",
            tuple(EXCLUDED_EVENT_NAMES) if EXCLUDED_EVENT_NAMES else (),
        ).fetchall()
    return {r["id"] for r in rows}


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

    # 4. Last-name match + first-name prefix-compatible (handles short-form
    #    first names like Shara↔Sharabutdin, Tony↔Anthony, where the short
    #    form is a prefix of the full form). Only fires when exactly one
    #    candidate disambiguates this way.
    if len(q_tokens) >= 2:
        last = q_tokens[-1]
        q_first = q_tokens[0]
        prefix_candidates: list[int] = []
        for k, v in lookup.items():
            kt = k.split()
            if len(kt) < 2 or kt[-1] != last:
                continue
            k_first = kt[0]
            if q_first.startswith(k_first) or k_first.startswith(q_first):
                prefix_candidates.append(v)
        if len(prefix_candidates) == 1:
            return prefix_candidates[0]

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
