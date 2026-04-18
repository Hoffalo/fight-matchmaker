"""
data/build_mma_db.py
Cross-reference Tapology + Sherdog into a single MMA fighters CSV.

Pipeline:
    1. Seed fighter list from Tapology rankings (ranked fighters across weight classes)
    2. For each fighter, fetch profile from Tapology (Selenium) and Sherdog (HTTP)
    3. Merge by normalized name (Sherdog wins on record, majority on country,
       max on last_fight_date, seed > tap > sd for weight_class)
    4. Dedupe and write to data/mma_fighters.csv

Usage:
    python -m data.build_mma_db                    # all ranked fighters
    python -m data.build_mma_db --limit 20         # smoke test on 20 fighters
    python -m data.build_mma_db --no-selenium      # skip Tapology enrichment step
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow "python data/build_mma_db.py" from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from scrapers.sherdog_scraper import SherdogScraper
from scrapers.tapology_scraper import TapologyScraper

logger = logging.getLogger(__name__)

CSV_PATH = DATA_DIR / "mma_fighters.csv"
CSV_COLUMNS = [
    "name", "weight_class", "record", "last_fight_date", "country",
    "sherdog_url", "tapology_url", "sources",
]


def normalize_name(name: str) -> str:
    """Normalize for cross-source matching: lowercase, strip punctuation/whitespace."""
    n = (name or "").lower()
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace(".", "")
    for fmt in ("%Y-%m-%d", "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def seed_from_tapology_rankings() -> list[dict]:
    """Scrape Tapology rankings to get the ranked-fighter seed list."""
    logger.info("Seeding from Tapology rankings...")
    with TapologyScraper(db=None, headless=True) as tap:  # db unused for seed phase
        rankings = tap.scrape_rankings()

    seeds = []
    for wc, fighters in rankings.items():
        for f in fighters:
            seeds.append({
                "name": f["name"],
                "weight_class": wc,
                "tapology_url": f.get("tapology_url", ""),
                "rank": f.get("rank"),
            })
    logger.info("Seed list: %d fighters", len(seeds))
    return seeds


def enrich_from_tapology(seeds: list[dict]) -> dict[str, dict]:
    """Visit each Tapology fighter page to extract record/country/last_fight_date."""
    logger.info("Enriching %d fighters from Tapology profile pages...", len(seeds))
    results = {}
    with TapologyScraper(db=None, headless=True) as tap:
        for i, seed in enumerate(seeds, 1):
            if not seed.get("tapology_url"):
                continue
            logger.info("[Tapology %d/%d] %s", i, len(seeds), seed["name"])
            profile = tap.scrape_fighter_page(seed["tapology_url"])
            results[normalize_name(seed["name"])] = profile
    return results


def scrape_sherdog(seeds: list[dict]) -> dict[str, dict]:
    """Parallel HTTP scrape of Sherdog profiles."""
    sherdog = SherdogScraper(requests_per_minute=20)
    results: dict[str, dict] = {}

    def _go(name):
        return name, sherdog.scrape_fighter(name)

    names = [s["name"] for s in seeds]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_go, n): n for n in names}
        for i, fut in enumerate(as_completed(futures), 1):
            name, data = fut.result()
            logger.info("[Sherdog %d/%d] %s", i, len(names), name)
            results[normalize_name(name)] = data

    return results


def merge_fighter(seed: dict, tap: dict, sd: dict) -> dict:
    """Merge per-source records into a single canonical row."""
    sources = []
    if tap and (tap.get("record") or tap.get("last_fight_date") or tap.get("country")):
        sources.append("tapology")
    if sd and sd.get("record"):
        sources.append("sherdog")

    record = (sd.get("record") if sd else None) or (tap.get("record") if tap else None) or ""

    countries = [x for x in [(sd or {}).get("country"), (tap or {}).get("country")] if x]
    country = max(set(countries), key=countries.count) if countries else ""

    weight_class = (
        seed.get("weight_class")
        or (tap or {}).get("weight_class")
        or (sd or {}).get("weight_class")
        or ""
    )

    candidates = [x for x in [(tap or {}).get("last_fight_date"), (sd or {}).get("last_fight_date")] if x]
    parsed = [(parse_date(c), c) for c in candidates if parse_date(c)]
    last_fight_date = max(parsed, key=lambda p: p[0])[1] if parsed else (candidates[0] if candidates else "")

    return {
        "name":            seed["name"],
        "weight_class":    weight_class,
        "record":          record,
        "last_fight_date": last_fight_date,
        "country":         country,
        "sherdog_url":     (sd or {}).get("sherdog_url", ""),
        "tapology_url":    seed.get("tapology_url", ""),
        "sources":         "|".join(sources),
    }


def build(limit: Optional[int] = None, skip_selenium: bool = False) -> Path:
    seeds = seed_from_tapology_rankings()
    if limit:
        seeds = seeds[:limit]
        logger.info("Limited seed list to %d fighters", len(seeds))

    # Dedupe seeds by normalized name (keep first occurrence = highest-ranking)
    seen = set()
    deduped = []
    for s in seeds:
        key = normalize_name(s["name"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    seeds = deduped
    logger.info("After dedupe: %d unique fighters", len(seeds))

    tap_data: dict = {}
    if not skip_selenium:
        tap_data = enrich_from_tapology(seeds)

    sd_data = scrape_sherdog(seeds)

    rows = []
    for seed in seeds:
        key = normalize_name(seed["name"])
        rows.append(merge_fighter(
            seed,
            tap_data.get(key, {}),
            sd_data.get(key, {}),
        ))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    logger.info("Wrote %d fighters to %s", len(rows), CSV_PATH)
    return CSV_PATH


def main():
    parser = argparse.ArgumentParser(description="Build cross-referenced MMA fighter CSV")
    parser.add_argument("--limit", type=int, default=None, help="Cap fighters (smoke test)")
    parser.add_argument("--no-selenium", action="store_true", help="Skip Tapology profile enrichment (ranking seed only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    build(limit=args.limit, skip_selenium=args.no_selenium)


if __name__ == "__main__":
    main()
