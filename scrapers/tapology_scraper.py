"""
scrapers/tapology_scraper.py
Selenium scraper for Tapology.com — supplementary source for:
  - Rankings (current pound-for-pound and divisional)
  - Historical betting odds context
  - Fan interest signals (fight ratings, view counts)
"""
import logging
import time
import re

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

from utils.driver import create_driver
from utils.rate_limiter import RateLimiter
from data.db import Database
from config import URLS, WEIGHT_CLASSES

logger = logging.getLogger(__name__)


class TapologyScraper:
    """Scrapes Tapology for fighter rankings and fan interest signals."""

    BASE = "https://www.tapology.com"

    def __init__(self, db: Database, headless: bool = True):
        self.db = db
        self.headless = headless
        self.driver = None
        self.rate_limiter = RateLimiter(requests_per_minute=10)

    def __enter__(self):
        self.driver = create_driver(headless=self.headless)
        return self

    def __exit__(self, *_):
        if self.driver:
            self.driver.quit()

    def _get(self, url: str) -> BeautifulSoup:
        self.rate_limiter.wait()
        self.driver.get(url)
        time.sleep(2)
        return BeautifulSoup(self.driver.page_source, "lxml")

    def scrape_rankings(self) -> dict[str, list[dict]]:
        """
        Scrape Tapology's divisional rankings.
        Returns {weight_class: [{name, rank, record}]}
        """
        logger.info("Scraping Tapology rankings...")
        results = {}

        try:
            soup = self._get(f"{self.BASE}/rankings")
        except Exception as e:
            logger.error("Failed to load Tapology rankings: %s", e)
            return results

        # Each ranking section has a weight class header
        sections = soup.select("div.rankingItemsContainer")
        for section in sections:
            header = section.select_one("h2, h3, .rankingDivision")
            if not header:
                continue
            wc_text = header.get_text(strip=True)
            wc = _match_weight_class(wc_text)
            if not wc:
                continue

            fighters = []
            for item in section.select("li.rankingItem, tr.rankingRow"):
                rank_el = item.select_one(".rank, .rankingNumber, td:first-child")
                name_el = item.select_one("a.name, .fighterName, td:nth-child(2) a")
                if not name_el:
                    continue
                rank_str = rank_el.get_text(strip=True) if rank_el else "0"
                rank = _safe_int(re.sub(r"\D", "", rank_str))
                fighters.append({
                    "name": name_el.get_text(strip=True),
                    "rank": rank,
                    "tapology_url": self.BASE + name_el.get("href", ""),
                })
            if fighters:
                results[wc] = fighters
                logger.info("Rankings scraped for %s: %d fighters", wc, len(fighters))

        return results

    def update_fighter_rankings(self, rankings: dict[str, list[dict]]):
        """Persist scraped rankings to DB."""
        for weight_class, fighters in rankings.items():
            for fighter in fighters:
                fighter_id = self.db.get_fighter_id(fighter["name"])
                if not fighter_id:
                    # Create a stub fighter entry
                    fighter_id = self.db.upsert_fighter({
                        "name": fighter["name"],
                        "weight_class": weight_class,
                        "tapology_url": fighter.get("tapology_url", ""),
                    })
                # Update ranking
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE fighters SET ranking=?, weight_class=? WHERE id=?",
                        (fighter["rank"], weight_class, fighter_id),
                    )
        logger.info("Rankings updated in DB.")

    def scrape_fighter_page(self, tapology_url: str) -> dict:
        """
        Scrape a Tapology fighter page for supplementary data:
        - Pro record
        - Gym affiliation
        - Country flag
        - Win streak
        """
        result = {"tapology_url": tapology_url}
        try:
            soup = self._get(tapology_url)
        except Exception as e:
            logger.warning("Tapology fighter page failed: %s", e)
            return result

        # Win streak signal
        streak_el = soup.select_one(".winStreak, [data-win-streak]")
        if streak_el:
            result["win_streak"] = _safe_int(streak_el.get_text(strip=True))

        return result


def _match_weight_class(text: str) -> str | None:
    text_lower = text.lower()
    for wc in WEIGHT_CLASSES:
        if wc.lower() in text_lower:
            return wc
    return None


def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
