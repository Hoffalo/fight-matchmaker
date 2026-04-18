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
        Returns {weight_class: [{name, rank, tapology_url}]}

        Tapology /rankings uses Tailwind utility classes and has no semantic
        anchor. Strategy: find a <div> whose text is exactly a weight class
        name (division header), then walk up to the ancestor that contains
        fighter links and extract them in order.
        """
        logger.info("Scraping Tapology rankings...")
        results: dict[str, list[dict]] = {}

        try:
            soup = self._get(f"{self.BASE}/rankings")
        except Exception as e:
            logger.error("Failed to load Tapology rankings: %s", e)
            return results

        for wc in WEIGHT_CLASSES:
            header = soup.find(
                lambda t: t.name in ("div", "span", "h2", "h3")
                and t.get_text(strip=True) == wc
            )
            if not header:
                continue

            # Walk up until we find an ancestor containing at least 3 fighter links
            section = header
            for _ in range(6):
                section = section.parent
                if not section:
                    break
                links = section.select('a[href*="/fightcenter/fighters/"]')
                if len(links) >= 3:
                    break
            else:
                continue

            fighters = []
            seen = set()
            for rank, a in enumerate(links, start=1):
                name = a.get_text(strip=True)
                href = a.get("href", "")
                if not name or not href or name in seen:
                    continue
                seen.add(name)
                fighters.append({
                    "name": name,
                    "rank": rank,
                    "tapology_url": self.BASE + href if href.startswith("/") else href,
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
        Scrape a Tapology fighter page for:
        - name, record, country, weight_class, last_fight_date
        """
        result = {"tapology_url": tapology_url, "source": "tapology"}
        try:
            soup = self._get(tapology_url)
        except Exception as e:
            logger.warning("Tapology fighter page failed: %s", e)
            return result

        name_el = soup.select_one("h1.fighterHeaderName, h1[class*='fighterName'], h1")
        if name_el:
            result["name"] = name_el.get_text(strip=True)

        # Details block: labeled fields like "Pro MMA Record:", "Born:", "Nationality:"
        for li in soup.select("div.details ul li, .detailsTop li, .details_two_columns li"):
            label_el = li.select_one("strong, .label")
            if not label_el:
                continue
            label = label_el.get_text(strip=True).lower().rstrip(":")
            value = li.get_text(" ", strip=True).replace(label_el.get_text(strip=True), "").strip(" :")

            if "pro mma record" in label or label == "record":
                m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", value)
                if m:
                    result["record"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            elif "nationality" in label or label == "born":
                # Nationality often just has country name; "Born" may have city, country
                result["country"] = value.split(",")[-1].strip()
            elif "weight class" in label or "last weigh" in label:
                result["weight_class"] = _match_weight_class(value) or value

        # Country fallback via flag image alt/title
        if not result.get("country"):
            flag = soup.select_one("img.flag, img[class*='flag']")
            if flag:
                result["country"] = flag.get("alt") or flag.get("title") or ""

        # Last fight date: fight history table, first row's date cell
        for row in soup.select("section.fighterFightResults li, table.fcLeaderboard tr, .fighterFightResults tr"):
            date_el = row.select_one("span.result, .date, time, td.date")
            if date_el:
                txt = date_el.get_text(" ", strip=True)
                m = re.search(r"([A-Z][a-z]{2,8}\.?\s+\d{1,2},?\s+\d{4})|(\d{4}-\d{2}-\d{2})", txt)
                if m:
                    result["last_fight_date"] = (m.group(1) or m.group(2)).replace(".", "")
                    break

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
