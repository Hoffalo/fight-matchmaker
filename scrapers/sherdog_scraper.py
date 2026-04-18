"""
scrapers/sherdog_scraper.py
Plain HTTP + BS4 scraper for Sherdog (no Selenium — pages are server-rendered).

Returns the 5 canonical fields used by the MMA DB build:
    name, record, country, weight_class, last_fight_date
plus sherdog_url for cross-referencing.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urljoin, quote_plus

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from utils.rate_limiter import RateLimiter
from config import SCRAPING, WEIGHT_CLASSES

logger = logging.getLogger(__name__)


class SherdogScraper:
    BASE = "https://www.sherdog.com"
    SEARCH = BASE + "/stats/fightfinder?SearchTxt={q}"

    def __init__(self, requests_per_minute: int = 20):
        self.rate = RateLimiter(requests_per_minute=requests_per_minute)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SCRAPING["user_agent"],
            "Accept-Language": "en-US,en;q=0.9",
        })

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def _get(self, url: str) -> BeautifulSoup:
        self.rate.wait()
        r = self.session.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")

    def find_fighter_url(self, name: str) -> Optional[str]:
        """Search Sherdog's FightFinder and return the top result URL (or None)."""
        try:
            soup = self._get(self.SEARCH.format(q=quote_plus(name)))
        except Exception as e:
            logger.warning("Sherdog search failed for '%s': %s", name, e)
            return None

        # Search results table: first row of .fightfinder_result or table.fightfinder_result
        row = soup.select_one("table.fightfinder_result tr:nth-of-type(2) a, .fightfinder_result a[href^='/fighter/']")
        if row and row.get("href"):
            return urljoin(self.BASE, row["href"])

        # Fallback: any fighter link on the page
        any_link = soup.select_one("a[href^='/fighter/']")
        return urljoin(self.BASE, any_link["href"]) if any_link else None

    def scrape_fighter(self, name: str) -> dict:
        """Search by name, then scrape the matched profile page."""
        result = {"source": "sherdog", "query_name": name}
        url = self.find_fighter_url(name)
        if not url:
            logger.info("Sherdog: no match for '%s'", name)
            return result
        result["sherdog_url"] = url

        try:
            soup = self._get(url)
        except Exception as e:
            logger.warning("Sherdog profile fetch failed for %s: %s", url, e)
            return result

        # Name
        name_el = soup.select_one("span.fn, h1.fighter-title span.fn, h1 span[itemprop='name']")
        if name_el:
            result["name"] = name_el.get_text(strip=True)

        # Country
        country_el = soup.select_one("[itemprop='nationality']")
        if country_el:
            result["country"] = country_el.get_text(strip=True)

        # Weight class — .association-class text like "ASSOCIATION ... CLASS Lightweight"
        assoc_el = soup.select_one(".association-class")
        if assoc_el:
            m = re.search(r"CLASS\s+([A-Za-z ]+)", assoc_el.get_text(" ", strip=True))
            if m:
                wc_text = m.group(1).strip()
                result["weight_class"] = _match_weight_class(wc_text) or wc_text

        # Record — first integer in each of .wins / .loses / .draws blocks
        wins = _first_int(soup.select_one(".wins"))
        losses = _first_int(soup.select_one(".loses"))
        draws = _first_int(soup.select_one(".draws"))
        if wins is not None or losses is not None:
            result["record"] = f"{wins or 0}-{losses or 0}-{draws or 0}"

        # Last fight date — first data row of fight history table, 3rd cell
        # Date format seen: "Nov / 15 / 2025" alongside event name
        history_rows = soup.select(".new_table.fighter tr, section.fighter_history table tr, .fight_history tr")
        for row in history_rows[1:2]:  # first data row only
            cells = row.select("td")
            if len(cells) >= 3:
                txt = cells[2].get_text(" ", strip=True)
                result["last_fight_date"] = _extract_date(txt) or ""
            break

        return result


def _first_int(el) -> Optional[int]:
    if not el:
        return None
    m = re.search(r"\d+", el.get_text(" ", strip=True))
    return int(m.group(0)) if m else None


def _extract_date(text: str) -> Optional[str]:
    """Handle 'Nov / 15 / 2025', 'Nov 15, 2025', '2025-11-15'. Returns normalized YYYY-MM-DD."""
    if not text:
        return None
    MONTHS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    # "Nov / 15 / 2025" or "Nov 15, 2025" or "Nov 15 2025"
    m = re.search(r"([A-Za-z]{3,9})\s*[/, ]\s*(\d{1,2})\s*[/, ]\s*(\d{4})", text)
    if m:
        mon = MONTHS.get(m.group(1)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return m.group(0)
    return None


def _match_weight_class(text: str) -> Optional[str]:
    t = (text or "").lower()
    for wc in WEIGHT_CLASSES:
        if wc.lower() in t:
            return wc
    return None
