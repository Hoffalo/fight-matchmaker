"""
scrapers/ufc_stats_scraper.py
Selenium scraper for UFCStats.com — the canonical source for per-fight
strike/takedown/control stats with full round-by-round breakdowns.
"""
import re
import logging
import time
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.driver import create_driver
from utils.rate_limiter import polite_delay, RateLimiter
from data.db import Database
from config import URLS

logger = logging.getLogger(__name__)


def parse_time_to_seconds(time_str: str, round_num: int) -> int:
    """Convert fight time like '4:32' in round 3 to total seconds."""
    if not time_str or ":" not in time_str:
        return 0
    try:
        mins, secs = time_str.strip().split(":")
        round_seconds = (round_num - 1) * 300 + int(mins) * 60 + int(secs)
        return round_seconds
    except (ValueError, AttributeError):
        return 0


def parse_stat(val: str) -> int:
    """Parse '45 of 120' → landed=45, attempted=120. Returns tuple."""
    if not val:
        return 0, 0
    val = val.strip()
    m = re.match(r"(\d+)\s+of\s+(\d+)", val)
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        return int(val), 0
    except ValueError:
        return 0, 0


def parse_pct(val: str) -> float:
    """Parse '67%' → 0.67"""
    if not val:
        return 0.0
    val = val.strip().replace("%", "")
    try:
        return float(val) / 100.0
    except ValueError:
        return 0.0


class UFCStatsScraper:
    """
    Scrapes UFCStats.com for:
    - All events (with pagination)
    - Per-fight totals (strikes, takedowns, control time)
    - Per-fighter career stats
    """

    BASE = "http://ufcstats.com"

    def __init__(self, db: Database, headless: bool = True):
        self.db = db
        self.headless = headless
        self.driver = None
        self.rate_limiter = RateLimiter(requests_per_minute=30)

    def __enter__(self):
        self.driver = create_driver(headless=self.headless)
        return self

    def __exit__(self, *_):
        if self.driver:
            self.driver.quit()

    def _get(self, url: str) -> BeautifulSoup:
        self.rate_limiter.wait()
        self.driver.get(url)
        time.sleep(0.5)
        return BeautifulSoup(self.driver.page_source, "lxml")

    # ── Events ────────────────────────────────────────────────────────────────

    def scrape_all_events(self, max_events: int = None) -> list[dict]:
        """
        Scrape the full completed events list from UFCStats.
        Returns list of {name, date, url}.
        """
        logger.info("Scraping event list from UFCStats...")
        soup = self._get(URLS["ufcstats_events"])

        rows = soup.select("tr.b-statistics__table-row")
        events = []
        for row in rows:
            cells = row.select("td")
            if len(cells) < 2:
                continue
            link = cells[0].select_one("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            url  = link.get("href", "")
            # Date lives in span.b-statistics__date inside cells[0]; cells[1] is location.
            date_span = cells[0].select_one("span.b-statistics__date")
            date_cell = date_span.get_text(strip=True) if date_span else ""
            if name and url:
                events.append({"name": name, "date": date_cell, "url": url})

        if max_events:
            events = events[:max_events]

        logger.info("Found %d events", len(events))
        return events

    # ── Single Event ─────────────────────────────────────────────────────────

    def scrape_event(self, event_url: str) -> dict:
        """
        Scrape a single event page, returning event metadata + list of fight URLs.
        """
        soup = self._get(event_url)

        name = soup.select_one("span.b-content__title-highlight")
        name = name.get_text(strip=True) if name else "Unknown Event"

        details = {}
        for li in soup.select("li.b-list__box-list-item"):
            text = li.get_text(separator="|", strip=True)
            if "|" in text:
                k, v = text.split("|", 1)
                details[k.strip().lower()] = v.strip()

        fight_links = []
        for row in soup.select("tr.b-fight-details__table-row"):
            link = row.get("data-link") or ""
            if link and "/fight-details/" in link:
                fight_links.append(link)

        return {
            "name": name,
            "date": details.get("date", ""),
            "location": details.get("location", ""),
            "venue": details.get("location", "").split(",")[0],
            "ufcstats_url": event_url,
            "fight_urls": fight_links,
        }

    # ── Single Fight ─────────────────────────────────────────────────────────

    def scrape_fight(self, fight_url: str) -> Optional[dict]:
        """
        Scrape a fight detail page. Returns dict with:
          - fight metadata (method, round, time, weight class)
          - fighter1_stats, fighter2_stats (per-fighter stat dicts)
        """
        try:
            soup = self._get(fight_url)
        except Exception as e:
            logger.warning("Failed to load fight %s: %s", fight_url, e)
            return None

        result = {"ufcstats_url": fight_url, "fighter1_stats": {}, "fighter2_stats": {}}

        # ── Fighter names and result ──────────────────────────────────────
        name_tags = soup.select("a.b-fight-details__person-link")
        if len(name_tags) >= 2:
            result["fighter1_name"] = name_tags[0].get_text(strip=True)
            result["fighter2_name"] = name_tags[1].get_text(strip=True)

        status_tags = soup.select("i.b-fight-details__person-status")
        if len(status_tags) >= 2:
            result["fighter1_result"] = status_tags[0].get_text(strip=True)
            result["fighter2_result"] = status_tags[1].get_text(strip=True)

        # ── Fight metadata ────────────────────────────────────────────────
        for li in soup.select("li.b-fight-details__text-item"):
            text = li.get_text(separator="|", strip=True)
            if "|" not in text:
                continue
            k, v = text.split("|", 1)
            k = k.strip().lower()
            v = v.strip()
            if "method" in k:
                result["method"] = v
            elif "round" in k:
                try:
                    result["round"] = int(v)
                except ValueError:
                    result["round"] = 0
            elif "time:" in k:
                result["time"] = v
            elif "time format" in k:
                result["time_format"] = v
            elif "referee" in k:
                result["referee"] = v

        wc_tag = soup.select_one("i.b-fight-details__fight-title")
        if wc_tag:
            wc_text = wc_tag.get_text(strip=True)
            result["weight_class"] = wc_text
            result["is_title_fight"] = int("title" in wc_text.lower())

        # ── Per-fighter totals ────────────────────────────────────────────
        totals_table = soup.select("table.b-fight-details__table")[0] if soup.select("table.b-fight-details__table") else None
        if totals_table:
            rows = totals_table.select("tr.b-fight-details__table-row")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if len(cells) < 10:
                    continue
                # Cells order: Fighter | KD | Sig.str. | Sig.str.% | Str. | TD | TD% | Sub.att | Rev. | Ctrl
                # Each cell has two values (one per fighter), separated by newline or space
                def split_cell(cell_text):
                    parts = re.split(r"\s{2,}|\n", cell_text.strip())
                    parts = [p.strip() for p in parts if p.strip()]
                    return parts[0] if parts else "", parts[1] if len(parts) > 1 else ""

                for fighter_idx, stat_dict_key in [(0, "fighter1_stats"), (1, "fighter2_stats")]:
                    stats = result[stat_dict_key]
                    try:
                        kd_vals = split_cell(cells[1])
                        stats["knockdowns"] = int(kd_vals[fighter_idx]) if kd_vals[fighter_idx].isdigit() else 0

                        sig_vals = split_cell(cells[2])
                        landed, att = parse_stat(sig_vals[fighter_idx] if fighter_idx < len(sig_vals) else "")
                        stats["sig_strikes_landed"] = landed
                        stats["sig_strikes_att"] = att

                        str_vals = split_cell(cells[4])
                        landed_t, att_t = parse_stat(str_vals[fighter_idx] if fighter_idx < len(str_vals) else "")
                        stats["total_strikes_landed"] = landed_t
                        stats["total_strikes_att"] = att_t

                        td_vals = split_cell(cells[5])
                        landed_td, att_td = parse_stat(td_vals[fighter_idx] if fighter_idx < len(td_vals) else "")
                        stats["td_landed"] = landed_td
                        stats["td_att"] = att_td

                        sub_vals = split_cell(cells[7])
                        stats["sub_attempts"] = int(sub_vals[fighter_idx]) if fighter_idx < len(sub_vals) and sub_vals[fighter_idx].isdigit() else 0

                        rev_vals = split_cell(cells[8])
                        stats["reversals"] = int(rev_vals[fighter_idx]) if fighter_idx < len(rev_vals) and rev_vals[fighter_idx].isdigit() else 0

                        ctrl_vals = split_cell(cells[9])
                        ctrl_str = ctrl_vals[fighter_idx] if fighter_idx < len(ctrl_vals) else "0:00"
                        ctrl_parts = ctrl_str.split(":")
                        if len(ctrl_parts) == 2:
                            try:
                                stats["ctrl_time_sec"] = int(ctrl_parts[0]) * 60 + int(ctrl_parts[1])
                            except ValueError:
                                stats["ctrl_time_sec"] = 0
                    except (IndexError, ValueError):
                        pass

        # ── Significant strikes by target/position ────────────────────────
        tables = soup.select("table.b-fight-details__table")
        if len(tables) >= 2:
            sig_table = tables[1]
            rows = sig_table.select("tr.b-fight-details__table-row")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if len(cells) < 9:
                    continue
                def split2(c):
                    parts = re.split(r"\s{2,}|\n", c.strip())
                    parts = [p.strip() for p in parts if p.strip()]
                    return parts[0] if parts else "", parts[1] if len(parts) > 1 else ""

                for fi, sk in [(0, "fighter1_stats"), (1, "fighter2_stats")]:
                    stats = result[sk]
                    try:
                        head = split2(cells[3])
                        l, a = parse_stat(head[fi] if fi < len(head) else "")
                        stats["head_landed"], stats["head_att"] = l, a

                        body = split2(cells[4])
                        l, a = parse_stat(body[fi] if fi < len(body) else "")
                        stats["body_landed"], stats["body_att"] = l, a

                        leg = split2(cells[5])
                        l, a = parse_stat(leg[fi] if fi < len(leg) else "")
                        stats["leg_landed"], stats["leg_att"] = l, a

                        dist = split2(cells[6])
                        l, a = parse_stat(dist[fi] if fi < len(dist) else "")
                        stats["distance_landed"], stats["distance_att"] = l, a

                        clinch = split2(cells[7])
                        l, a = parse_stat(clinch[fi] if fi < len(clinch) else "")
                        stats["clinch_landed"], stats["clinch_att"] = l, a

                        ground = split2(cells[8])
                        l, a = parse_stat(ground[fi] if fi < len(ground) else "")
                        stats["ground_landed"], stats["ground_att"] = l, a
                    except (IndexError, ValueError):
                        pass

        # ── Compute fight-level quality signals ───────────────────────────
        total_sig = (
            result["fighter1_stats"].get("sig_strikes_landed", 0)
            + result["fighter2_stats"].get("sig_strikes_landed", 0)
        )
        result["total_sig_strikes"] = total_sig
        result["total_tds"] = (
            result["fighter1_stats"].get("td_landed", 0)
            + result["fighter2_stats"].get("td_landed", 0)
        )
        result["knockdowns"] = (
            result["fighter1_stats"].get("knockdowns", 0)
            + result["fighter2_stats"].get("knockdowns", 0)
        )

        # Sig strike share
        if total_sig > 0:
            for sk in ["fighter1_stats", "fighter2_stats"]:
                result[sk]["sig_strike_share"] = (
                    result[sk].get("sig_strikes_landed", 0) / total_sig
                )
        total_ctrl = (
            result["fighter1_stats"].get("ctrl_time_sec", 0)
            + result["fighter2_stats"].get("ctrl_time_sec", 0)
        )
        if total_ctrl > 0:
            for sk in ["fighter1_stats", "fighter2_stats"]:
                result[sk]["ctrl_share"] = result[sk].get("ctrl_time_sec", 0) / total_ctrl

        total_td = result["total_tds"]
        if total_td > 0:
            for sk in ["fighter1_stats", "fighter2_stats"]:
                result[sk]["td_share"] = result[sk].get("td_landed", 0) / total_td

        # Total time
        rnd = result.get("round", 1)
        time_str = result.get("time", "5:00")
        result["total_time_sec"] = parse_time_to_seconds(time_str, rnd)
        if result["total_time_sec"] > 0 and result["total_time_sec"] <= 1500:
            result["sig_strikes_pm"] = round(total_sig / (result["total_time_sec"] / 60), 2)
        else:
            result["sig_strikes_pm"] = 0.0

        return result

    # ── Fighter career stats page ─────────────────────────────────────────────

    def scrape_fighter_stats(self, fighter_url: str) -> Optional[dict]:
        """Scrape a fighter's career stats page on UFCStats."""
        try:
            soup = self._get(fighter_url)
        except Exception as e:
            logger.warning("Failed to load fighter %s: %s", fighter_url, e)
            return None

        result = {"ufcstats_url": fighter_url}

        name_tag = soup.select_one("span.b-content__title-highlight")
        if name_tag:
            result["name"] = name_tag.get_text(strip=True)

        nickname_tag = soup.select_one("p.b-content__Nickname")
        if nickname_tag:
            result["nickname"] = nickname_tag.get_text(strip=True).strip('"')

        for li in soup.select("li.b-list__box-list-item_type_block"):
            text = li.get_text(separator="|", strip=True)
            if "|" not in text:
                continue
            k, v = text.split("|", 1)
            k, v = k.strip().lower(), v.strip()
            if "height" in k:
                result["height_raw"] = v
            elif "weight" in k:
                result["weight_lbs"] = _parse_weight(v)
            elif "reach" in k:
                result["reach_raw"] = v
            elif "stance" in k:
                result["stance"] = v
            elif "dob" in k or "date of birth" in k:
                result["birthdate"] = v

        # Career stats box
        for li in soup.select("li.b-list__box-list-item"):
            text = li.get_text(separator="|", strip=True)
            if "|" not in text:
                continue
            k, v = text.split("|", 1)
            k, v = k.strip().lower(), v.strip()
            if "slpm" in k or "str. landed" in k:
                result["sig_strikes_pm"] = _parse_float(v)
            elif "str. acc" in k:
                result["sig_strike_acc"] = parse_pct(v)
            elif "sapm" in k or "str. absorbed" in k:
                result["sig_strikes_abs_pm"] = _parse_float(v)
            elif "str. def" in k:
                result["sig_strike_def"] = parse_pct(v)
            elif "td avg" in k:
                result["td_avg"] = _parse_float(v)
            elif "td acc" in k:
                result["td_acc"] = parse_pct(v)
            elif "td def" in k:
                result["td_def"] = parse_pct(v)
            elif "sub. avg" in k:
                result["sub_avg"] = _parse_float(v)

        result["height_cm"] = _parse_height_to_cm(result.get("height_raw", ""))
        result["reach_cm"] = _parse_reach_to_cm(result.get("reach_raw", ""))

        return result

    # ── All fighters (alphabetical) ───────────────────────────────────────────

    def scrape_all_fighter_urls(self) -> list[str]:
        """Collect all fighter page URLs from UFCStats alphabetical index."""
        urls = []
        for letter in "abcdefghijklmnopqrstuvwxyz":
            page_url = URLS["ufcstats_fighters"].format(letter=letter)
            logger.info("Scraping fighter index: %s", letter.upper())
            try:
                soup = self._get(page_url)
                for a in soup.select("a.b-link_style_black"):
                    href = a.get("href", "")
                    if "/fighter-details/" in href:
                        urls.append(href)
            except Exception as e:
                logger.warning("Failed index page %s: %s", letter, e)
        logger.info("Total fighter URLs collected: %d", len(urls))
        return list(set(urls))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_float(val: str) -> float:
    try:
        return float(val.replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _parse_weight(val: str) -> float:
    """'155 lbs.' → 155.0"""
    m = re.search(r"([\d.]+)", val or "")
    return float(m.group(1)) if m else 0.0


def _parse_height_to_cm(val: str) -> float:
    """'5' 11"' → 180.3"""
    m = re.match(r"(\d+)'\s*(\d+)", val or "")
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return round((feet * 12 + inches) * 2.54, 1)
    return 0.0


def _parse_reach_to_cm(val: str) -> float:
    """'72.0"' → 182.9"""
    m = re.search(r"([\d.]+)", val or "")
    if m:
        inches = float(m.group(1))
        return round(inches * 2.54, 1)
    return 0.0
