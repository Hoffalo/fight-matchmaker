"""
data/pipeline.py
Full data collection pipeline — orchestrates all scrapers,
deduplicates, and populates the SQLite database.
"""
import logging
from tqdm import tqdm
from typing import Optional

from data.db import Database
from scrapers.ufc_stats_scraper import UFCStatsScraper
from scrapers.ufc_api_wrapper import UFCApiWrapper
from scrapers.tapology_scraper import TapologyScraper
from config import WEIGHT_CLASSES

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates the full data collection workflow:
    1. Scrape all events from UFCStats
    2. Scrape fight details + per-fight stats for each event
    3. Enrich fighter profiles with career stats from UFCStats
    4. Supplement with Sherdog data via ufc-api
    5. Pull rankings from Tapology
    """

    def __init__(self, db: Database, headless: bool = True):
        self.db = db
        self.headless = headless

    def run_full_collection(
        self,
        max_events: int = None,
        max_fighters: int = None,
        skip_rankings: bool = False,
    ):
        """
        Run the complete data collection pipeline.

        Args:
            max_events:    Max events to scrape (None = all)
            max_fighters:  Max fighters to enrich with career stats
            skip_rankings: Skip Tapology rankings scrape
        """
        logger.info("=" * 60)
        logger.info("UFC MATCHMAKER — Data Collection Pipeline")
        logger.info("=" * 60)

        # ── Steps 1+2: Share one Chrome instance for all UFCStats work ───
        logger.info("\n[1/4] Collecting events and fights from UFCStats...")
        with UFCStatsScraper(self.db, headless=self.headless) as scraper:
            self._collect_events_and_fights(max_events=max_events, scraper=scraper)
            logger.info("\n[2/4] Enriching fighter profiles with career stats...")
            self._enrich_fighter_stats(max_fighters=max_fighters, scraper=scraper)

        # ── Step 3: Rankings ──────────────────────────────────────────────
        if not skip_rankings:
            logger.info("\n[3/4] Scraping rankings from Tapology...")
            self._collect_rankings()
        else:
            logger.info("\n[3/4] Skipping rankings (--skip-rankings)")

        # ── Step 4: Compute derived metrics ──────────────────────────────
        logger.info("\n[4/4] Computing derived metrics and fight quality scores...")
        self._compute_derived_metrics()

        stats = self.db.get_stats()
        logger.info("\n✓ Pipeline complete!")
        logger.info("  Fighters: %d", stats["fighters"])
        logger.info("  Fights:   %d", stats["fights"])
        logger.info("  Events:   %d", stats["events"])

    def _collect_events_and_fights(self, max_events: int = None, scraper=None):
        events = scraper.scrape_all_events(max_events=max_events)

        if not events:
            logger.warning("scrape_all_events returned 0 events — check URLS['ufcstats_events'] and page selectors")

        for event_meta in tqdm(events, desc="Events"):
            if self.db.event_already_scraped(event_meta["url"]):
                continue
            try:
                event_data = scraper.scrape_event(event_meta["url"])
                fight_urls = event_data.get("fight_urls", [])
                if not fight_urls:
                    logger.debug("No fight URLs found for event %s — selector 'tr.b-fight-details__table-row[data-link]' may have changed", event_meta["url"])
                event_id = self.db.upsert_event({
                    "name":         event_data["name"],
                    "date":         event_data["date"],
                    "location":     event_data.get("location", ""),
                    "venue":        event_data.get("venue", ""),
                    "ufcstats_url": event_data["ufcstats_url"],
                })

                for fight_url in fight_urls:
                    self._process_fight(scraper, fight_url, event_id, event_data["date"])

            except Exception as e:
                logger.warning("Failed to process event %s: %s", event_meta["url"], e)

    def _process_fight(
        self,
        scraper: UFCStatsScraper,
        fight_url: str,
        event_id: int,
        event_date: str,
    ):
        """Scrape a single fight and persist it with both fighters' stats."""
        fight_data = scraper.scrape_fight(fight_url)
        if not fight_data:
            return

        # Ensure both fighters exist in DB
        f1_name = fight_data.get("fighter1_name", "")
        f2_name = fight_data.get("fighter2_name", "")
        if not f1_name or not f2_name:
            return

        f1_id = self.db.get_fighter_id(f1_name) or self.db.upsert_fighter({"name": f1_name})
        f2_id = self.db.get_fighter_id(f2_name) or self.db.upsert_fighter({"name": f2_name})

        # Determine winner
        winner_id = None
        if fight_data.get("fighter1_result", "").lower() == "w":
            winner_id = f1_id
        elif fight_data.get("fighter2_result", "").lower() == "w":
            winner_id = f2_id

        method_full = fight_data.get("method", "")
        method_parts = method_full.split(" - ", 1)
        method = method_parts[0].strip() if method_parts else method_full
        method_detail = method_parts[1].strip() if len(method_parts) > 1 else ""

        fight_id = self.db.upsert_fight({
            "event_id":           event_id,
            "weight_class":       fight_data.get("weight_class", ""),
            "is_title_fight":     fight_data.get("is_title_fight", 0),
            "fighter1_id":        f1_id,
            "fighter2_id":        f2_id,
            "winner_id":          winner_id,
            "method":             method,
            "method_detail":      method_detail,
            "round":              fight_data.get("round", 0),
            "time":               fight_data.get("time", ""),
            "total_time_sec":     fight_data.get("total_time_sec", 0),
            "total_sig_strikes":  fight_data.get("total_sig_strikes", 0),
            "sig_strikes_pm":     fight_data.get("sig_strikes_pm", 0.0),
            "total_tds":          fight_data.get("total_tds", 0),
            "knockdowns":         fight_data.get("knockdowns", 0),
            "ufcstats_url":       fight_url,
        })

        # Per-fighter stats
        for fighter_id, stats_key in [(f1_id, "fighter1_stats"), (f2_id, "fighter2_stats")]:
            stats = fight_data.get(stats_key, {})
            if stats:
                self.db.insert_fight_stats({
                    "fight_id":              fight_id,
                    "fighter_id":            fighter_id,
                    "knockdowns":            stats.get("knockdowns", 0),
                    "sig_strikes_landed":    stats.get("sig_strikes_landed", 0),
                    "sig_strikes_att":       stats.get("sig_strikes_att", 0),
                    "total_strikes_landed":  stats.get("total_strikes_landed", 0),
                    "total_strikes_att":     stats.get("total_strikes_att", 0),
                    "head_landed":           stats.get("head_landed", 0),
                    "head_att":              stats.get("head_att", 0),
                    "body_landed":           stats.get("body_landed", 0),
                    "body_att":              stats.get("body_att", 0),
                    "leg_landed":            stats.get("leg_landed", 0),
                    "leg_att":               stats.get("leg_att", 0),
                    "distance_landed":       stats.get("distance_landed", 0),
                    "distance_att":          stats.get("distance_att", 0),
                    "clinch_landed":         stats.get("clinch_landed", 0),
                    "clinch_att":            stats.get("clinch_att", 0),
                    "ground_landed":         stats.get("ground_landed", 0),
                    "ground_att":            stats.get("ground_att", 0),
                    "td_landed":             stats.get("td_landed", 0),
                    "td_att":                stats.get("td_att", 0),
                    "sub_attempts":          stats.get("sub_attempts", 0),
                    "reversals":             stats.get("reversals", 0),
                    "ctrl_time_sec":         stats.get("ctrl_time_sec", 0),
                    "sig_strike_share":      stats.get("sig_strike_share", 0.5),
                    "td_share":              stats.get("td_share", 0.5),
                    "ctrl_share":            stats.get("ctrl_share", 0.5),
                })

    def _enrich_fighter_stats(self, max_fighters: int = None, scraper=None):
        """Pull career stats from UFCStats fighter pages for all DB fighters."""
        all_urls = scraper.scrape_all_fighter_urls()
        if not all_urls:
            logger.warning("scrape_all_fighter_urls returned 0 URLs — selector 'a.b-link_style_black' may have changed on UFCStats")
        if max_fighters:
            all_urls = all_urls[:max_fighters]

        for url in tqdm(all_urls, desc="Fighter stats"):
            if self.db.fighter_url_already_scraped(url):
                continue
            try:
                stats = scraper.scrape_fighter_stats(url)
                if stats and stats.get("name"):
                    self.db.upsert_fighter(stats)
            except Exception as e:
                logger.debug("Fighter stats enrichment failed for %s: %s", url, e)

    def _collect_rankings(self):
        """Pull rankings from Tapology and update DB."""
        with TapologyScraper(self.db, headless=self.headless) as scraper:
            rankings = scraper.scrape_rankings()
            if rankings:
                scraper.update_fighter_rankings(rankings)

    def _compute_derived_metrics(self):
        """
        Compute and store derived per-fighter metrics:
        - grapple_ratio
        - finish_rate
        - fight quality scores for all historical fights
        """
        from models.feature_engineering import compute_fighter_style_metrics

        fighters = self.db.get_all_fighters()
        for fighter in tqdm(fighters, desc="Computing metrics"):
            metrics = compute_fighter_style_metrics(fighter, self.db)
            if metrics:
                with self.db.connect() as conn:
                    placeholders = ", ".join([f"{k}=?" for k in metrics.keys()])
                    conn.execute(
                        f"UPDATE fighters SET {placeholders} WHERE id=?",
                        list(metrics.values()) + [fighter["id"]],
                    )

        # Score all historical fights
        from models.feature_engineering import compute_fight_quality_score
        fights = self.db.get_all_fights()
        for fight in tqdm(fights, desc="Scoring fights"):
            score = compute_fight_quality_score(fight, self.db)
            if score is not None:
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE fights SET fight_quality_score=? WHERE id=?",
                        (score, fight["id"]),
                    )
