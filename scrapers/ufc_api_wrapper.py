"""
scrapers/ufc_api_wrapper.py
Wrapper around FritzCapuyan/ufc-api for fighter + event data from Sherdog.
Adds error handling, normalization, and DB persistence.
"""
import logging
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from data.db import Database

logger = logging.getLogger(__name__)

# Lazy import — ufc_api may not be installed in all envs
try:
    from ufc import get_fighter, get_event
    UFC_API_AVAILABLE = True
except ImportError:
    UFC_API_AVAILABLE = False
    logger.warning("ufc_api not installed. Run: pip install ufc_api")


class UFCApiWrapper:
    """
    Wraps the FritzCapuyan ufc-api to pull fighter and event data from Sherdog,
    normalizes it, and persists to the local DB.
    """

    def __init__(self, db: Database):
        self.db = db
        if not UFC_API_AVAILABLE:
            raise RuntimeError(
                "ufc_api package not found. Install with: pip install ufc_api"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_fighter(self, name: str) -> Optional[dict]:
        """
        Fetch fighter data from Sherdog via ufc-api.
        Returns normalized dict ready for DB insertion.
        """
        try:
            raw = get_fighter(name)
        except Exception as e:
            logger.warning("ufc-api get_fighter('%s') failed: %s", name, e)
            return None

        if not raw or "name" not in raw:
            return None

        wins = raw.get("wins", {})
        losses = raw.get("losses", {})
        fights = raw.get("fights", [])

        wins_ko   = _safe_int(wins.get("ko/tko", 0))
        wins_sub  = _safe_int(wins.get("submissions", 0))
        wins_dec  = _safe_int(wins.get("decisions", 0))
        wins_tot  = _safe_int(wins.get("total", 0))
        losses_ko  = _safe_int(losses.get("ko/tko", 0))
        losses_sub = _safe_int(losses.get("submissions", 0))
        losses_dec = _safe_int(losses.get("decisions", 0))
        losses_tot = _safe_int(losses.get("total", 0))

        # Derived style metrics
        ko_rate     = wins_ko  / wins_tot  if wins_tot > 0 else 0.0
        sub_rate    = wins_sub / wins_tot  if wins_tot > 0 else 0.0
        dec_rate    = wins_dec / wins_tot  if wins_tot > 0 else 0.0
        finish_rate = (wins_ko + wins_sub) / wins_tot if wins_tot > 0 else 0.0

        normalized = {
            "name":         raw.get("name", "").strip(),
            "nickname":     raw.get("nickname", ""),
            "nationality":  raw.get("nationality", ""),
            "birthdate":    raw.get("birthdate", ""),
            "weight_class": _normalize_weight_class(raw.get("weight_class", "")),
            "wins_total":   wins_tot,
            "wins_ko":      wins_ko,
            "wins_sub":     wins_sub,
            "wins_dec":     wins_dec,
            "losses_total": losses_tot,
            "losses_ko":    losses_ko,
            "losses_sub":   losses_sub,
            "losses_dec":   losses_dec,
            "ko_rate":      ko_rate,
            "sub_rate":     sub_rate,
            "dec_rate":     dec_rate,
            "finish_rate":  finish_rate,
        }

        # Parse height/weight if present
        height_raw = raw.get("height", "")
        weight_raw = raw.get("weight", "")
        normalized["height_cm"] = _parse_height(height_raw)
        normalized["weight_lbs"] = _parse_weight(weight_raw)

        return normalized

    def fetch_event(self, event_name: str) -> Optional[dict]:
        """
        Fetch event data from Sherdog via ufc-api.
        Returns normalized dict with fight list.
        """
        try:
            raw = get_event(event_name)
        except Exception as e:
            logger.warning("ufc-api get_event('%s') failed: %s", event_name, e)
            return None

        if not raw:
            return None

        fights = []
        for fight in raw.get("fights", []):
            red = fight.get("red corner", {})
            blue = fight.get("blue corner", {})
            fights.append({
                "weight_class":    _normalize_weight_class(fight.get("weightclass", "")),
                "fighter1_name":   red.get("name", ""),
                "fighter2_name":   blue.get("name", ""),
                "fighter1_result": red.get("result", ""),
                "fighter2_result": blue.get("result", ""),
                "fighter1_odds":   red.get("odds", ""),
                "fighter2_odds":   blue.get("odds", ""),
                "round":           _safe_int(fight.get("round", 0)),
                "time":            fight.get("time", ""),
                "method":          fight.get("method", ""),
            })

        return {
            "name":     raw.get("name", ""),
            "date":     raw.get("date", ""),
            "location": raw.get("location", ""),
            "venue":    raw.get("venue", ""),
            "fights":   fights,
        }

    def bulk_fetch_fighters(self, names: list[str]) -> list[dict]:
        """Fetch multiple fighters, returning successfully parsed ones."""
        results = []
        for name in names:
            logger.info("Fetching fighter: %s", name)
            data = self.fetch_fighter(name)
            if data:
                fid = self.db.upsert_fighter(data)
                data["id"] = fid
                results.append(data)
        return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _parse_height(val: str) -> float:
    """'6\\'4"' → 193.0 cm"""
    m = re.search(r"(\d+)'\\?\"?(\d+)", val or "")
    if not m:
        m = re.search(r"(\d+)\s*'\s*(\d+)", val or "")
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return round((feet * 12 + inches) * 2.54, 1)
    return 0.0


def _parse_weight(val: str) -> float:
    m = re.search(r"([\d.]+)", val or "")
    return float(m.group(1)) if m else 0.0


def _normalize_weight_class(raw: str) -> str:
    """Normalize variant spellings to canonical weight class names."""
    raw = (raw or "").strip().lower()
    mapping = {
        "strawweight":        "Strawweight",
        "atomweight":         "Strawweight",
        "flyweight":          "Flyweight",
        "bantamweight":       "Bantamweight",
        "featherweight":      "Featherweight",
        "lightweight":        "Lightweight",
        "welterweight":       "Welterweight",
        "middleweight":       "Middleweight",
        "light heavyweight":  "Light Heavyweight",
        "light heavy":        "Light Heavyweight",
        "heavyweight":        "Heavyweight",
        "super heavyweight":  "Heavyweight",
        "catch weight":       "Catchweight",
        "catchweight":        "Catchweight",
        "open weight":        "Open Weight",
    }
    for key, canonical in mapping.items():
        if key in raw:
            return canonical
    return raw.title()
