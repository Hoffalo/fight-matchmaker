"""
utils/rate_limiter.py — Polite scraping rate limiter
"""
import time
import random
import logging
from config import SCRAPING

logger = logging.getLogger(__name__)


def polite_delay(min_s: float = None, max_s: float = None):
    """Sleep a random amount to avoid hammering servers."""
    lo = min_s or SCRAPING["request_delay_min"]
    hi = max_s or SCRAPING["request_delay_max"]
    delay = random.uniform(lo, hi)
    logger.debug("Rate limit delay: %.2fs", delay)
    time.sleep(delay)


class RateLimiter:
    """Token-bucket style rate limiter for bulk scraping."""

    def __init__(self, requests_per_minute: int = 20):
        self.min_interval = 60.0 / requests_per_minute
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            jitter = random.uniform(0, 0.5)
            time.sleep(wait_time + jitter)
        self._last_call = time.time()
