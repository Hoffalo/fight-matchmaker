"""
utils/driver.py — Selenium WebDriver factory with anti-detection settings
"""
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from config import SCRAPING

logger = logging.getLogger(__name__)


def create_driver(headless: bool = None) -> webdriver.Chrome:
    """
    Create a configured Chrome WebDriver instance.
    Applies anti-detection measures and polite scraping settings.
    """
    if headless is None:
        headless = SCRAPING["headless"]

    options = Options()

    if headless:
        options.add_argument("--headless=new")

    # Core stability flags
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Anti-detection
    options.add_argument(f"--user-agent={SCRAPING['user_agent']}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Performance
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.page_load_strategy = "eager"  # Don't wait for all resources

    # Block images, CSS, and fonts — scraping plain HTML, don't need them
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
    })

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Patch navigator.webdriver to avoid detection
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    driver.set_page_load_timeout(SCRAPING["page_load_timeout"])
    driver.implicitly_wait(SCRAPING["implicit_wait"])

    logger.info("Chrome WebDriver initialized (headless=%s)", headless)
    return driver
