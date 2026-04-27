"""
config.py — Central configuration for UFC Matchmaker
"""
# Configuration for UFC Fight Matchmaker
# Pipeline: 72-dim features → binary classification (bonus fight prediction) → matchmaker ranking
# Legacy 48-dim regression pipeline has been deprecated

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "ufc_matchmaker.db"

# ── Scraping ──────────────────────────────────────────────────────────────────
SCRAPING = {
    "headless": True,               # Run Chrome headless
    "page_load_timeout": 30,
    "implicit_wait": 5,
    "request_delay_min": 0.5,       # Seconds between requests (polite)
    "request_delay_max": 1.5,
    "max_retries": 3,
    "retry_delay": 5,
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

URLS = {
    "ufcstats_base":   "http://ufcstats.com",
    "ufcstats_events": "http://ufcstats.com/statistics/events/completed?page=all",
    "ufcstats_fighters":"http://ufcstats.com/statistics/fighters?char={letter}&page=all",
    "tapology_base":   "https://www.tapology.com",
    "tapology_fighters":"https://www.tapology.com/rankings",
    "ufc_rankings":    "https://www.ufc.com/rankings",
}

# ── Feature dimensions ────────────────────────────────────────────────────────
FEATURE_DIM = 72
USE_CROSS_FEATURES = True

# ── Binary classification pipeline ───────────────────────────────────────────
BINARY_CLASSIFICATION_CONFIG = {
    "target": "is_bonus_fight",
    "positive_label": 1,
    "feature_dim": 72,
    "split_dates": {
        "train_end": "2025-09-01",
        "val_end": "2026-01-01",
    },
    "random_seed": 42,
}

# ── Neural Network (legacy regression model — kept for backward compat) ──────
NN = {
    "input_dim": 72,
    "hidden_layers": [256, 128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 150,
    "val_split": 0.2,
    "early_stopping_patience": 15,
    "model_save_path": str(MODELS_DIR / "fight_bonus_classifier.pt"),
    "scaler_save_path": str(MODELS_DIR / "feature_scaler.pkl"),
}

# Removed: heuristic scoring replaced by ML-based binary classification

# ── Weight Classes ────────────────────────────────────────────────────────────
WEIGHT_CLASSES = [
    "Strawweight",        # 115 lbs
    "Flyweight",          # 125 lbs
    "Bantamweight",       # 135 lbs
    "Featherweight",      # 145 lbs
    "Lightweight",        # 155 lbs
    "Welterweight",       # 170 lbs
    "Middleweight",       # 185 lbs
    "Light Heavyweight",  # 205 lbs
    "Heavyweight",        # 265 lbs
]

WEIGHT_CLASS_LIMITS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}
