"""
config.py — Central configuration for UFC Matchmaker

Final pipeline:
  fighter stats × 2 → build_full_matchup_vector() [115-dim]
                    → subset_full_feature_vector() [12 RFECV features]
                    → StandardScaler.transform()
                    → FightBonusNN.forward() → sigmoid → P(bonus fight)

The 115-dim full vector is still built end-to-end so the matchmaker reuses the
same feature pipeline as training; ``SELECTED_FEATURES`` in pipeline_config.py
selects the 12 columns the trained model expects.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "ufc_matchmaker.db"

# ── Final shipped model ──────────────────────────────────────────────────────
# The matchmaker (models/matchmaker_v2.py) loads the checkpoint + scaler below.
FINAL_MODEL = {
    "type": "neural_network",
    "checkpoint": str(MODELS_DIR / "checkpoints" / "nn_12feat.pt"),
    "scaler": str(MODELS_DIR / "checkpoints" / "scaler_12feat.pkl"),
    "features": "RFECV 12-dim subset of 115",
    "auc": 0.5991,  # reference val AUC reported during selection sweep
    "params": 257,
    "architecture": "12 -> 16 -> 1 (GELU + BatchNorm + Dropout)",
}

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

# ── Binary classification (temporal train ~338 unique fights, Jan–Aug 2025) ──
# 91 bonus / 247 not → 26.9% positive; same ratio holds after (A,B)/(B,A) augmentation.
TRAIN_UNIQUE_FIGHTS = 338
TRAIN_POSITIVE_FIGHTS = 91
TRAIN_NEGATIVE_FIGHTS = 247
TRAIN_POSITIVE_RATE = TRAIN_POSITIVE_FIGHTS / TRAIN_UNIQUE_FIGHTS  # ≈ 0.269
SCALE_POS_WEIGHT = TRAIN_NEGATIVE_FIGHTS / TRAIN_POSITIVE_FIGHTS  # ≈ 2.714 (XGBoost / diagnostics)

# ── Feature dimensions ────────────────────────────────────────────────────────
# FEATURE_DIM is the *full* engineered vector width. Do not change without also
# updating ALL_FEATURE_NAMES in feature_engineering.py — data_loader.py asserts
# the two stay in sync. The trained classifier consumes a SUBSET of these
# columns (see SELECTED_FEATURES in pipeline_config.py).
FEATURE_DIM = 115
SELECTED_FEATURE_DIM = 12
USE_CROSS_FEATURES = True

# ── Binary classification pipeline ───────────────────────────────────────────
BINARY_CLASSIFICATION_CONFIG = {
    "target": "is_bonus_fight",
    "positive_label": 1,
    "feature_dim": 115,
    "train_unique_fights": TRAIN_UNIQUE_FIGHTS,
    "train_positive_rate": TRAIN_POSITIVE_RATE,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "split_dates": {
        "train_end": "2025-09-01",
        "val_end": "2026-01-01",
    },
    "random_seed": 42,
}

# ── Legacy regression NN config ──────────────────────────────────────────────
# Kept only so models/training.py and models/matchmaker.py (both marked LEGACY)
# can still import. The production model uses BinaryNNConfig in models/nn_binary.py.
NN = {
    "input_dim": 115,
    "hidden_layers": [256, 128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 150,
    "val_split": 0.2,
    "early_stopping_patience": 15,
    "model_save_path": str(MODELS_DIR / "fight_quality_nn.pt"),
    "scaler_save_path": str(MODELS_DIR / "feature_scaler.pkl"),
}

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
