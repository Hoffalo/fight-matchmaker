"""
Training / inference pipeline configuration.

``SELECTED_FEATURES`` restricts ``get_canonical_splits()`` and matchmaker scoring to a
subset of columns (names in ``models.feature_engineering.ALL_FEATURE_NAMES``).

- ``None`` → full ``config.FEATURE_DIM`` (115) columns.
- Non-empty list → **retrain** saved models / scaler for that width; see
  ``models.matchmaker_v2`` (symmetric vector applies the same subset).

Default below is **RFECV-optimal** on the 115-dim vector (TimeSeriesSplit-5, LogReg,
~338 unique train fights, 26.9% positive). Run ``OMP_NUM_THREADS=1 python -m models.feature_selection``
to refresh.

``config.SCALE_POS_WEIGHT`` (≈2.71) matches 247/91 class ratio on unique train fights.
"""
from __future__ import annotations

from typing import Optional

# RFECV optimal n=12 (115-dim feature_selection run). Set to None to use all 115 features.
RFECV_OPTIMAL_115: tuple[str, ...] = (
    "style_clash_score",
    "is_five_rounder",
    "f1_roll_recent_knockdowns_norm",
    "f1_roll_performance_consistency",
    "f1_roll_strike_trend_norm",
    "f1_roll_damage_trend_norm",
    "f2_roll_recent_knockdowns_norm",
    "f2_roll_performance_consistency",
    "f2_roll_strike_trend_norm",
    "f2_roll_damage_trend_norm",
    "variance_clash",
    "recent_output_combined",
)

SELECTED_FEATURES: Optional[list[str]] = list(RFECV_OPTIMAL_115)
