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

# RFECV optimal n=12 on the FULL 115-dim space (LogReg, TimeSeriesSplit-5).
# Retained for reference — DO NOT use for new training. 8 of these 12 are
# rolling-derived and have train_std ≈ 1 / test_std ≈ 6 (cold-start gap), so
# the trained model can't generalize and SHAP shows them at zero.
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

# Distribution-stable RFECV pick: ran feature_selection_clean.py which dropped
# the 38 features with test_std/train_std > 1.5 (rolling + rolling-derived) and
# re-ran RFECV on the surviving 77. CV AUC 0.5922 (vs 0.5832 on the tainted
# 12). Added f1_grapple_ratio for matchmaker symmetry — RFECV picked only the
# f2 side because they're identically distributed after augmentation.
RFECV_OPTIMAL_CLEAN: tuple[str, ...] = (
    "f1_grapple_ratio",
    "f2_grapple_ratio",
    "style_clash_score",
    "is_title_fight",
    "is_five_rounder",
)

SELECTED_FEATURES: Optional[list[str]] = list(RFECV_OPTIMAL_115)
