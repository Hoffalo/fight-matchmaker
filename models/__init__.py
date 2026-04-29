"""
models/__init__.py

Light eager imports (no torch). Heavy modules are loaded lazily on first
attribute access so that ``from models.feature_engineering import ...`` keeps
working without importing torch.

    from models import MatchmakerV2          # lazy — pulls in nn_binary + torch
    from models import get_canonical_splits  # eager — pure-numpy data pipeline
    from models import SELECTED_FEATURES, FEATURE_DIM
"""
from __future__ import annotations

import os

# macOS / Conda: default thread caps before NumPy/XGBoost init to avoid OpenMP
# clashes that manifest as ``segmentation fault`` (see matchmaker_v2 CLI).
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import importlib
from typing import Any

from .feature_engineering import (
    ALL_FEATURE_NAMES,
    build_full_matchup_vector,
    extract_fighter_features,
    extract_matchup_features,
    subset_full_feature_vector,
)
from .data_loader import get_canonical_splits
from .pipeline_config import SELECTED_FEATURES

from config import FEATURE_DIM, SELECTED_FEATURE_DIM, FINAL_MODEL

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Final shipped pipeline
    "MatchmakerV2": ("matchmaker_v2", "MatchmakerV2"),
    "FightBonusNN": ("nn_binary", "FightBonusNN"),
    "load_binary_nn": ("nn_binary", "load_binary_nn"),
    # Reference / legacy (still importable for retraining + presentation)
    "BaselineComparison": ("baselines", "BaselineComparison"),
    "backtest": ("backtesting", "backtest"),
}

__all__ = [
    "MatchmakerV2",
    "FightBonusNN",
    "load_binary_nn",
    "BaselineComparison",
    "backtest",
    "get_canonical_splits",
    "build_full_matchup_vector",
    "extract_fighter_features",
    "extract_matchup_features",
    "subset_full_feature_vector",
    "ALL_FEATURE_NAMES",
    "SELECTED_FEATURES",
    "FEATURE_DIM",
    "SELECTED_FEATURE_DIM",
    "FINAL_MODEL",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        mod_name, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {x for x in globals() if not x.startswith("_")})
