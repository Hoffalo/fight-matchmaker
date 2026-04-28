"""
models/__init__.py

Heavy modules (torch — FightQualityNN, training, matchmaker) load lazily so that
lightweight imports work without PyTorch:

    from models.data_loader import get_canonical_splits  # OK without torch
    from models.feature_engineering import build_full_matchup_vector

Use lazy attributes for training / NN / matchmaker:

    from models import Matchmaker, FightQualityNN, train
"""

from __future__ import annotations

import importlib
from typing import Any

from .feature_engineering import (
    build_full_matchup_vector,
    build_matchup_vector,
    compute_fight_quality_score,
    compute_fighter_style_metrics,
    extract_fighter_features,
    extract_matchup_features,
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FightQualityNN": ("fight_quality_nn", "FightQualityNN"),
    "Matchmaker": ("matchmaker", "Matchmaker"),
    "HeuristicMatchmaker": ("matchmaker", "HeuristicMatchmaker"),
    "MatchupResult": ("matchmaker", "MatchupResult"),
    "FighterProfile": ("matchmaker", "FighterProfile"),
    "train": ("training", "train"),
    "load_model": ("training", "load_model"),
    "load_scaler": ("training", "load_scaler"),
    "evaluate": ("training", "evaluate"),
}

__all__ = [
    "FightQualityNN",
    "Matchmaker",
    "HeuristicMatchmaker",
    "MatchupResult",
    "FighterProfile",
    "train",
    "load_model",
    "load_scaler",
    "evaluate",
    "build_full_matchup_vector",
    "build_matchup_vector",
    "extract_fighter_features",
    "extract_matchup_features",
    "compute_fight_quality_score",
    "compute_fighter_style_metrics",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        mod_name, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {x for x in globals() if not x.startswith("_")})
