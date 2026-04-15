"""
models/__init__.py
Exports the key model classes so callers can do:

    from models import Matchmaker, HeuristicMatchmaker, FightQualityNN, train
"""
from .fight_quality_nn import FightQualityNN
from .matchmaker import Matchmaker, HeuristicMatchmaker, MatchupResult, FighterProfile
from .training import train, load_model, load_scaler, evaluate
from .feature_engineering import (
    build_matchup_vector,
    extract_fighter_features,
    extract_matchup_features,
    compute_fight_quality_score,
    compute_fighter_style_metrics,
)
