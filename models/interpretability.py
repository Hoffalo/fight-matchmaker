"""
models/interpretability.py
SHAP-based model interpretability for the UFC fight entertainment classifier.

Generates presentation-ready plots that explain:
  - Which features matter most globally (beeswarm, bar)
  - Why the matchup cross-features (48-71) are the key differentiator
  - Why a *specific* matchup is predicted as entertaining (waterfall)
  - How different models agree/disagree on feature importance
  - Whether importance is concentrated in matchup dynamics vs individual stats

All plots are 300 DPI with 14pt+ labels on white backgrounds.
"""
import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature schema
# ─────────────────────────────────────────────────────────────────────────────

FIGHTER_A_FEATURES = [
    "A_sig_strike_rate", "A_takedown_accuracy", "A_ko_rate", "A_sub_rate",
    "A_strike_defense", "A_takedown_defense", "A_cardio_index", "A_reach",
    "A_height", "A_age", "A_win_streak", "A_loss_streak", "A_fights_total",
    "A_finish_rate", "A_rounds_fought", "A_control_time_rate",
    "A_reversals_rate", "A_knockdowns_landed_rate", "A_knockdowns_absorbed_rate",
    "A_head_strike_pct", "A_body_strike_pct", "A_leg_strike_pct",
    "A_clinch_strike_pct", "A_ground_strike_pct",
]

FIGHTER_B_FEATURES = [
    "B_sig_strike_rate", "B_takedown_accuracy", "B_ko_rate", "B_sub_rate",
    "B_strike_defense", "B_takedown_defense", "B_cardio_index", "B_reach",
    "B_height", "B_age", "B_win_streak", "B_loss_streak", "B_fights_total",
    "B_finish_rate", "B_rounds_fought", "B_control_time_rate",
    "B_reversals_rate", "B_knockdowns_landed_rate", "B_knockdowns_absorbed_rate",
    "B_head_strike_pct", "B_body_strike_pct", "B_leg_strike_pct",
    "B_clinch_strike_pct", "B_ground_strike_pct",
]

CROSS_FEATURES = [
    "style_clash", "striking_differential", "grappling_differential",
    "aggression_mismatch", "pace_mismatch", "reach_advantage",
    "orthodox_vs_southpaw", "offensive_vs_defensive", "wrestling_vs_striker",
    "experience_gap", "durability_mismatch", "cardio_mismatch",
    "finish_rate_combined", "ko_probability", "sub_probability",
    "decision_probability", "competitive_balance", "action_density_prediction",
    "ground_game_clash", "clinch_threat", "upset_potential", "rankings_gap",
    "momentum_differential", "marketability_score",
]

ALL_FEATURE_NAMES = FIGHTER_A_FEATURES + FIGHTER_B_FEATURES + CROSS_FEATURES

CATEGORY_SLICES = {
    "Fighter A Stats": slice(0, 24),
    "Fighter B Stats": slice(24, 48),
    "Matchup Cross-Features": slice(48, 72),
}

DOMINANCE_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_DEFAULTS = dict(dpi=300, bbox_inches="tight", facecolor="white")


def _apply_style(ax: plt.Axes) -> None:
    """Presentation-ready axis formatting: 14pt+ labels, clean spines."""
    ax.tick_params(labelsize=13)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    if ax.get_title():
        ax.title.set_size(16)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Global SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    model,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    model_name: str = "model",
    output_dir: str = "outputs/",
) -> shap.Explanation:
    """
    Compute SHAP values with TreeExplainer and generate global plots.

    Saves
    -----
    {model_name}_shap_beeswarm.png    — full beeswarm summary
    {model_name}_shap_bar.png         — top-20 global bar importance
    {model_name}_crossfeature_importance.png — cross-features only (48-71)

    Returns
    -------
    shap.Explanation with .values, .base_values, .data, .feature_names
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    out = _ensure_dir(output_dir)
    logger.info("Computing SHAP values for %s on %d samples...", model_name, len(X_test))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # TreeExplainer on binary classifiers can return shape (n, features, 2).
    # We want the positive-class explanation.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values.feature_names = feature_names

    # ── Sanity check: dominant feature warning ───────────────────────────
    _check_dominance(shap_values, feature_names)

    # ── (a) Beeswarm ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.sca(ax)
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    ax.set_title(f"{model_name} — SHAP Beeswarm (Top 20 Features)", fontsize=16, pad=12)
    _apply_style(ax)
    path_bee = out / f"{model_name}_shap_beeswarm.png"
    fig.savefig(path_bee, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path_bee)

    # ── (b) Global bar — top 20 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.bar(shap_values, max_display=20, show=False)
    ax.set_title(f"{model_name} — Global Feature Importance (Top 20)", fontsize=16, pad=12)
    _apply_style(ax)
    path_bar = out / f"{model_name}_shap_bar.png"
    fig.savefig(path_bar, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path_bar)

    # ── (c) Cross-features only ──────────────────────────────────────────
    cross_slice = CATEGORY_SLICES["Matchup Cross-Features"]
    cross_explanation = shap_values[:, cross_slice]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.bar(cross_explanation, max_display=24, show=False)
    ax.set_title(
        f"{model_name} — Matchup Cross-Feature Importance (indices 48-71)",
        fontsize=15, pad=12,
    )
    _apply_style(ax)
    path_cross = out / f"{model_name}_crossfeature_importance.png"
    fig.savefig(path_cross, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path_cross)

    return shap_values


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-matchup explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_matchup(
    model,
    fighter_a_name: str,
    fighter_b_name: str,
    X_single: np.ndarray,
    feature_names: list[str] | None = None,
    model_name: str = "model",
    output_dir: str = "outputs/",
) -> dict:
    """
    Generate a SHAP waterfall plot for one specific matchup and print a
    plain-English explanation of the prediction.

    Parameters
    ----------
    X_single : 1-D array of shape (72,) or 2-D of shape (1, 72)

    Saves
    -----
    {model_name}_{fighter_a}_vs_{fighter_b}_explanation.png

    Returns
    -------
    dict with keys: prediction, top_positive, top_negative, shap_explanation
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    out = _ensure_dir(output_dir)
    X_single = np.atleast_2d(X_single)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_single)

    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values.feature_names = feature_names
    single_expl = shap_values[0]

    # ── Top contributors ─────────────────────────────────────────────────
    sv = single_expl.values
    order = np.argsort(sv)
    top_pos_idx = order[-3:][::-1]
    top_neg_idx = order[:3]

    top_positive = [(feature_names[i], float(sv[i])) for i in top_pos_idx if sv[i] > 0]
    top_negative = [(feature_names[i], float(sv[i])) for i in top_neg_idx if sv[i] < 0]

    prob = float(model.predict_proba(X_single)[0, 1])

    # ── Plain-English explanation ────────────────────────────────────────
    pos_str = ", ".join(f"{name} (+{val:.3f})" for name, val in top_positive) or "none"
    neg_str = ", ".join(f"{name} ({val:.3f})" for name, val in top_negative) or "none"

    explanation_text = (
        f"\n{'=' * 70}\n"
        f"  {fighter_a_name} vs {fighter_b_name}\n"
        f"  P(bonus fight) = {prob:.1%}\n"
        f"{'=' * 70}\n"
        f"  Predicted as entertaining because:  {pos_str}\n"
        f"  Working against it:                 {neg_str}\n"
        f"{'=' * 70}\n"
    )
    print(explanation_text)
    logger.info(explanation_text)

    # ── Waterfall plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.waterfall(single_expl, max_display=15, show=False)
    ax.set_title(
        f"{fighter_a_name} vs {fighter_b_name} — Prediction Breakdown ({model_name})",
        fontsize=15, pad=12,
    )
    _apply_style(ax)

    safe_a = _sanitize_filename(fighter_a_name)
    safe_b = _sanitize_filename(fighter_b_name)
    path = out / f"{model_name}_{safe_a}_vs_{safe_b}_explanation.png"
    fig.savefig(path, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path)

    return {
        "prediction": prob,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "shap_explanation": single_expl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-model feature importance comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_feature_importance(
    models_dict: dict[str, object],
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 15,
    output_dir: str = "outputs/",
) -> dict[str, np.ndarray]:
    """
    Side-by-side top-N feature importance from multiple models.

    Saves
    -----
    feature_importance_comparison.png

    Returns
    -------
    dict mapping model_name → mean absolute SHAP values array (72,)
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    out = _ensure_dir(output_dir)
    n_models = len(models_dict)

    mean_abs: dict[str, np.ndarray] = {}
    for name, model in models_dict.items():
        logger.info("Computing SHAP for %s...", name)
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_test)
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
        mean_abs[name] = np.abs(sv.values).mean(axis=0)

    # Union of the top-N features across all models
    top_features: set[int] = set()
    for vals in mean_abs.values():
        top_features.update(np.argsort(vals)[-top_n:].tolist())
    top_indices = sorted(top_features, key=lambda i: -max(v[i] for v in mean_abs.values()))
    top_indices = top_indices[:top_n]

    labels = [feature_names[i] for i in top_indices]
    x = np.arange(len(top_indices))
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 8))
    for offset, (name, vals) in enumerate(mean_abs.items()):
        heights = [vals[i] for i in top_indices]
        ax.barh(x + offset * bar_width, heights, bar_width, label=name, alpha=0.85)

    ax.set_yticks(x + bar_width * (n_models - 1) / 2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance Comparison — Top {top_n}", fontsize=16, pad=12)
    ax.legend(fontsize=13)
    _apply_style(ax)

    path = out / "feature_importance_comparison.png"
    fig.savefig(path, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path)

    return mean_abs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category-level importance
# ─────────────────────────────────────────────────────────────────────────────

def feature_category_importance(
    model,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    model_name: str = "model",
    output_dir: str = "outputs/",
) -> dict[str, float]:
    """
    Aggregate SHAP importance by feature category and visualise.

    Categories
    ----------
    Fighter A Stats (0-23), Fighter B Stats (24-47), Matchup Cross-Features (48-71)

    If cross-features dominate, it validates the decision to engineer them.

    Saves
    -----
    {model_name}_feature_category_importance.png

    Returns
    -------
    dict mapping category name → total mean |SHAP|
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    out = _ensure_dir(output_dir)

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_test)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv.values).mean(axis=0)

    category_importance: dict[str, float] = {}
    for cat_name, cat_slice in CATEGORY_SLICES.items():
        category_importance[cat_name] = float(mean_abs[cat_slice].sum())

    total = sum(category_importance.values())
    cats = list(category_importance.keys())
    vals = [category_importance[c] for c in cats]
    pcts = [v / total * 100 for v in vals]

    # ── Bar chart ────────────────────────────────────────────────────────
    colors = ["#4C78A8", "#F58518", "#E45756"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(cats, vals, color=colors, edgecolor="white", linewidth=1.5)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + max(vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", fontsize=14, fontweight="bold",
        )

    ax.set_xlabel("Total Mean |SHAP value|", fontsize=15)
    ax.set_title(
        f"{model_name} — Feature Category Importance", fontsize=16, pad=12
    )
    ax.invert_yaxis()
    _apply_style(ax)

    path = out / f"{model_name}_feature_category_importance.png"
    fig.savefig(path, **_PLOT_DEFAULTS)
    plt.close(fig)
    logger.info("Saved %s", path)

    # ── Log the result ───────────────────────────────────────────────────
    for cat, val, pct in zip(cats, vals, pcts):
        logger.info("  %-28s  %.4f  (%.1f%%)", cat, val, pct)

    if pcts[2] > pcts[0] and pcts[2] > pcts[1]:
        logger.info(
            "Matchup cross-features are the dominant category (%.1f%%) — "
            "validates the feature engineering investment.",
            pcts[2],
        )

    return category_importance


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sanity check — dominant-feature flag
# ─────────────────────────────────────────────────────────────────────────────

def _check_dominance(
    shap_values: shap.Explanation,
    feature_names: list[str],
) -> list[str]:
    """
    Flag features that account for >30% of total SHAP importance.

    Returns list of warning strings (empty if no flags).
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    total = mean_abs.sum()
    if total == 0:
        return []

    warnings: list[str] = []
    for i, val in enumerate(mean_abs):
        share = val / total
        if share > DOMINANCE_THRESHOLD:
            msg = (
                f"SANITY CHECK: feature '{feature_names[i]}' accounts for "
                f"{share:.1%} of total SHAP importance (threshold={DOMINANCE_THRESHOLD:.0%}). "
                f"This may indicate data leakage or a degenerate feature."
            )
            logger.warning(msg)
            warnings.append(msg)

    if not warnings:
        logger.info("Sanity check passed — no single feature exceeds %.0f%% importance.", DOMINANCE_THRESHOLD * 100)

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Strip non-alphanumeric characters for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — test with placeholder data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib.util

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    np.random.seed(42)

    # Load baselines module directly to avoid the torch-dependent __init__.py
    spec = importlib.util.spec_from_file_location(
        "baselines", Path(__file__).parent / "baselines.py"
    )
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)

    print("Training baseline models on placeholder data...")
    bc = baselines.BaselineComparison()
    bc.load_data()
    bc.train_all()

    X_test = bc._data["X_test_scaled"]
    y_test = bc._data["y_test"]
    feature_names = ALL_FEATURE_NAMES

    # Pick tree-based models for SHAP TreeExplainer
    tree_models = {
        name: model
        for name, model in bc.models.items()
        if name in ("XGBoost", "RandomForest")
    }

    # ── 1. Global SHAP analysis per model ────────────────────────────────
    for name, model in tree_models.items():
        print(f"\n--- SHAP analysis: {name} ---")
        run_shap_analysis(model, X_test, feature_names, model_name=name)

    # ── 2. Single-matchup explanation ────────────────────────────────────
    best_name = max(tree_models, key=lambda m: bc.results[m]["AUC-ROC"])
    best_model = tree_models[best_name]

    print(f"\n--- Matchup explanation ({best_name}) ---")
    sample_idx = np.where(y_test == 1)[0]
    sample_idx = sample_idx[0] if len(sample_idx) > 0 else 0
    explain_matchup(
        best_model,
        fighter_a_name="Charles Oliveira",
        fighter_b_name="Justin Gaethje",
        X_single=X_test[sample_idx],
        feature_names=feature_names,
        model_name=best_name,
    )

    # ── 3. Multi-model comparison ────────────────────────────────────────
    print("\n--- Feature importance comparison ---")
    compare_feature_importance(tree_models, X_test, feature_names)

    # ── 4. Category-level importance ─────────────────────────────────────
    print(f"\n--- Category importance ({best_name}) ---")
    cats = feature_category_importance(best_model, X_test, feature_names, model_name=best_name)

    print("\n" + "=" * 60)
    print("  FEATURE CATEGORY BREAKDOWN")
    print("=" * 60)
    total = sum(cats.values())
    for cat, val in cats.items():
        print(f"  {cat:30s}  {val:.4f}  ({val / total * 100:.1f}%)")
    print("=" * 60)
    print("\nAll plots saved to outputs/")
