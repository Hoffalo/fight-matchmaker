"""
models/eval_calibration_spearman.py
Mattheus's M4 + M5 evaluation: calibration analysis + Spearman rank correlation.

M4 — Calibration: for each model, plot the calibration curve on the held-out
     test set (Jan 2026+). A well-calibrated model follows the diagonal —
     deviations are reported honestly. Output: PNG to outputs/.

M5 — Spearman: for each model, compute Spearman's rho between predicted
     probability and the actual binary label on the test set, plus a
     per-event analysis (does the model rank bonus fights above non-bonus
     fights *within each card*?). The per-event story is the headline
     metric for a "matchmaker" — ranking matters more than calibration.

Run:
    python -m models.eval_calibration_spearman                # all available models
    python -m models.eval_calibration_spearman --models logreg rf

Outputs (written to ``outputs/``):
    calibration_curves.png             — all models on one figure
    calibration_<model>.png            — per-model close-up
    spearman_analysis.md               — markdown table for slide 11
    eval_summary.json                  — machine-readable summary

Both deliverables iterate over models in MODELS_TO_RUN; missing dependencies
(xgboost / libomp) are skipped gracefully with a warning so the script
always produces a partial output rather than failing.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Model factories — each returns a fitted estimator with .predict_proba
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    key: str
    label: str
    factory: Callable           # zero-arg factory returning unfitted estimator
    available: bool = True
    skip_reason: str = ""


def _logreg_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
    )


def _rf_factory():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def _xgb_factory():
    from xgboost import XGBClassifier  # may raise if libomp missing
    try:
        from config import SCALE_POS_WEIGHT
    except Exception:
        SCALE_POS_WEIGHT = 2.7
    return XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=1.0, min_child_weight=5,
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
    )


def _check_xgb() -> tuple[bool, str]:
    try:
        import xgboost  # noqa
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0]}"


def get_model_specs() -> list[ModelSpec]:
    xgb_ok, xgb_err = _check_xgb()
    return [
        ModelSpec("logreg", "Logistic Regression", _logreg_factory),
        ModelSpec("rf",     "Random Forest",       _rf_factory),
        ModelSpec("xgb",    "XGBoost",             _xgb_factory,
                  available=xgb_ok, skip_reason=xgb_err),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Calibration analysis
# ─────────────────────────────────────────────────────────────────────────────

def calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """
    Brier score + Expected Calibration Error (ECE).

    ECE is the mean absolute deviation between bin-mean predicted probability
    and bin-mean observed frequency, weighted by bin size.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    brier = float(brier_score_loss(y_true, y_proba))
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile",
    )

    bin_edges = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(y_proba, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            ece += abs(y_proba[mask].mean() - y_true[mask].mean()) * mask.mean()

    return {
        "brier": brier,
        "ece":   float(ece),
        "prob_true_per_bin": prob_true.tolist(),
        "prob_pred_per_bin": prob_pred.tolist(),
    }


def plot_calibration_all(
    results: list[dict],
    save_path: Path,
) -> None:
    """One plot with every model overlaid. The diagonal = perfect calibration."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")

    for r in results:
        ax.plot(
            r["calibration"]["prob_pred_per_bin"],
            r["calibration"]["prob_true_per_bin"],
            marker="o", linewidth=2,
            label=f"{r['label']}  (Brier={r['calibration']['brier']:.3f}, ECE={r['calibration']['ece']:.3f})",
        )

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual bonus fights")
    ax.set_title("Calibration on test set (Jan 2026+)\nLower Brier / ECE = better calibrated")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_calibration_one(result: dict, save_path: Path) -> None:
    """Per-model calibration plot (close-up)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
    ax.plot(
        result["calibration"]["prob_pred_per_bin"],
        result["calibration"]["prob_true_per_bin"],
        marker="o", linewidth=2, color="C0", label=result["label"],
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual bonus fights")
    ax.set_title(
        f"{result['label']} — Calibration\n"
        f"Brier={result['calibration']['brier']:.3f}  "
        f"ECE={result['calibration']['ece']:.3f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Spearman rank correlation
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_per_fight(
    fight_ids: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse (A,B) and (B,A) augmented rows of the same fight into one row.
    The two rows always share the same label; their probabilities are averaged
    so the matchmaker's per-fight prediction is corner-order-invariant.
    """
    unique_fids, idx = np.unique(fight_ids, return_inverse=True)
    n = len(unique_fids)
    yt = np.zeros(n, dtype=y_true.dtype)
    yp = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for i, j in enumerate(idx):
        yt[j] = y_true[i]
        yp[j] += y_proba[i]
        counts[j] += 1
    yp /= counts
    return unique_fids, yt, yp


def _real_event_ids_for_fights(
    fight_ids: np.ndarray, db_path: str = "data/ufc_matchmaker.db"
) -> np.ndarray:
    """Look up real DB event_ids for a set of fight_ids."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in fight_ids)
    rows = conn.execute(
        f"SELECT id, event_id FROM fights WHERE id IN ({placeholders})",
        [int(f) for f in fight_ids],
    ).fetchall()
    conn.close()
    fid_to_eid = {fid: eid for fid, eid in rows}
    return np.array(
        [fid_to_eid.get(int(f), -1) for f in fight_ids], dtype=np.int64
    )


def spearman_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fight_ids: np.ndarray,
    db_path: str = "data/ufc_matchmaker.db",
) -> dict:
    """
    Overall Spearman's rho on the test set + per-event ranking analysis.

    The (A,B) / (B,A) augmented rows are deduplicated to one row per fight
    (probabilities averaged across the two orderings). Per-event analysis
    groups fights by their real DB event_id (UFC card).

    Per-event: for each card with at least one bonus fight, compute the
    mean predicted-probability rank (1=highest) of the actual bonus fights.
    Random ranking baseline = (n+1)/2.
    """
    from scipy.stats import spearmanr

    # Deduplicate to one row per fight
    fids, yt_fight, yp_fight = _dedup_per_fight(fight_ids, y_true, y_proba)

    rho, p = spearmanr(yp_fight, yt_fight)
    overall = {
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "p_value":      float(p)   if not np.isnan(p)   else None,
        "n_fights":     int(len(yt_fight)),
    }

    # Real event_ids from DB
    real_event_ids = _real_event_ids_for_fights(fids, db_path)

    per_event = {"events": []}
    all_mean_ranks: list[float] = []
    all_random_baseline: list[float] = []
    all_top1_hit: list[int] = []
    all_event_rhos: list[float] = []

    for eid in np.unique(real_event_ids):
        if eid < 0:
            continue
        mask = real_event_ids == eid
        yt = yt_fight[mask]
        yp = yp_fight[mask]
        n = len(yt)
        n_pos = int(yt.sum())
        if n < 2 or n_pos == 0 or n_pos == n:
            # Need at least 2 fights AND at least one of each class
            continue
        # Rank highest probability = 1 (ties broken arbitrarily; we use argsort).
        ranks = (-yp).argsort().argsort() + 1
        bonus_ranks = ranks[yt == 1]
        mean_rank = float(np.mean(bonus_ranks))
        random_baseline = (n + 1) / 2
        top1_hit = int(1 in bonus_ranks)

        # Per-event Spearman ρ between probability ranking and binary label.
        # Constant-arm guard already passed above (n_pos in [1, n-1]).
        ev_rho, _ = spearmanr(yp, yt)
        ev_rho = float(ev_rho) if not np.isnan(ev_rho) else 0.0

        all_mean_ranks.append(mean_rank)
        all_random_baseline.append(random_baseline)
        all_top1_hit.append(top1_hit)
        all_event_rhos.append(ev_rho)
        per_event["events"].append({
            "event_id":         int(eid),
            "n_fights":         n,
            "n_bonus":          n_pos,
            "spearman_rho":     ev_rho,
            "bonus_mean_rank":  mean_rank,
            "random_baseline":  random_baseline,
            "top1_was_bonus":   bool(top1_hit),
        })

    if all_mean_ranks:
        per_event["summary"] = {
            "n_events":             len(all_mean_ranks),
            "mean_event_rho":       float(np.mean(all_event_rhos)),
            "mean_bonus_rank":      float(np.mean(all_mean_ranks)),
            "mean_random_baseline": float(np.mean(all_random_baseline)),
            "top1_hit_rate":        float(np.mean(all_top1_hit)),
        }

    return {"overall": overall, "per_event": per_event}


# ─────────────────────────────────────────────────────────────────────────────
# Per-event Spearman bar chart  +  best-model calibration close-up
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_event_spearman(
    result: dict,
    save_path: Path,
    db_path: str = "data/ufc_matchmaker.db",
) -> None:
    """One bar per held-out UFC card; sorted by ρ; reference line at 0."""
    import matplotlib.pyplot as plt
    import sqlite3

    events = result["spearman"]["per_event"].get("events", [])
    if not events:
        logger.warning("No per-event data for %s; skipping bar chart", result["label"])
        return

    # Look up event names for labels
    conn = sqlite3.connect(db_path)
    eids = [e["event_id"] for e in events]
    placeholders = ",".join("?" for _ in eids)
    name_lookup = {
        eid: name for eid, name in conn.execute(
            f"SELECT id, name FROM events WHERE id IN ({placeholders})", eids,
        ).fetchall()
    }
    conn.close()

    events_sorted = sorted(events, key=lambda e: e["spearman_rho"])
    rhos   = [e["spearman_rho"] for e in events_sorted]
    labels = [
        (name_lookup.get(e["event_id"], f"Event #{e['event_id']}"))[:35] +
        f"\n(n={e['n_fights']}, b={e['n_bonus']})"
        for e in events_sorted
    ]
    colors = ["#2ca02c" if r > 0 else "#d62728" if r < 0 else "#7f7f7f" for r in rhos]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(rhos))))
    ax.barh(range(len(rhos)), rhos, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)

    summary = result["spearman"]["per_event"].get("summary", {})
    mean_rho = summary.get("mean_event_rho", 0.0)
    ax.axvline(mean_rho, color="C0", linewidth=1.5, linestyle="--",
               label=f"mean ρ = {mean_rho:+.3f}")

    ax.set_xlabel("Spearman ρ (within event)")
    ax.set_title(
        f"{result['label']} — per-event rank correlation\n"
        f"global ρ = {result['spearman']['overall']['spearman_rho']:+.3f}  "
        f"|  per-event mean ρ = {mean_rho:+.3f}  "
        f"|  top-1 hit = {summary.get('top1_hit_rate', 0):.0%}"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def pick_best_model(results: list[dict]) -> dict | None:
    """
    Pick the model that maximises per-event mean Spearman ρ — the
    matchmaker's actual objective. Falls back to global ρ if no
    per-event summary is available.
    """
    if not results:
        return None
    def _score(r):
        s = r["spearman"]["per_event"].get("summary", {})
        return s.get("mean_event_rho", r["spearman"]["overall"]["spearman_rho"] or -1)
    return max(results, key=_score)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    model_keys: list[str] | None = None,
    db_path: str = "data/ufc_matchmaker.db",
    n_bins: int = 10,
) -> dict:
    """
    End-to-end M4 + M5: load canonical splits → train each model on train →
    evaluate on test set → save plots + markdown.
    """
    from models.data_loader import get_canonical_splits

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical splits...")
    data = get_canonical_splits(db_path=db_path, subset_features=False)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"],  data["y_test"]

    # Recover the real fight_ids for the test set so we can dedupe (A,B)/(B,A)
    # and group by real DB event_id. data_loader's `event_ids_test` is just
    # a per-fight pseudo-id (each fight = its own event), which is useless
    # for matchmaker-style ranking analysis.
    from data.db import Database
    from models.data_splits import build_raw_pairs, temporal_split_raw, augment_pair, build_full_matchup_vector
    raw = build_raw_pairs(Database(db_path))
    _, _, raw_test = temporal_split_raw(raw)
    _, _, meta_test = augment_pair(raw_test, vector_fn=build_full_matchup_vector)
    test_fight_ids = meta_test["fight_id"]

    specs = get_model_specs()
    if model_keys:
        specs = [s for s in specs if s.key in model_keys]

    results: list[dict] = []
    skipped: list[dict] = []

    for spec in specs:
        if not spec.available:
            logger.warning("Skipping %s — %s", spec.label, spec.skip_reason)
            skipped.append({"key": spec.key, "label": spec.label, "reason": spec.skip_reason})
            continue

        logger.info("Training %s on %d train samples...", spec.label, len(y_tr))
        clf = spec.factory()
        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_te)[:, 1]

        cal = calibration_metrics(y_te, y_proba, n_bins=n_bins)
        spr = spearman_metrics(y_te, y_proba, fight_ids=test_fight_ids, db_path=db_path)

        result = {
            "key":        spec.key,
            "label":      spec.label,
            "calibration": cal,
            "spearman":    spr,
        }
        results.append(result)

        plot_calibration_one(result, OUTPUTS_DIR / f"calibration_{spec.key}.png")

    if results:
        plot_calibration_all(results, OUTPUTS_DIR / "calibration_curves.png")

    # Markdown summary
    md = _format_markdown(results, skipped, data["summary"])
    md_path = OUTPUTS_DIR / "spearman_analysis.md"
    md_path.write_text(md)
    logger.info("Saved %s", md_path)

    summary = {
        "models":  results,
        "skipped": skipped,
        "data":    data["summary"],
    }
    json_path = OUTPUTS_DIR / "eval_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved %s", json_path)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: evaluate from precomputed test predictions (Raji's npz)
# ─────────────────────────────────────────────────────────────────────────────

NPZ_MODEL_LABELS = {
    "logreg_y_proba":  "Logistic Regression",
    "rf_y_proba":      "Random Forest (tuned)",
    "xgb_y_proba":     "XGBoost (tuned)",
    "nn_y_proba":      "Neural Net (12→16→1)",
}


def run_eval_from_npz(
    npz_path: str,
    db_path: str = "data/ufc_matchmaker.db",
    n_bins: int = 10,
) -> dict:
    """
    Load test-set predictions from a precomputed .npz (Raji's checkpoint
    output) and run M4 + M5 against the actual tuned models.

    The .npz must contain ``y_true`` and one or more of the keys in
    NPZ_MODEL_LABELS (``logreg_y_proba``, ``rf_y_proba``, ``xgb_y_proba``,
    ``nn_y_proba``). Row ordering must match get_canonical_splits()'s test
    split — verified at runtime against the local pipeline.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(npz_path)
    y_true = npz["y_true"].astype(np.float32)

    # Verify ordering matches our local pipeline so fight_id lookup works
    from data.db import Database
    from models.data_splits import (
        build_raw_pairs, temporal_split_raw, augment_pair, build_full_matchup_vector,
    )
    raw = build_raw_pairs(Database(db_path))
    _, _, raw_test = temporal_split_raw(raw)
    _, y_test_local, meta_test = augment_pair(raw_test, vector_fn=build_full_matchup_vector)

    if not np.array_equal(y_true, y_test_local.astype(np.float32)):
        raise ValueError(
            "y_true in npz does not match the local test split. "
            "Row ordering is required to match get_canonical_splits()."
        )
    test_fight_ids = meta_test["fight_id"]

    # Build a results entry per available model
    results: list[dict] = []
    for key, label in NPZ_MODEL_LABELS.items():
        if key not in npz.files:
            logger.info("Skipping %s — %s not in npz", label, key)
            continue
        y_proba = npz[key].astype(np.float64)
        cal = calibration_metrics(y_true, y_proba, n_bins=n_bins)
        spr = spearman_metrics(y_true, y_proba, fight_ids=test_fight_ids, db_path=db_path)
        result = {
            "key":         key.replace("_y_proba", ""),
            "label":       label,
            "calibration": cal,
            "spearman":    spr,
        }
        results.append(result)
        plot_calibration_one(result, OUTPUTS_DIR / f"calibration_{result['key']}.png")
        plot_per_event_spearman(result, OUTPUTS_DIR / f"spearman_per_event_{result['key']}.png", db_path=db_path)

    if results:
        plot_calibration_all(results, OUTPUTS_DIR / "calibration_curves.png")

    # Best-model dedicated outputs (slide 11 placeholders)
    best = pick_best_model(results)
    if best is not None:
        plot_calibration_one(best, OUTPUTS_DIR / "calibration_best.png")
        plot_per_event_spearman(best, OUTPUTS_DIR / "spearman_per_event_best.png", db_path=db_path)
        logger.info("Best model = %s (per-event mean ρ = %.3f)",
                    best["label"],
                    best["spearman"]["per_event"].get("summary", {}).get("mean_event_rho", 0.0))

    # Test-set summary for the markdown header
    n_pos = int(y_true.sum()); n_total = len(y_true)
    data_summary = {
        "test": {
            "fights":  n_total // 2,  # accounting for (A,B)/(B,A) augmentation
            "rows":    n_total,
            "pos":     n_pos // 2,
            "pos_pct": 100.0 * n_pos / max(n_total, 1),
        }
    }

    md_path = OUTPUTS_DIR / "spearman_analysis.md"
    md_path.write_text(_format_markdown(results, [], data_summary, best=best))
    logger.info("Saved %s", md_path)

    json_path = OUTPUTS_DIR / "eval_summary.json"
    json_path.write_text(json.dumps(
        {"models": results, "best": best["key"] if best else None, "data": data_summary},
        indent=2, default=str,
    ))
    logger.info("Saved %s", json_path)
    return {"models": results, "best": best, "data": data_summary}


def _format_markdown(results: list[dict], skipped: list[dict], data_summary: dict, best: dict | None = None) -> str:
    test = data_summary.get("test", {})
    lines = [
        "# M4 + M5 — Calibration & Spearman Analysis",
        "",
        f"**Test set:** Jan 2026+ ({test.get('fights', '?')} fights, "
        f"{test.get('pos', '?')} positives, {test.get('pos_pct', 0):.1f}% bonus rate)",
        "",
        "*All numbers below are on the held-out test set, which was never seen "
        "during training, validation, or hyperparameter tuning.*",
        "",
    ]
    if best is not None:
        bsum = best["spearman"]["per_event"].get("summary", {})
        lines += [
            f"**Best model (by per-event ρ): {best['label']}**",
            "",
            f"- Global Spearman ρ: **{best['spearman']['overall']['spearman_rho']:+.3f}**",
            f"- Per-event mean ρ: **{bsum.get('mean_event_rho', 0):+.3f}** "
            f"(across {bsum.get('n_events', 0)} held-out cards)",
            f"- Top-1 hit rate: **{bsum.get('top1_hit_rate', 0):.0%}**",
            f"- Brier: {best['calibration']['brier']:.3f}  ·  ECE: {best['calibration']['ece']:.3f}",
            "",
        ]
    lines += [
        "## Model Comparison",
        "",
        "| Model | Brier ↓ | ECE ↓ | Global ρ ↑ | Per-event mean ρ ↑ | Mean bonus rank ↓ | Random baseline | Top-1 hit ↑ |",
        "|-------|---------|-------|------------|--------------------|--------------------|-----------------|-------------|",
    ]
    for r in results:
        cal = r["calibration"]
        spr = r["spearman"]
        rho = spr["overall"]["spearman_rho"]
        rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
        pev = spr["per_event"].get("summary", {})
        ev_rho = pev.get("mean_event_rho")
        mr  = pev.get("mean_bonus_rank")
        rb  = pev.get("mean_random_baseline")
        t1  = pev.get("top1_hit_rate")
        marker = " ★" if best and r is best else ""
        if ev_rho is not None:
            lines.append(
                f"| {r['label']}{marker} | {cal['brier']:.3f} | {cal['ece']:.3f} | "
                f"{rho_s} | {ev_rho:+.3f} | {mr:.2f} | {rb:.2f} | {t1:.0%} |"
            )
        else:
            lines.append(
                f"| {r['label']}{marker} | {cal['brier']:.3f} | {cal['ece']:.3f} | "
                f"{rho_s} | n/a | n/a | n/a | n/a |"
            )

    if skipped:
        lines += ["", "## Skipped models", ""]
        for s in skipped:
            lines.append(f"- **{s['label']}** — {s['reason']}")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Brier / ECE**: closer to 0 means probabilities are trustworthy as absolutes.",
        "  If both are high, predicted probabilities are off-scale but ranking can still be valid.",
        "- **Spearman ρ**: rank correlation between predicted probability and actual label",
        "  on the test set. The headline metric for a *matchmaker*: even if probabilities",
        "  aren't well-calibrated, a high ρ means the model orders fights correctly.",
        "- **Per-event mean bonus rank**: within each UFC card, what's the average rank",
        "  (1 = highest predicted probability) of the actual bonus fights? Lower than the",
        "  random baseline `(n+1)/2` means the model is genuinely ranking bonus fights",
        "  near the top of each card.",
        "- **Top-1 hit rate**: fraction of cards where the model's #1-ranked fight was an",
        "  actual bonus fight.",
        "",
        "Generated by `models/eval_calibration_spearman.py`.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="M4 (calibration) + M5 (Spearman)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model keys to run when training fresh: logreg, rf, xgb")
    parser.add_argument("--db", default="data/ufc_matchmaker.db")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--npz", default=None,
                        help="Path to a precomputed test-predictions npz "
                             "(skips training; loads y_proba per model from disk).")
    args = parser.parse_args()
    if args.npz:
        run_eval_from_npz(args.npz, db_path=args.db, n_bins=args.bins)
    else:
        run_eval(model_keys=args.models, db_path=args.db, n_bins=args.bins)


if __name__ == "__main__":
    main()
