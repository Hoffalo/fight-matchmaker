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

        all_mean_ranks.append(mean_rank)
        all_random_baseline.append(random_baseline)
        all_top1_hit.append(top1_hit)
        per_event["events"].append({
            "event_id":         int(eid),
            "n_fights":         n,
            "n_bonus":          n_pos,
            "bonus_mean_rank":  mean_rank,
            "random_baseline":  random_baseline,
            "top1_was_bonus":   bool(top1_hit),
        })

    if all_mean_ranks:
        per_event["summary"] = {
            "n_events":             len(all_mean_ranks),
            "mean_bonus_rank":      float(np.mean(all_mean_ranks)),
            "mean_random_baseline": float(np.mean(all_random_baseline)),
            "top1_hit_rate":        float(np.mean(all_top1_hit)),
        }

    return {"overall": overall, "per_event": per_event}


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


def _format_markdown(results: list[dict], skipped: list[dict], data_summary: dict) -> str:
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
        "## Model Comparison",
        "",
        "| Model | Brier ↓ | ECE ↓ | Spearman ρ ↑ | Per-event mean rank ↓ | Random baseline | Top-1 hit ↑ |",
        "|-------|---------|-------|--------------|------------------------|-----------------|-------------|",
    ]
    for r in results:
        cal = r["calibration"]
        spr = r["spearman"]
        rho = spr["overall"]["spearman_rho"]
        rho_s = f"{rho:.3f}" if rho is not None else "n/a"
        pev = spr["per_event"].get("summary", {})
        mr  = pev.get("mean_bonus_rank")
        rb  = pev.get("mean_random_baseline")
        t1  = pev.get("top1_hit_rate")
        lines.append(
            f"| {r['label']} | {cal['brier']:.3f} | {cal['ece']:.3f} | {rho_s} | "
            f"{mr:.2f} | {rb:.2f} | {t1:.2%} |"
            if mr is not None else
            f"| {r['label']} | {cal['brier']:.3f} | {cal['ece']:.3f} | {rho_s} | n/a | n/a | n/a |"
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
                        help="Subset of model keys to run: logreg, rf, xgb")
    parser.add_argument("--db", default="data/ufc_matchmaker.db")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    run_eval(model_keys=args.models, db_path=args.db, n_bins=args.bins)


if __name__ == "__main__":
    main()
