"""
One-shot generation of presentation artifacts (test-set evaluation, plots, JSON).

Production model: tuned XGBoost (max_depth=2) on 12 RFECV features
(``models/checkpoints/xgb_tuned_12feat.pkl`` — Pipeline: StandardScaler + XGBClassifier).

Run from repo root (single-thread env avoids macOS XGBoost/SHAP segfaults):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m models.generate_final_outputs
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import BASE_DIR, MODELS_DIR
from models.data_loader import get_canonical_splits
from models.nn_binary import load_binary_nn, predict_proba as nn_predict_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINTS = MODELS_DIR / "checkpoints"
DB_PATH = BASE_DIR / "data" / "ufc_matchmaker.db"
RANDOM_STATE = 42

COLORS = {
    "Logistic Regression": "#3B82F6",
    "Random Forest": "#6B7280",
    "XGBoost (tuned)": "#22C55E",
    "Neural Net (12→16→1)": "#EF4444",
}


def _raw_from_splits(splits: dict) -> tuple[np.ndarray, np.ndarray]:
    sc = splits["scaler"]
    X_train_raw = sc.inverse_transform(splits["X_train"]).astype(np.float64)
    X_test_raw = sc.inverse_transform(splits["X_test"]).astype(np.float64)
    return X_train_raw, X_test_raw


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def part1_model_comparison(
    splits: dict,
    y_test: np.ndarray,
    X_test_raw: np.ndarray,
    X_test_scaled: np.ndarray,
) -> dict:
    X_train_scaled = np.asarray(splits["X_train"], dtype=np.float64)
    y_train = splits["y_train"].astype(np.int64)

    logreg = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
        solver="lbfgs",
    )
    logreg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    xgb_path = CHECKPOINTS / "xgb_tuned_12feat.pkl"
    if not xgb_path.is_file():
        raise FileNotFoundError(f"Missing {xgb_path}")
    xgb_pipeline = joblib.load(xgb_path)

    nn_path = CHECKPOINTS / "nn_12feat.pt"
    if not nn_path.is_file():
        raise FileNotFoundError(f"Missing {nn_path}")
    nn_model = load_binary_nn(str(nn_path))

    models_dict: dict = {}

    def eval_model(name: str, y_proba: np.ndarray) -> None:
        y_proba = np.asarray(y_proba, dtype=np.float64).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        rho, pval = spearmanr(y_test, y_proba)
        rho_f = float(rho) if rho == rho else float("nan")
        print(
            f"  {name:<30} AUC={auc:.4f}  F1={f1:.4f}  ρ={rho_f:.4f}",
        )
        models_dict[name] = {
            "test_auc": round(float(auc), 4),
            "test_f1": round(float(f1), 4),
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "spearman_rho": round(rho_f, 4) if rho_f == rho_f else None,
            "spearman_p": round(float(pval), 4) if pval == pval else None,
            "y_proba": y_proba,
        }

    eval_model("Logistic Regression", logreg.predict_proba(X_test_scaled)[:, 1])
    eval_model("Random Forest", rf.predict_proba(X_test_scaled)[:, 1])
    eval_model("XGBoost (tuned)", xgb_pipeline.predict_proba(X_test_raw)[:, 1])
    eval_model("Neural Net (12→16→1)", nn_predict_proba(nn_model, X_test_scaled))

    y_mean = float(np.mean(y_test))
    models_dict["Baseline (majority)"] = {
        "test_auc": 0.5,
        "test_f1": 0.0,
        "test_accuracy": round(1.0 - y_mean, 4),
        "test_precision": 0.0,
        "test_recall": 0.0,
        "spearman_rho": 0.0,
        "spearman_p": 1.0,
    }

    out_path = OUTPUT_DIR / "final_model_comparison.json"
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "y_proba"}
        for k, v in models_dict.items()
    }
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return models_dict


def part2_roc_overlay(
    models_dict: dict,
    y_test: np.ndarray,
    n_test_fights: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in models_dict.items():
        if name == "Baseline (majority)" or "y_proba" not in data:
            continue
        y_proba = data["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_lbl = data["test_auc"]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc_lbl:.3f})",
            color=COLORS.get(name, "#999999"),
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(
        f"ROC Curves — Held-Out Test Set ({n_test_fights} fights, augmented rows)",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = PLOTS_DIR / "roc_overlay_final.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def part3_confusion_matrices(models_dict: dict, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panel = [
        (k, v)
        for k, v in models_dict.items()
        if k != "Baseline (majority)" and "y_proba" in v
    ]
    for idx, (name, data) in enumerate(panel):
        y_pred = (np.asarray(data["y_proba"]) >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Bonus", "Bonus"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(name.split("(")[0].strip(), fontsize=11)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def part4_calibration(models_dict: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in models_dict.items():
        if name == "Baseline (majority)" or "y_proba" not in data:
            continue
        prob_true, prob_pred = calibration_curve(
            y_test, data["y_proba"], n_bins=8, strategy="uniform"
        )
        ax.plot(
            prob_pred,
            prob_true,
            "s-",
            label=name,
            linewidth=2,
            markersize=6,
            color=COLORS.get(name, "#999999"),
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=14)
    ax.set_ylabel("Fraction of Actual Positives", fontsize=14)
    ax.set_title("Calibration — Reliability Diagram", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = PLOTS_DIR / "calibration_final.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def part5_shap_xgb(
    splits: dict,
    models_dict: dict,
    X_test_raw: np.ndarray,
) -> None:
    from models.pipeline_config import SELECTED_FEATURES

    xgb_path = CHECKPOINTS / "xgb_tuned_12feat.pkl"
    xgb_pipeline = joblib.load(xgb_path)
    xgb_model = xgb_pipeline.named_steps["xgb"]
    pipe_scaler = xgb_pipeline.named_steps["scaler"]
    X_test_scaled = pipe_scaler.transform(X_test_raw)

    feature_names = list(SELECTED_FEATURES or [])
    if len(feature_names) != X_test_scaled.shape[1]:
        feature_names = list(splits.get("feature_names", []))

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_scaled)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap_values.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title(
        "XGBoost — SHAP Feature Importance (Test Set)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path_bee = PLOTS_DIR / "shap_beeswarm_xgb.png"
    fig.savefig(path_bee, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_bee}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.sca(ax)
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title(
        "XGBoost — Mean |SHAP| Value (Feature Importance)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path_bar = PLOTS_DIR / "shap_bar_xgb.png"
    fig.savefig(path_bar, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_bar}")

    xgb_proba = np.asarray(models_dict["XGBoost (tuned)"]["y_proba"])
    top_idx = int(np.argmax(xgb_proba))
    single_expl = shap_values[top_idx]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(single_expl, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — Highest Predicted Matchup (P={xgb_proba[top_idx]:.1%})",
        fontsize=13,
    )
    plt.tight_layout()
    path_wf = PLOTS_DIR / "shap_waterfall_top.png"
    fig.savefig(path_wf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_wf}")


def part6_per_event_spearman(
    models_dict: dict,
    meta_test: dict,
    y_test: np.ndarray,
    db_path: Path,
) -> None:
    xgb_proba = np.asarray(models_dict["XGBoost (tuned)"]["y_proba"])
    fight_ids = meta_test["fight_id"]
    unique_ids = np.unique(fight_ids)

    fight_event: dict[int, dict] = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        qmarks = ",".join("?" * len(unique_ids))
        rows = conn.execute(
            f"""
            SELECT f.id AS fight_id, f.is_bonus_fight, e.id AS event_id,
                   e.name AS event_name, e.date AS event_date
            FROM fights f
            JOIN events e ON e.id = f.event_id
            WHERE f.id IN ({qmarks})
            """,
            [int(x) for x in unique_ids.tolist()],
        ).fetchall()
        for r in rows:
            fight_event[int(r["fight_id"])] = dict(r)
    finally:
        conn.close()

    per_fight_proba: dict[int, float] = {}
    per_fight_label: dict[int, int] = {}
    for fid in unique_ids:
        idxs = np.where(fight_ids == fid)[0]
        per_fight_proba[int(fid)] = float(np.mean(xgb_proba[idxs]))
        per_fight_label[int(fid)] = int(y_test[idxs[0]])

    events_agg: dict[int, dict] = {}
    for fid, p in per_fight_proba.items():
        info = fight_event.get(fid)
        if not info:
            continue
        eid = int(info["event_id"])
        if eid not in events_agg:
            events_agg[eid] = {
                "event_name": info["event_name"],
                "event_date": info["event_date"],
                "y_true": [],
                "y_pred": [],
            }
        events_agg[eid]["y_true"].append(per_fight_label[fid])
        events_agg[eid]["y_pred"].append(p)

    print("\n" + "=" * 70)
    print("  PER-EVENT SPEARMAN RANK CORRELATION (XGBoost, deduped per fight)")
    print("=" * 70)
    per_event_rhos: list[float] = []
    top1_hits = 0
    top3_hits = 0
    total_events = 0
    event_rows: list[dict] = []

    for eid, data in sorted(
        events_agg.items(), key=lambda kv: (kv[1]["event_date"] or "", kv[0])
    ):
        y_true = np.array(data["y_true"], dtype=np.int64)
        y_pred = np.array(data["y_pred"], dtype=np.float64)
        n = len(y_true)
        if n < 3 or y_true.sum() == 0:
            continue
        rho, _ = spearmanr(y_true, y_pred)
        rho_f = float(rho) if rho == rho else float("nan")
        per_event_rhos.append(rho_f)

        top1_idx = int(np.argmax(y_pred))
        if y_true[top1_idx] == 1:
            top1_hits += 1

        top3_idx = np.argsort(y_pred)[-3:]
        if any(y_true[i] == 1 for i in top3_idx):
            top3_hits += 1

        total_events += 1
        bonus_str = f"{int(y_true.sum())}b"
        hit = "✓" if y_true[top1_idx] == 1 else "✗"
        print(
            f"  {str(data['event_name']):<50} n={n:>2}, {bonus_str}, ρ={rho_f:+.3f}, top1={hit}",
        )
        event_rows.append(
            {
                "event_id": eid,
                "event_name": data["event_name"],
                "event_date": data["event_date"],
                "n_fights": n,
                "n_bonuses": int(y_true.sum()),
                "spearman_rho": round(rho_f, 4) if rho_f == rho_f else None,
                "top1_hit": bool(y_true[top1_idx] == 1),
            },
        )

    mean_rho = float(np.nanmean(per_event_rhos)) if per_event_rhos else None
    if total_events > 0:
        print(f"\n  Mean per-event ρ: {mean_rho:+.3f}" if mean_rho is not None else "")
        print(f"  Top-1 hit rate: {top1_hits}/{total_events} = {top1_hits/total_events:.0%}")
        print(f"  Top-3 hit rate: {top3_hits}/{total_events} = {top3_hits/total_events:.0%}")
    print("=" * 70)

    out = {
        "mean_rho": round(mean_rho, 4) if mean_rho is not None and mean_rho == mean_rho else None,
        "top1_hit_rate": f"{top1_hits}/{total_events}" if total_events > 0 else None,
        "top3_hit_rate": f"{top3_hits}/{total_events}" if total_events > 0 else None,
        "n_events": total_events,
        "events": event_rows,
    }
    path = OUTPUT_DIR / "per_event_spearman.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def part7_matchmaker_demo(y_test: np.ndarray, models_dict: dict) -> None:
    try:
        from models.matchmaker_v2 import MatchmakerV2

        mm = MatchmakerV2(
            db_path=str(DB_PATH),
            backend="nn",
            checkpoint_path=str(CHECKPOINTS / "nn_12feat.pt"),
            scaler_path=str(CHECKPOINTS / "scaler_12feat.pkl"),
        )

        print("\n" + "=" * 70)
        print("  MATCHMAKER DEMO — LIGHTWEIGHT DIVISION (NN checkpoint)")
        print("=" * 70)
        results = mm.rank_weight_class("Lightweight", top_n=10)

        demo_output = []
        for r in results:
            demo_output.append(
                {
                    "fighter_a": r["fighter_a"],
                    "fighter_b": r["fighter_b"],
                    "probability": r["probability"],
                    "stars": r["stars"],
                    "reasons": r.get("reasons", r.get("top_factors", [])),
                }
            )

        p = OUTPUT_DIR / "matchmaker_demo_lightweight.json"
        p.write_text(json.dumps(demo_output, indent=2), encoding="utf-8")
        print(f"Saved: {p}")

        card = mm.build_card(total_fights=5)
        p2 = OUTPUT_DIR / "matchmaker_dream_card.json"
        p2.write_text(
            json.dumps(
                [
                    {
                        "fighter_a": r["fighter_a"],
                        "fighter_b": r["fighter_b"],
                        "probability": r["probability"],
                        "weight_class": r.get("weight_class", ""),
                    }
                    for r in card
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved: {p2}")

    except Exception as exc:
        logger.warning("Matchmaker failed: %s", exc)
        print(f"\n  Matchmaker failed: {exc}")
        print("  Falling back to raw test predictions...")
        xgb_proba = np.asarray(models_dict["XGBoost (tuned)"]["y_proba"])
        top_indices = np.argsort(xgb_proba)[-10:][::-1]
        print("\n  Top 10 predicted entertainment fights in test set (augmented rows):")
        for rank, idx in enumerate(top_indices, 1):
            actual = "BONUS ✓" if y_test[idx] == 1 else "no bonus"
            print(f"    #{rank}  Fight index {idx}: P={xgb_proba[idx]:.1%} — {actual}")


def part8_auc_progression() -> None:
    milestones = [
        ("48-dim\nBaseline", 0.484, "#EF4444"),
        ("72-dim\n+Cross", 0.540, "#22C55E"),
        ("81-dim\n+Odds/Ctx", 0.592, "#22C55E"),
        ("115-dim\n+Rolling", 0.606, "#22C55E"),
        ("RFECV 12", 0.595, "#A855F7"),
        ("XGB\nTuned", 0.583, "#F59E0B"),
        ("NN\n12→16→1", 0.599, "#F59E0B"),
        ("PCA-10\nNN", 0.581, "#14B8A6"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    labels, aucs, clrs = zip(*milestones)
    bars = ax.bar(
        range(len(milestones)),
        aucs,
        color=clrs,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.004,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xticks(range(len(milestones)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ROC-AUC (5-fold CV)", fontsize=13)
    ax.set_title(
        "Model Development Journey — 15 Experiments",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(0.45, 0.65)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = PLOTS_DIR / "auc_progression_final.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def part9_list_outputs() -> None:
    print("\n" + "=" * 70)
    print("  ALL GENERATED OUTPUTS (this run)")
    print("=" * 70)
    for folder in (OUTPUT_DIR, PLOTS_DIR):
        if folder.exists():
            for f in sorted(folder.iterdir()):
                if f.is_file():
                    print(f"  {f!s:<52} {f.stat().st_size:>8,} bytes")
    print("=" * 70)
    print("\nDone. All outputs ready for presentation.")


def main(db_path: str | None = None) -> None:
    _ensure_dirs()
    db = Path(db_path) if db_path else DB_PATH
    if not db.is_file():
        raise FileNotFoundError(db)

    splits = get_canonical_splits(str(db))
    X_train_raw, X_test_raw = _raw_from_splits(splits)
    X_test_scaled = np.asarray(splits["X_test"], dtype=np.float64)
    y_test = splits["y_test"].astype(np.int64)
    meta_test = splits["meta_test"]
    n_test_fights = int(len(np.unique(meta_test["fight_id"])))

    print("=" * 70)
    print("  PART 1 — Final model comparison (first evaluation on held-out test)")
    print("=" * 70)
    models_dict = part1_model_comparison(splits, y_test, X_test_raw, X_test_scaled)

    part2_roc_overlay(models_dict, y_test, n_test_fights)
    part3_confusion_matrices(models_dict, y_test)
    part4_calibration(models_dict, y_test)

    print("\n" + "=" * 70)
    print("  PART 5 — SHAP (XGBoost booster on pipeline-scaled raw features)")
    print("=" * 70)
    part5_shap_xgb(splits, models_dict, X_test_raw)

    print("\n" + "=" * 70)
    print("  PART 6 — Per-event Spearman (deduped fights)")
    print("=" * 70)
    part6_per_event_spearman(models_dict, meta_test, y_test, db)

    part7_matchmaker_demo(y_test, models_dict)

    print("\n" + "=" * 70)
    print("  PART 8 — Experiment summary chart")
    print("=" * 70)
    part8_auc_progression()

    part9_list_outputs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
