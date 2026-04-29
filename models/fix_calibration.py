"""
Stable calibration visuals for small held-out test sets (many augmented rows, few unique fights).

Saves:
- outputs/plots/calibration_final.png — 5 quantile bins + prediction histograms
- outputs/plots/calibration_bootstrap.png — XGBoost mean curve + bootstrap band
- outputs/plots/calibration_simple_bars.png — 3-bin bar chart (slide-friendly)

Run:  python -m models.fix_calibration
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from config import BASE_DIR, MODELS_DIR
from models.baselines import RANDOM_STATE
from models.data_loader import get_canonical_splits

CHECKPOINTS = MODELS_DIR / "checkpoints"
PLOTS = BASE_DIR / "outputs" / "plots"
META_PATH = BASE_DIR / "outputs" / "calibration_plot_meta.json"


def _unique_fight_count(splits: dict) -> int | None:
    m = splits.get("meta_test")
    if m is None or "fight_id" not in m:
        return None
    return int(len(np.unique(m["fight_id"])))


def main(db_path: str = "data/ufc_matchmaker.db") -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits(db_path)
    X_test = np.asarray(splits["X_test"], dtype=np.float64)
    y_test = np.asarray(splits["y_test"], dtype=np.int64)
    X_train = np.asarray(splits["X_train"], dtype=np.float64)
    y_train = np.asarray(splits["y_train"], dtype=np.int64)
    scaler = splits["scaler"]
    n_aug = len(y_test)
    n_fights = _unique_fight_count(splits)

    xgb_path = CHECKPOINTS / "xgb_tuned_12feat.pkl"
    if not xgb_path.is_file():
        raise FileNotFoundError(f"Missing {xgb_path}")
    xgb_pipeline = joblib.load(xgb_path)

    X_test_raw = scaler.inverse_transform(X_test).astype(np.float64)
    X_train_raw = scaler.inverse_transform(X_train).astype(np.float64)

    xgb_proba = xgb_pipeline.predict_proba(X_test_raw)[:, 1].astype(np.float64)

    logreg = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
        solver="lbfgs",
    )
    logreg.fit(X_train, y_train)
    lr_proba = logreg.predict_proba(X_test)[:, 1].astype(np.float64)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1].astype(np.float64)

    models: dict[str, np.ndarray] = {
        "XGBoost (tuned)": xgb_proba,
        "Logistic Regression": lr_proba,
        "Random Forest": rf_proba,
    }
    colors = {
        "XGBoost (tuned)": "#22C55E",
        "Logistic Regression": "#3B82F6",
        "Random Forest": "#6B7280",
    }

    # ── FIX 1: fewer bins, quantile strategy ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for name, proba in models.items():
        try:
            prob_true, prob_pred = calibration_curve(
                y_test, proba, n_bins=5, strategy="quantile",
            )
            brier = brier_score_loss(y_test, proba)
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                label=f"{name} (Brier={brier:.3f})",
                color=colors[name],
                linewidth=2,
                markersize=8,
            )
        except ValueError as e:
            print(f"  Calibration failed for {name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=13)
    ax.set_ylabel("Fraction of positives", fontsize=13)
    fight_note = f" ({n_fights} unique fights)" if n_fights is not None else ""
    ax.set_title(
        f"Calibration — 5 quantile bins (n = {n_aug} augmented rows{fight_note})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    for name, proba in models.items():
        ax2.hist(
            proba,
            bins=20,
            alpha=0.4,
            label=name,
            color=colors[name],
            edgecolor="white",
        )
    ax2.axvline(
        x=float(y_test.mean()),
        color="red",
        linestyle="--",
        label=f'Base rate ({float(y_test.mean()):.1%})',
    )
    ax2.set_xlabel("Predicted probability", fontsize=13)
    ax2.set_ylabel("Count", fontsize=13)
    ax2.set_title("Prediction distribution — test set", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out1 = PLOTS / "calibration_final.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── FIX 2: bootstrap band (XGB only) ─────────────────────────────────
    rng = np.random.default_rng(RANDOM_STATE)
    n_bootstrap = 500
    grid = np.linspace(0.05, 0.95, 25)
    interp_rows: list[np.ndarray] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_aug, size=n_aug, replace=True)
        try:
            pt, pp = calibration_curve(
                y_test[idx],
                xgb_proba[idx],
                n_bins=5,
                strategy="quantile",
            )
        except ValueError:
            continue
        if len(pt) < 3:
            continue
        order = np.argsort(pp)
        pp_s = pp[order]
        pt_s = pt[order]
        # Strictly increasing x for interp; jitter dupes slightly
        dpp = np.diff(pp_s)
        if (dpp <= 0).any():
            eps = 1e-6 * np.arange(len(pp_s))
            pp_s = pp_s + eps
        try:
            yi = np.interp(grid, pp_s, pt_s, left=np.nan, right=np.nan)
        except Exception:
            continue
        if np.isfinite(yi).sum() >= len(grid) // 2:
            interp_rows.append(yi)

    fig, ax = plt.subplots(figsize=(8, 7))
    if interp_rows:
        mat = np.asarray(interp_rows, dtype=np.float64)
        mean_curve = np.nanmean(mat, axis=0)
        lower = np.nanpercentile(mat, 10, axis=0)
        upper = np.nanpercentile(mat, 90, axis=0)
        ax.fill_between(
            grid,
            lower,
            upper,
            alpha=0.25,
            color="#22C55E",
            label="80% band (bootstrap)",
        )
        ax.plot(
            grid,
            mean_curve,
            "o-",
            color="#22C55E",
            linewidth=2,
            markersize=5,
            label="Bootstrap mean curve",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    brier_xgb = float(brier_score_loss(y_test, xgb_proba))
    subtitle = (
        f"Brier = {brier_xgb:.3f}  |  n = {n_aug} test rows"
        + (f"  |  {n_fights} fights" if n_fights else "")
    )
    ax.set_xlabel("Mean predicted probability", fontsize=14)
    ax.set_ylabel("Fraction of positives", fontsize=14)
    ax.set_title(
        "XGBoost calibration with uncertainty\n" + subtitle,
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    per_bin = max(1, n_aug // 5)
    ax.text(
        0.95,
        0.05,
        f"Quantile bins ≈{per_bin} rows/bin.\nWide bands are expected\nwith small n.",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
    )
    plt.tight_layout()
    out2 = PLOTS / "calibration_bootstrap.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # ── FIX 3: simple 3-bin bars ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_labels = ["Low\n(<30%)", "Medium\n(30–50%)", "High\n(≥50%)"]
    bin_colors = ["#EF4444", "#F59E0B", "#22C55E"]
    proba = xgb_proba
    b0 = proba < 0.3
    b1 = (proba >= 0.3) & (proba < 0.5)
    b2 = proba >= 0.5
    masks = [b0, b1, b2]

    rates: list[float] = []
    counts: list[int] = []
    for mask in masks:
        c = int(mask.sum())
        if c > 0:
            rates.append(float(y_test[mask].mean()))
            counts.append(c)
        else:
            rates.append(0.0)
            counts.append(0)

    x = np.arange(3)
    bars = ax.bar(x, rates, color=bin_colors, width=0.55, edgecolor="white", linewidth=1.5)
    ymax = max(rates) if rates else float(y_test.mean())
    for bar, rate, count in zip(bars, rates, counts):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02,
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    base = float(y_test.mean())
    ax.axhline(y=base, color="gray", linestyle="--", alpha=0.5, label=f"Base rate ({base:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.set_ylabel("Actual bonus rate", fontsize=14)
    ax.set_title(
        "XGBoost — does higher predicted P(bonus) match more bonuses?",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, min(ymax + 0.2, 1.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out3 = PLOTS / "calibration_simple_bars.png"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    meta = {
        "n_test_augmented": n_aug,
        "n_test_unique_fights": n_fights,
        "base_rate": base,
        "xgb_brier": brier_xgb,
        "outputs": [str(out1), str(out2), str(out3)],
        "framing": (
            "Reliability curves are high-variance with ~100–200 test rows and ~5 bins; "
            "quantile binning equalizes counts but does not remove sampling noise. "
            "Prefer the 3-bin bar chart on the main slide; use bootstrap or appendix "
            "for honesty about uncertainty."
        ),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {META_PATH}")

    print("\n" + "=" * 60)
    print("  Calibration outputs generated")
    print("=" * 60)
    print(f"  1. {out1}")
    print("     → 5 quantile bins + prediction histograms (3 models)")
    print(f"  2. {out2}")
    print("     → XGBoost bootstrap mean + 80% band")
    print(f"  3. {out3}")
    print("     → 3-bin bar chart (recommended main slide)")
    print("")
    print("  Small-sample note:")
    print(meta["framing"])
    print("=" * 60)


if __name__ == "__main__":
    main()
