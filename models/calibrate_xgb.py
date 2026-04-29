"""
Post-hoc probability calibration for any prefit XGB pipeline.

Wraps a saved ``Pipeline(scaler, XGBClassifier)`` with both isotonic and sigmoid
(Platt) calibration fit on the val set, picks sigmoid by default (more stable
on small val sets — see code comment), saves the calibrated wrapper, and writes
a before/after calibration plot.

Run:  python -m models.calibrate_xgb
      python -m models.calibrate_xgb xgb_tuned_12feat_robust.pkl
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from config import BASE_DIR, MODELS_DIR
from models.data_loader import get_canonical_splits

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
DEFAULT_SRC_NAME = "xgb_tuned_12feat.pkl"


def _calibrated_dst(src_path: Path) -> Path:
    # xgb_tuned_12feat.pkl  →  xgb_tuned_12feat_calibrated.pkl
    return src_path.with_name(f"{src_path.stem}_calibrated{src_path.suffix}")


def _fit_calibrator(pipe, X_val, y_val, method: str):
    # FrozenEstimator keeps the base pipeline's fit state intact (replaces the
    # removed cv='prefit' API in sklearn>=1.6).
    cal = CalibratedClassifierCV(estimator=FrozenEstimator(pipe), method=method, cv=None)
    cal.fit(X_val, y_val)
    return cal


def main(
    src_name: str = DEFAULT_SRC_NAME,
    db_path: str = "data/ufc_matchmaker.db",
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    src_path = CHECKPOINT_DIR / src_name
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Missing {src_path}. Train it first: python -m models.xgb_tuning"
        )
    dst_path = _calibrated_dst(src_path)

    splits = get_canonical_splits(db_path)
    scaler = splits["scaler"]
    # The saved pipeline scales internally, so we feed it raw features.
    X_val = scaler.inverse_transform(splits["X_val"]).astype(np.float64)
    X_test = scaler.inverse_transform(splits["X_test"]).astype(np.float64)
    y_val = np.asarray(splits["y_val"], dtype=np.int64)
    y_test = np.asarray(splits["y_test"], dtype=np.int64)

    pipe = joblib.load(src_path)
    print(f"Calibrating: {src_path.name}")

    pre_val = pipe.predict_proba(X_val)[:, 1]
    pre_test = pipe.predict_proba(X_test)[:, 1]

    cal_iso = _fit_calibrator(pipe, X_val, y_val, "isotonic")
    cal_sig = _fit_calibrator(pipe, X_val, y_val, "sigmoid")

    iso_val = cal_iso.predict_proba(X_val)[:, 1]
    iso_test = cal_iso.predict_proba(X_test)[:, 1]
    sig_val = cal_sig.predict_proba(X_val)[:, 1]
    sig_test = cal_sig.predict_proba(X_test)[:, 1]

    rows = [
        ("uncalibrated", pre_val, pre_test),
        ("isotonic",     iso_val, iso_test),
        ("sigmoid",      sig_val, sig_test),
    ]

    print()
    print("=" * 64)
    print("  Brier score (lower=better)            AUC (should be ~unchanged)")
    print(f"  {'method':<14} {'val':>8} {'test':>8}     {'val':>8} {'test':>8}")
    print("=" * 64)
    for name, v, t in rows:
        print(
            f"  {name:<14} "
            f"{brier_score_loss(y_val, v):>8.4f} {brier_score_loss(y_test, t):>8.4f}     "
            f"{roc_auc_score(y_val, v):>8.4f} {roc_auc_score(y_test, t):>8.4f}"
        )
    print("=" * 64)

    # Default to sigmoid (Platt): with only ~340 val rows, isotonic overfits
    # the val curve and can degrade test AUC, while sigmoid preserves AUC and
    # achieves equivalent test Brier. Override only if isotonic improves val
    # Brier by a clear margin (>0.01).
    iso_val_brier = brier_score_loss(y_val, iso_val)
    sig_val_brier = brier_score_loss(y_val, sig_val)
    if iso_val_brier < sig_val_brier - 0.01:
        chosen_name, chosen_model, chosen_test = "isotonic", cal_iso, iso_test
    else:
        chosen_name, chosen_model, chosen_test = "sigmoid", cal_sig, sig_test
    print(f"\nSelected method: {chosen_name} "
          f"(val Brier iso={iso_val_brier:.4f}, sig={sig_val_brier:.4f})")

    joblib.dump(chosen_model, dst_path)
    print(f"Saved: {dst_path}")

    fig, ax = plt.subplots(figsize=(8, 7))
    base_rate = float(y_test.mean())

    series = [
        ("Uncalibrated XGB", pre_test, "#9CA3AF"),
        ("Isotonic-calibrated", iso_test, "#22C55E"),
        ("Sigmoid-calibrated", sig_test, "#3B82F6"),
    ]
    for name, proba, color in series:
        prob_true, prob_pred = calibration_curve(
            y_test, proba, n_bins=5, strategy="quantile",
        )
        brier = brier_score_loss(y_test, proba)
        ax.plot(
            prob_pred, prob_true, "o-", linewidth=2, markersize=7,
            color=color, label=f"{name}  (Brier={brier:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.axvline(base_rate, color="red", alpha=0.4, linestyle=":", label=f"Base rate ({base_rate:.0%})")
    ax.set_xlabel("Mean predicted probability", fontsize=13)
    ax.set_ylabel("Fraction of actual positives", fontsize=13)
    ax.set_title(
        f"Calibration before vs. after — {src_path.stem} (test n={len(y_test)})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / f"calibration_before_after_{src_path.stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC_NAME)
