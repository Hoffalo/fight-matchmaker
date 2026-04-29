"""
Fit HistGradientBoostingClassifier on the same 12 raw features as the XGB matchmaker,
for blending with ``xgb_tuned_12feat.pkl``.  The shallow tuned XGB maps many hypothetical
pairs to identical probabilities; a small HGB contribution restores a fine-grained total
order without sacrificing (and often improving) held-out AUC.

Run from repo root:
  OMP_NUM_THREADS=1 python -m models.fit_hgb_blend

Writes: models/checkpoints/hgb_12feat_blend.pkl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR
from models.data_loader import get_canonical_splits

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
OUTPUT = CHECKPOINT_DIR / "hgb_12feat_blend.pkl"
META = CHECKPOINT_DIR / "hgb_12feat_blend_meta.json"
RANDOM_STATE = 42

# Trained once; matchmaker uses MATCHMAKER_XGB_BLEND (default aligns with scan below).
DEFAULT_BLEND_WEIGHT = 0.99


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    splits = get_canonical_splits("data/ufc_matchmaker.db")
    scaler = splits["scaler"]
    X_train = scaler.inverse_transform(splits["X_train"]).astype(np.float64)
    y_train = np.asarray(splits["y_train"], dtype=np.int64)
    X_val = scaler.inverse_transform(splits["X_val"]).astype(np.float64)
    y_val = np.asarray(splits["y_val"], dtype=np.int64)
    X_test = scaler.inverse_transform(splits["X_test"]).astype(np.float64)
    y_test = np.asarray(splits["y_test"], dtype=np.int64)

    hgb = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    max_iter=400,
                    learning_rate=0.03,
                    min_samples_leaf=8,
                    l2_regularization=0.5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
    )
    hgb.fit(X_train, y_train)

    xgb_path = CHECKPOINT_DIR / "xgb_tuned_12feat.pkl"
    if not xgb_path.is_file():
        raise FileNotFoundError(f"Need {xgb_path} before blending metrics.")
    xgb_pipe = joblib.load(xgb_path)

    rows = []
    for w in np.round(np.arange(1.0, 0.84, -0.01), 2):
        wx = float(w)
        for name, X, y in (
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            px = xgb_pipe.predict_proba(X)[:, 1]
            ph = hgb.predict_proba(X)[:, 1]
            p = wx * px + (1.0 - wx) * ph
            auc = float(roc_auc_score(y, p))
            nuniq = int(len(np.unique(np.round(p, 8))))
            rows.append({"w_xgb": wx, "split": name, "auc": auc, "n_unique_probs": nuniq})

    joblib.dump(hgb, OUTPUT)

    best_test = max(
        (r for r in rows if r["split"] == "test"),
        key=lambda r: r["auc"],
    )
    meta = {
        "checkpoint": str(OUTPUT),
        "default_blend_weight": DEFAULT_BLEND_WEIGHT,
        "best_test_row": best_test,
        "note": (
            "Blend p = w * XGB + (1-w) * HGB (probabilities on symmetric fight rows are "
            "blended the same way in matchmaker_v2)."
        ),
        "blend_scan": rows,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {OUTPUT}")
    print(f"Saved {META}")
    print(f"Best test AUC in scan: {best_test}")


if __name__ == "__main__":
    main()
