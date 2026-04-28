"""
PCA dimensionality analysis for the 72-dim matchup feature space.

Uses the same scaled training matrix as classifiers (StandardScaler fit on train).
Reports cumulative explained variance and PC1 loadings — useful for redundancy checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def run_pca_analysis(
    X_train: np.ndarray,
    *,
    feature_names: Optional[list[str]] = None,
    max_components: int | None = None,
    variance_thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict:
    """
    Fit PCA on training features (typically StandardScaler-transformed).

    Parameters
    ----------
    X_train : (N, 72) float array
    feature_names : optional 72 names for PC1 loading report
    max_components : defaults to min(n_samples, n_features)
    """
    X = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0)
    n_samples, n_feat = X.shape
    n_comp = max_components if max_components is not None else min(n_samples, n_feat)
    n_comp = max(1, min(n_comp, n_samples, n_feat))

    if n_samples < 2:
        raise ValueError(
            "PCA needs at least 2 training samples after augmentation. "
            "Add more dated fights to the DB (see data/seed_minimal_splits_db.py) "
            "or run: python main.py collect"
        )

    pca = PCA(n_components=n_comp, random_state=42, svd_solver="full")
    pca.fit(X)

    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    evr = np.nan_to_num(evr, nan=0.0, posinf=0.0, neginf=0.0)
    if evr.sum() > 0:
        evr = evr / evr.sum()
    cum = np.cumsum(evr)

    threshold_hits: dict[str, int] = {}
    for t in variance_thresholds:
        idx = int(np.searchsorted(cum, t, side="left"))
        threshold_hits[f"n_components_{int(t * 100)}pct_var"] = min(idx + 1, len(cum))

    out: dict = {
        "n_samples": n_samples,
        "n_features": n_feat,
        "n_components_fitted": n_comp,
        "explained_variance_ratio": evr,
        "cumulative_variance": cum,
        **threshold_hits,
    }

    if feature_names is not None and len(feature_names) == n_feat:
        loadings = pca.components_[0]
        order = np.argsort(np.abs(loadings))[::-1][:15]
        out["pc1_top_features"] = [
            (feature_names[i], float(loadings[i])) for i in order
        ]

    return out


def format_pca_report(result: dict) -> str:
    """Human-readable summary for stdout."""
    lines = [
        "",
        "=" * 72,
        "  PCA — cumulative variance (scaled 72-dim train features)",
        "=" * 72,
        f"  Samples: {result['n_samples']}   Features: {result['n_features']}",
        f"  PCs fitted: {result['n_components_fitted']}",
        "",
    ]
    evr = np.asarray(result["explained_variance_ratio"], dtype=np.float64)
    cum = np.asarray(result["cumulative_variance"], dtype=np.float64)
    parts = []
    for i, label in enumerate(("PC1", "PC2", "PC3")):
        if i < len(evr):
            parts.append(f"{label}: {100 * evr[i]:.1f}%")
    lines.append("  " + "   ".join(parts) if parts else "  (no variance ratios)")
    if result["n_samples"] < 10:
        lines.append(
            f"  Note: only {result['n_samples']} train samples — "
            "percentages are noisy; prefer ≥30 rows for stable PCA."
        )
    if len(cum) > 0:
        i10 = min(9, len(cum) - 1)
        lines.append(f"  Cumulative after PC10 (or max): {100 * cum[i10]:.1f}%")
    for key, val in sorted(result.items()):
        if key.startswith("n_components_") and key.endswith("pct_var"):
            pct = key.replace("n_components_", "").replace("pct_var", "")
            lines.append(f"  Components for ~{pct}% cumulative variance: {val}")
    top = result.get("pc1_top_features")
    if top:
        lines.append("")
        lines.append("  Largest |loadings| on PC1:")
        for name, ld in top[:10]:
            lines.append(f"    {name:<44} {ld:+.4f}")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def run_pca_from_db(db_path: str = "data/ufc_matchmaker.db") -> dict:
    """
    Load canonical train split (scaled) and run PCA — entry point for CLI.
    """
    from models.data_loader import get_canonical_splits

    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    splits = get_canonical_splits(db_path=str(path))
    X_train = splits["X_train"]
    names = splits.get("feature_names") or []

    result = run_pca_analysis(
        X_train,
        feature_names=list(names) if names else None,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        r = run_pca_from_db()
        print(format_pca_report(r))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
