"""
models/baselines.py
Baseline classification models for UFC fight entertainment prediction.

Target: is_bonus_fight (binary) — 1 if the fight earned a UFC bonus award
        (Fight of the Night / Performance of the Night), 0 otherwise.

Feature vector: 72 floats per fight
    [0:24]   Fighter A stats (physical, career, offense, defense, style, activity)
    [24:48]  Fighter B stats (same schema)
    [48:72]  Matchup cross-features (style clash, offense-vs-defense, competitive balance)

Evaluation covers both classification quality and ranking quality, because at
inference time the matchmaker scores ALL possible pairings and ranks them — so
the model's ability to produce a well-ordered ranking matters as much as its
ability to classify individual fights.

Temporal split: train = pre-2023, val = 2023, test = 2024+.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FEATURES = 72

# Features 48-55 map to style_clash_score, striker_vs_grappler, finisher_clash,
# ko_power_clash, sub_threat_clash, strike_off_vs_def, td_off_vs_def, sub_off_vs_def
# in the real feature schema (see feature_engineering.py MATCHUP_FEATURE_NAMES).
STYLE_CLASH_INDICES = list(range(48, 56))


# ─────────────────────────────────────────────────────────────────────────────
# Real data loading (requires populated DB with is_bonus_fight labels)
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(db_path: str = "data/ufc_matchmaker.db", **kwargs) -> dict:
    """
    Load 72-dim feature vectors from the real UFC database.

    Thin wrapper around models.data_loader.load_real_data() that provides
    the canonical data loading path for this codebase.

    Returns dict with: X_train, y_train, X_val, y_val, X_test, y_test,
    scaler, feature_names, event_ids_test, summary.
    """
    from models.data_loader import load_real_data as _load
    return _load(db_path=db_path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder data (for development without a populated DB)
# ─────────────────────────────────────────────────────────────────────────────

def load_placeholder_data(
    n_samples: int = 4000,
    positive_rate: float = 0.12,
    n_events_test: int = 20,
) -> dict:
    """
    Generate synthetic data shaped like the real UFC dataset.

    For real data, use load_real_data() instead — it reads from the DB
    populated by the Wikipedia bonus scraper and applies temporal splitting.

    The synthetic data has a learnable signal: positive-class samples (bonus
    fights) have elevated values in features 48-55, which correspond to style
    clash and action density cross-features in the real feature schema.

    Parameters
    ----------
    n_samples : total number of synthetic fights
    positive_rate : fraction that are bonus fights (~12% in real UFC data)
    n_events_test : number of simulated events in the test set

    Returns
    -------
    dict with keys:
        X_train, y_train     — training set (pre-2023 analog)
        X_val, y_val         — validation set (2023 analog)
        X_test, y_test       — test set (2024+ analog)
        event_ids_test       — int array mapping each test fight to an event
    """
    np.random.seed(RANDOM_STATE)

    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_val

    X_all = np.random.randn(n_samples, N_FEATURES).astype(np.float32)

    n_positive = int(n_samples * positive_rate)
    y_all = np.zeros(n_samples, dtype=np.int32)
    positive_indices = np.random.choice(n_samples, size=n_positive, replace=False)
    y_all[positive_indices] = 1

    # Primary signal: bonus fights have elevated style clash / action density
    signal_strength = 1.2
    noise_scale = 0.3
    for idx in positive_indices:
        X_all[idx, STYLE_CLASH_INDICES] += (
            signal_strength + np.random.randn(len(STYLE_CLASH_INDICES)) * noise_scale
        )

    # Secondary signal in other matchup features (finish_rate_sum, total_sig_output,
    # total_td_output, combined_finish_rate, form_clash) for realism
    secondary_indices = [62, 63, 64, 69, 71]
    for idx in positive_indices:
        X_all[idx, secondary_indices] += (
            0.6 + np.random.randn(len(secondary_indices)) * 0.2
        )

    # Temporal split — NOT random, preserving temporal ordering
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]
    X_test = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]

    # Assign event IDs to the test set (~30 fights per event, like a real UFC card)
    event_ids_test = np.zeros(n_test, dtype=np.int32)
    fights_per_event = n_test // n_events_test
    for i in range(n_events_test):
        start = i * fights_per_event
        end = (i + 1) * fights_per_event if i < n_events_test - 1 else n_test
        event_ids_test[start:end] = i

    logger.info(
        "Placeholder data loaded: train=%d (%.1f%% pos), val=%d (%.1f%% pos), "
        "test=%d (%.1f%% pos), %d test events",
        len(y_train), y_train.mean() * 100,
        len(y_val), y_val.mean() * 100,
        len(y_test), y_test.mean() * 100,
        n_events_test,
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "event_ids_test": event_ids_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    event_ids_test: Optional[np.ndarray] = None,
    ndcg_ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """
    Evaluate a trained classifier on the test set.

    Classification metrics
    ----------------------
    Accuracy, Precision, Recall, F1, AUC-ROC

    Ranking metrics (critical for matchmaker use case)
    --------------------------------------------------
    Spearman_rho        : rank correlation between predicted P(bonus) and labels
    NDCG@k              : if we pick the top k predicted matchups, how many are
                          actual bonus fights? Computed for each k in ndcg_ks.
    PerEvent_MeanBonusRank : within each UFC event, rank all fights by predicted
                             probability and report the mean rank position of
                             actual bonus fights (lower = better).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
    }

    rho, _ = spearmanr(y_prob, y_test)
    metrics["Spearman_rho"] = rho

    # NDCG treats the binary labels as relevance grades
    y_true_2d = y_test.reshape(1, -1).astype(np.float64)
    y_prob_2d = y_prob.reshape(1, -1)
    for k in ndcg_ks:
        if len(y_test) >= k:
            metrics[f"NDCG@{k}"] = ndcg_score(y_true_2d, y_prob_2d, k=k)

    if event_ids_test is not None:
        metrics["PerEvent_MeanBonusRank"] = _per_event_bonus_rank(
            y_prob, y_test, event_ids_test
        )

    return metrics


def _per_event_bonus_rank(
    y_prob: np.ndarray,
    y_test: np.ndarray,
    event_ids: np.ndarray,
) -> float:
    """
    For each event, rank fights by predicted P(bonus) descending, then report
    the mean rank position of actual bonus fights.

    Lower is better — a perfect ranker places all bonus fights at rank 1.
    Events with zero bonus fights are excluded.
    """
    unique_events = np.unique(event_ids)
    event_ranks: list[float] = []

    for eid in unique_events:
        mask = event_ids == eid
        probs = y_prob[mask]
        labels = y_test[mask]

        if labels.sum() == 0:
            continue

        # rank 1 = highest predicted probability
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        bonus_ranks = ranks[labels == 1]
        event_ranks.append(float(bonus_ranks.mean()))

    return float(np.mean(event_ranks)) if event_ranks else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

class BaselineComparison:
    """
    Train and compare LogisticRegression, RandomForest, and XGBoost baselines
    for fight entertainment prediction (is_bonus_fight classification).

    Usage
    -----
        bc = BaselineComparison()
        bc.load_data()                        # placeholder; swap with real data
        bc.train_all()                        # fits + tunes all three models
        df = bc.compare_all()                 # comparison DataFrame
        name, model = bc.get_best_model()     # best by AUC-ROC
    """

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.models: dict[str, object] = {}
        self.results: dict[str, dict[str, float]] = {}
        self._data: Optional[dict] = None

    # ── Data loading ─────────────────────────────────────────────────────

    def load_data(self, data: Optional[dict] = None) -> None:
        """
        Load data and fit a StandardScaler on the training split ONLY.

        Parameters
        ----------
        data : dict with keys X_train, y_train, X_val, y_val, X_test, y_test,
               event_ids_test.  If None, uses load_placeholder_data().
        """
        self._data = data if data is not None else load_placeholder_data()

        self.scaler = StandardScaler()
        self.scaler.fit(self._data["X_train"])

        self._data["X_train_scaled"] = self.scaler.transform(self._data["X_train"])
        self._data["X_val_scaled"] = self.scaler.transform(self._data["X_val"])
        self._data["X_test_scaled"] = self.scaler.transform(self._data["X_test"])

        logger.info(
            "Data loaded and scaled. Train=%s, Val=%s, Test=%s",
            self._data["X_train"].shape,
            self._data["X_val"].shape,
            self._data["X_test"].shape,
        )

    # ── Model factories ──────────────────────────────────────────────────

    def _build_logistic_regression(self) -> LogisticRegression:
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )

    def _build_random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def _build_xgboost(self) -> XGBClassifier:
        n_neg = int((self._data["y_train"] == 0).sum())
        n_pos = int((self._data["y_train"] == 1).sum())
        return XGBClassifier(
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

    # ── Hyperparameter tuning ────────────────────────────────────────────

    def _tune_random_forest(self) -> RandomForestClassifier:
        """RandomizedSearchCV over RF hyperparameters with TimeSeriesSplit."""
        logger.info("Tuning RandomForest (n_iter=30, 5-fold TimeSeriesSplit)...")

        param_distributions = {
            "n_estimators": [200, 500, 1000],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        search = RandomizedSearchCV(
            estimator=self._build_random_forest(),
            param_distributions=param_distributions,
            n_iter=30,
            scoring="roc_auc",
            cv=TimeSeriesSplit(n_splits=5),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self._data["X_train_scaled"], self._data["y_train"])

        logger.info(
            "RF best params: %s  (CV AUC-ROC=%.4f)",
            search.best_params_, search.best_score_,
        )
        return search.best_estimator_

    def _tune_xgboost(self) -> XGBClassifier:
        """RandomizedSearchCV over XGBoost hyperparameters with TimeSeriesSplit."""
        logger.info("Tuning XGBoost (n_iter=50, 5-fold TimeSeriesSplit)...")

        param_distributions = {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "n_estimators": [100, 300, 500],
        }

        search = RandomizedSearchCV(
            estimator=self._build_xgboost(),
            param_distributions=param_distributions,
            n_iter=50,
            scoring="roc_auc",
            cv=TimeSeriesSplit(n_splits=5),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self._data["X_train_scaled"], self._data["y_train"])

        logger.info(
            "XGB best params: %s  (CV AUC-ROC=%.4f)",
            search.best_params_, search.best_score_,
        )
        return search.best_estimator_

    # ── Training orchestration ───────────────────────────────────────────

    def train_all(self) -> None:
        """
        Train all three baselines. RF and XGBoost are hyperparameter-tuned
        via RandomizedSearchCV; LogisticRegression is trained as-is (floor baseline).
        """
        if self._data is None:
            raise RuntimeError("Call load_data() before train_all()")

        X_train = self._data["X_train_scaled"]
        y_train = self._data["y_train"]
        X_test = self._data["X_test_scaled"]
        y_test = self._data["y_test"]
        event_ids = self._data.get("event_ids_test")

        # 1. Logistic Regression — floor baseline, no tuning needed
        logger.info("Training LogisticRegression (floor baseline)...")
        lr = self._build_logistic_regression()
        lr.fit(X_train, y_train)
        self.models["LogisticRegression"] = lr
        self.results["LogisticRegression"] = evaluate_model(
            lr, X_test, y_test, event_ids
        )

        # 2. Random Forest — tuned
        rf = self._tune_random_forest()
        self.models["RandomForest"] = rf
        self.results["RandomForest"] = evaluate_model(rf, X_test, y_test, event_ids)

        # 3. XGBoost — tuned
        xgb = self._tune_xgboost()
        self.models["XGBoost"] = xgb
        self.results["XGBoost"] = evaluate_model(xgb, X_test, y_test, event_ids)

        logger.info("All baselines trained and evaluated on the test set.")

    # ── Results ──────────────────────────────────────────────────────────

    def compare_all(self) -> pd.DataFrame:
        """
        Return a DataFrame comparing all models, sorted by AUC-ROC descending.

        Rows = model names, columns = metrics.
        """
        if not self.results:
            raise RuntimeError("Call train_all() before compare_all()")

        df = pd.DataFrame(self.results).T
        df.index.name = "Model"
        return df.sort_values("AUC-ROC", ascending=False)

    def get_best_model(self) -> tuple[str, object]:
        """
        Return (model_name, fitted_model) for the model with the highest
        AUC-ROC on the test set.

        This is the model the matchmaker will use for scoring hypothetical
        fighter pairings at inference time.
        """
        if not self.results:
            raise RuntimeError("Call train_all() before get_best_model()")

        best_name = max(self.results, key=lambda m: self.results[m]["AUC-ROC"])
        return best_name, self.models[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    np.random.seed(RANDOM_STATE)

    # Try real data; fall back to placeholder if DB not available
    try:
        data = load_real_data()
        print("  [Using REAL data from DB]")
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Real data unavailable (%s), using placeholder data.", e)
        data = load_placeholder_data()
        print("  [Using PLACEHOLDER data]")

    bc = BaselineComparison()
    bc.load_data(data)
    bc.train_all()

    print("\n" + "=" * 90)
    print("  BASELINE COMPARISON — UFC Fight Entertainment Prediction (is_bonus_fight)")
    print("=" * 90)
    print(bc.compare_all().to_string(float_format="%.4f"))

    best_name, best_model = bc.get_best_model()
    best_auc = bc.results[best_name]["AUC-ROC"]
    print(f"\n  Best model: {best_name}  (AUC-ROC = {best_auc:.4f})")
    print("=" * 90)
