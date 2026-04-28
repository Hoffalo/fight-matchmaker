"""
models/nn_binary.py
Binary classification neural network for UFC fight entertainment prediction.

Replaces the regression FightQualityNN with a classifier that predicts
P(bonus fight) — the probability a matchup produces a bonus-worthy fight.

At inference time the matchmaker calls predict_proba() on every possible
fighter pairing in a weight class, then ranks by probability.  The model
therefore needs well-calibrated probabilities and strong ranking ability.

Architecture changes from the original regression NN:
  - Input dim: 115 (24+24+24 cross + 5 odds + 4 context + 15+15 rolling + 4 rolling matchup)
  - Output: raw logit (no sigmoid); BCEWithLogitsLoss handles the numerics
  - BatchNorm between layers (was LayerNorm)
  - GELU activation (kept)
  - Kaiming init with nonlinearity='leaky_relu' (closest match to GELU)
  - Early stopping on val AUC-ROC (was val loss — AUC-ROC better for ranking)
"""
import itertools
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURE_DIM

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BinaryNNConfig:
    input_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 150
    patience: int = 15
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    # Populated at training time from the data
    pos_weight: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model
# ─────────────────────────────────────────────────────────────────────────────

class FightBonusNN(nn.Module):
    """
    MLP binary classifier for fight entertainment prediction.

    Input:  FEATURE_DIM-dim matchup vector (career + cross + odds + context + rolling)
    Output: 1 raw logit (apply sigmoid externally for probability)
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, FEATURE_DIM) float tensor

        Returns
        -------
        (batch, 1) raw logits — NO sigmoid applied.
        """
        return self.head(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training
# ─────────────────────────────────────────────────────────────────────────────

def train_binary_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[BinaryNNConfig] = None,
    verbose: bool = True,
) -> tuple["FightBonusNN", dict]:
    """
    Train a FightBonusNN with BCEWithLogitsLoss, AdamW, cosine annealing LR,
    gradient clipping, and early stopping on validation AUC-ROC.

    Parameters
    ----------
    X_train, y_train : training features and binary labels
    X_val, y_val     : validation features and binary labels
    config           : BinaryNNConfig (uses defaults if None)
    verbose          : print per-epoch metrics

    Returns
    -------
    (best_model, history) where history has per-epoch metrics.
    """
    if config is None:
        config = BinaryNNConfig()

    config.input_dim = int(X_train.shape[1])
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # ── Data loaders ─────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # ── pos_weight for class imbalance ───────────────────────────────────
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    config.pos_weight = float(pos_weight.item())
    logger.info(
        "Class balance: %d pos / %d neg (ratio 1:%.1f)  → pos_weight=%.2f",
        n_pos, n_neg, n_neg / max(n_pos, 1), config.pos_weight,
    )

    # ── Model, optimiser, scheduler, criterion ───────────────────────────
    model = FightBonusNN(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.epochs, eta_min=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_auc = -1.0
    best_state: Optional[dict] = None
    patience_left = config.patience

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_auc": [], "val_spearman": [],
        "lr": [],
    }

    for epoch in range(1, config.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimiser.zero_grad()
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────
        val_loss, val_auc, val_f1, val_spearman = _validate(
            model, val_loader, criterion, device,
        )

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["val_spearman"].append(val_spearman)
        history["lr"].append(current_lr)

        # ── Logging ──────────────────────────────────────────────────────
        if verbose and (epoch <= 3 or epoch % 10 == 0 or epoch == config.epochs):
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_loss=%.4f  "
                "val_AUC=%.4f  val_F1=%.4f  val_ρ=%.4f  lr=%.2e",
                epoch, config.epochs, epoch_loss, val_loss,
                val_auc, val_f1, val_spearman, current_lr,
            )

        # ── Early stopping on AUC-ROC ────────────────────────────────────
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "Early stopping at epoch %d (no AUC-ROC improvement for %d epochs)",
                    epoch, config.patience,
                )
                break

    # ── Restore best and save checkpoint ─────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "fight_bonus_nn_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_dim": config.input_dim,
                "hidden_dims": config.hidden_dims,
                "dropout": config.dropout,
            },
            "best_val_auc": best_val_auc,
        },
        ckpt_path,
    )
    logger.info("Best model saved to %s  (val AUC-ROC=%.4f)", ckpt_path, best_val_auc)

    return model, history


def _validate(
    model: FightBonusNN,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Run validation pass, return (loss, auc_roc, f1, spearman_rho)."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b).squeeze(-1)
            total_loss += criterion(logits, y_b).item()
            all_logits.append(logits.cpu())
            all_labels.append(y_b.cpu())

    total_loss /= len(val_loader)
    logits_cat = torch.cat(all_logits).numpy()
    labels_cat = torch.cat(all_labels).numpy().astype(int)
    probs = _sigmoid_np(logits_cat)

    # AUC needs both classes present
    if len(np.unique(labels_cat)) < 2:
        return total_loss, 0.5, 0.0, 0.0

    auc = roc_auc_score(labels_cat, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels_cat, preds, zero_division=0)
    rho, _ = spearmanr(probs, labels_cat)

    return total_loss, auc, f1, float(rho)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_combos: int = 30,
) -> tuple[BinaryNNConfig, "FightBonusNN"]:
    """
    Random hyperparameter sweep over architecture and optimiser settings.

    Samples n_combos random configurations from the search space, trains each
    to completion (with early stopping), and returns the best by val AUC-ROC.

    Prints a top-10 leaderboard on completion.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    search_space = {
        "hidden_dims": [(256, 128, 64), (128, 64, 32), (64, 32)],
        "dropout": [0.2, 0.3, 0.5],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [32, 64],
    }

    # Enumerate all combinations, sample a random subset
    keys = list(search_space.keys())
    all_combos = list(itertools.product(*(search_space[k] for k in keys)))
    random.shuffle(all_combos)
    combos = all_combos[:n_combos]

    logger.info(
        "Sweep: %d / %d total combinations (search space size = %d)",
        len(combos), len(all_combos), len(all_combos),
    )

    results: list[tuple[float, BinaryNNConfig, FightBonusNN]] = []

    for i, values in enumerate(combos, 1):
        cfg = BinaryNNConfig(**dict(zip(keys, values)))
        tag = (
            f"[{i}/{len(combos)}] dims={cfg.hidden_dims} "
            f"drop={cfg.dropout} lr={cfg.learning_rate} "
            f"wd={cfg.weight_decay} bs={cfg.batch_size}"
        )
        logger.info("Sweep %s", tag)

        model, history = train_binary_nn(
            X_train, y_train, X_val, y_val,
            config=cfg, verbose=False,
        )

        best_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
        results.append((best_auc, cfg, model))
        logger.info("  → val AUC-ROC = %.4f  (epochs ran: %d)", best_auc, len(history["val_auc"]))

    # ── Leaderboard ──────────────────────────────────────────────────────
    results.sort(key=lambda r: r[0], reverse=True)

    print("\n" + "=" * 90)
    print("  HYPERPARAMETER SWEEP — Top 10 Configurations")
    print("=" * 90)
    print(f"  {'Rank':<5} {'AUC-ROC':<10} {'Hidden Dims':<20} {'Drop':<6} "
          f"{'LR':<10} {'WD':<10} {'BS':<4}")
    print("-" * 90)
    for rank, (auc, cfg, _) in enumerate(results[:10], 1):
        print(
            f"  {rank:<5} {auc:<10.4f} {str(cfg.hidden_dims):<20} "
            f"{cfg.dropout:<6.2f} {cfg.learning_rate:<10.1e} "
            f"{cfg.weight_decay:<10.1e} {cfg.batch_size:<4}"
        )
    print("=" * 90)

    best_auc, best_config, best_model = results[0]
    logger.info("Best config: %s  (AUC-ROC=%.4f)", best_config, best_auc)

    return best_config, best_model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_proba(
    model: FightBonusNN,
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """
    Score matchups and return P(bonus fight) for each row.

    This is the function the matchmaker calls at inference time to rank
    all possible fighter pairings in a weight class.

    Parameters
    ----------
    model  : trained FightBonusNN
    X      : (N, FEATURE_DIM) feature array (raw or pre-scaled)
    scaler : if provided, applies transform before inference

    Returns
    -------
    (N,) numpy array of probabilities in [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device

    if scaler is not None:
        X = scaler.transform(X)

    with torch.no_grad():
        tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tensor).squeeze(-1)
        return _sigmoid_np(logits.cpu().numpy())


def load_binary_nn(path: str) -> FightBonusNN:
    """Load a trained FightBonusNN from a checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    model = FightBonusNN(
        input_dim=cfg["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.info(
        "Loaded FightBonusNN from %s  (val AUC-ROC=%.4f)",
        path, checkpoint.get("best_val_auc", float("nan")),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _generate_placeholder_data() -> dict:
    """
    Synthetic data (FEATURE_DIM) with learnable signal for testing.

    For real data, use models.data_loader.load_real_data() which queries
    the populated DB and applies proper temporal splitting + augmentation.
    """
    np.random.seed(RANDOM_STATE)

    n_samples = 4000
    positive_rate = 0.12
    n_features = FEATURE_DIM

    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    n_pos = int(n_samples * positive_rate)
    pos_idx = np.random.choice(n_samples, size=n_pos, replace=False)
    y[pos_idx] = 1.0

    X[pos_idx, 48:56] += 1.2 + np.random.randn(n_pos, 8) * 0.3
    for col in [62, 63, 64, 69, 71]:
        X[pos_idx, col] += 0.6 + np.random.randn(n_pos) * 0.2
    # Odds features (indices 72-76): close lines correlate with bonus fights
    X[pos_idx, 72] += 0.8 + np.random.randn(n_pos) * 0.2  # odds_closeness
    X[pos_idx, 74] += 0.5  # is_close_line
    # Context (77-80): marquee / main-event / five-round / finish-heavy divisions
    X[pos_idx, 77] += 0.4
    X[pos_idx, 78] += 0.35
    X[pos_idx, 79] += 0.25
    X[pos_idx, 80] += 0.15
    if n_features > 81:
        X[pos_idx, 81:n_features] += 0.25 + np.random.randn(n_pos, n_features - 81) * 0.08

    scaler = StandardScaler().fit(X[:n_train])
    X = scaler.transform(X).astype(np.float32)

    return {
        "X_train": X[:n_train], "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val], "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:], "y_test": y[n_train + n_val:],
        "scaler": scaler,
    }


def _load_data() -> dict:
    """Try real data first; fall back to synthetic placeholder."""
    try:
        from models.data_loader import load_real_data
        data = load_real_data()
        logger.info("Using REAL data from DB")
        return data
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.warning("Real data unavailable (%s), using placeholder.", e)
        return _generate_placeholder_data()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    data = _load_data()

    # ── 1. Single training run with default config ───────────────────────
    print("\n=== Single training run (default config) ===")
    model, history = train_binary_nn(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    best_epoch = int(np.argmax(history["val_auc"]))
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"  val AUC-ROC  = {history['val_auc'][best_epoch]:.4f}")
    print(f"  val F1       = {history['val_f1'][best_epoch]:.4f}")
    print(f"  val Spearman = {history['val_spearman'][best_epoch]:.4f}")

    # ── 2. Inference demo ────────────────────────────────────────────────
    print("\n=== Inference demo ===")
    probs = predict_proba(model, data["X_test"])
    test_auc = roc_auc_score(data["y_test"].astype(int), probs)
    print(f"Test set AUC-ROC: {test_auc:.4f}")
    print(f"Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Mean P(bonus) for actual bonus fights:     {probs[data['y_test'] == 1].mean():.4f}")
    print(f"Mean P(bonus) for non-bonus fights:        {probs[data['y_test'] == 0].mean():.4f}")

    # ── 3. Hyperparameter sweep (small — 10 combos for demo speed) ──────
    print("\n=== Hyperparameter sweep (10 combos for demo) ===")
    best_cfg, best_model = sweep_nn(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        n_combos=10,
    )

    probs_best = predict_proba(best_model, data["X_test"])
    sweep_auc = roc_auc_score(data["y_test"].astype(int), probs_best)
    print(f"\nSweep winner test AUC-ROC: {sweep_auc:.4f}")
