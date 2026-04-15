"""
models/training.py
═══════════════════════════════════════════════════════════════════════════════
Training pipeline for the FightQualityNN.

Flow:
  1. Load all historical fights with computed quality scores from the DB
  2. Build (fighter_A_features || fighter_B_features) input vectors
  3. Use the fight_quality_score column as the regression target
  4. Train with MSE loss + L2 regularisation (weight_decay)
  5. Early-stopping on validation loss
  6. Save best model weights + feature scaler to disk

Why both orderings?
  A fight between Fighter X and Fighter Y is the same fight regardless of which
  corner they were in. We therefore add BOTH (A, B) and (B, A) vectors pointing
  at the same target score — this doubles the training set and teaches the model
  that matchup quality is symmetric.

Targets are normalised to [0, 1] for the Sigmoid output head, then multiplied
by 100 when producing human-readable scores.
═══════════════════════════════════════════════════════════════════════════════
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

from data.db import Database
from models.feature_engineering import build_matchup_vector
from models.fight_quality_nn import FightQualityNN
from config import NN

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(db: Database) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the DB and build (X, y) arrays for training.

    Returns
    -------
    X : float32 array of shape (N, 48)
        Each row is a matchup feature vector.
    y : float32 array of shape (N,)
        Each value is a fight quality score normalised to [0, 1].
    """
    logger.info("Building training dataset from DB...")

    # Only use fights that have a computed quality score AND both fighter IDs
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT
                f.id,
                f.fighter1_id,
                f.fighter2_id,
                f.fight_quality_score,
                e.date
            FROM fights f
            LEFT JOIN events e ON f.event_id = e.id
            WHERE f.fight_quality_score IS NOT NULL
              AND f.fighter1_id IS NOT NULL
              AND f.fighter2_id IS NOT NULL
            ORDER BY e.date DESC
            """
        ).fetchall()

    rows = [dict(r) for r in rows]
    logger.info("Found %d fights with quality scores", len(rows))

    if not rows:
        raise ValueError(
            "No fights with quality scores found in DB. "
            "Run the data pipeline first: python main.py collect"
        )

    # Build a fighter lookup cache to avoid N×2 DB hits
    logger.info("Loading fighter profiles into memory cache...")
    fighters_cache: dict[int, dict] = {}
    with db.connect() as conn:
        for row in conn.execute("SELECT * FROM fighters").fetchall():
            fighters_cache[row["id"]] = dict(row)

    X_rows, y_rows = [], []

    for fight in rows:
        f1 = fighters_cache.get(fight["fighter1_id"])
        f2 = fighters_cache.get(fight["fighter2_id"])

        # Skip if we don't have profile data for either fighter
        if f1 is None or f2 is None:
            continue

        score_norm = fight["fight_quality_score"] / 100.0  # → [0, 1]

        # Add both orderings so the model learns symmetry
        vec_ab = build_matchup_vector(f1, f2)
        vec_ba = build_matchup_vector(f2, f1)

        X_rows.append(vec_ab)
        y_rows.append(score_norm)
        X_rows.append(vec_ba)
        y_rows.append(score_norm)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    logger.info(
        "Dataset ready: %d samples, %d features. Score range: %.1f – %.1f",
        len(X), X.shape[1], y.min() * 100, y.max() * 100
    )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Scaler: standardise input features to zero mean / unit variance
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray, save_path: str = None) -> StandardScaler:
    """
    Fit a StandardScaler on the training data and optionally persist it.
    The same scaler MUST be used at inference time.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler saved to %s", save_path)
    return scaler


def load_scaler(path: str) -> StandardScaler:
    """Load a previously fitted scaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(db: Database, cfg: dict = None) -> FightQualityNN:
    """
    Full training loop.

    Parameters
    ----------
    db  : Database instance with populated fights + fighters
    cfg : Optional override dict for NN hyperparameters (see config.NN)

    Returns the best (lowest val loss) trained model.
    """
    if cfg is None:
        cfg = NN

    # ── 1. Build dataset ─────────────────────────────────────────────────
    X, y = build_training_dataset(db)

    # ── 2. Fit + apply feature scaler ────────────────────────────────────
    scaler = fit_scaler(X, save_path=cfg["scaler_save_path"])
    X_scaled = scaler.transform(X).astype(np.float32)

    # ── 3. Split into train / validation ─────────────────────────────────
    dataset   = TensorDataset(
        torch.from_numpy(X_scaled),
        torch.from_numpy(y),
    )
    val_size  = max(1, int(len(dataset) * cfg["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    logger.info("Train: %d samples  |  Val: %d samples", train_size, val_size)

    # ── 4. Instantiate model, optimiser, scheduler ───────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    model = FightQualityNN(
        input_dim=cfg["input_dim"],
        hidden_layers=cfg["hidden_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Cosine annealing LR schedule — gently reduces LR over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=1e-5
    )

    # Huber loss is less sensitive to outlier scores than pure MSE
    criterion = nn.HuberLoss(delta=0.1)

    # ── 5. Training loop ──────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train phase ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            preds = model(X_batch).squeeze(-1)   # (batch,)
            loss  = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validation phase ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds   = model(X_batch).squeeze(-1)
                val_loss += criterion(preds, y_batch).item()
        val_loss /= len(val_loader)

        # ── Logging ──────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            # Convert Huber loss → approximate RMSE on 0-100 scale for readability
            rmse_approx = (val_loss ** 0.5) * 100
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  (~RMSE %.1f pts)  lr=%.2e",
                epoch, cfg["epochs"], train_loss, val_loss, rmse_approx,
                scheduler.get_last_lr()[0],
            )

        # ── Early stopping ────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch, cfg["early_stopping_patience"])
                break

    # ── 6. Restore + save best model ────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = cfg["model_save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":   model.state_dict(),
            "config":        cfg,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )
    logger.info("✓ Best model saved to %s  (val_loss=%.4f)", save_path, best_val_loss)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (used by matchmaker at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str = None, cfg: dict = None) -> FightQualityNN:
    """
    Load a trained model from disk.

    Parameters
    ----------
    path : Path to the .pt checkpoint. Defaults to config.NN["model_save_path"].
    cfg  : NN config dict. If None, read from checkpoint.
    """
    if path is None:
        path = NN["model_save_path"]
    if cfg is None:
        cfg = NN

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg  = checkpoint.get("config", cfg)

    model = FightQualityNN(
        input_dim=saved_cfg["input_dim"],
        hidden_layers=saved_cfg["hidden_layers"],
        dropout=saved_cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.info("Model loaded from %s  (val_loss=%.4f)",
                path, checkpoint.get("best_val_loss", float("nan")))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: FightQualityNN, db: Database, cfg: dict = None) -> dict:
    """
    Run the model on the full dataset and report regression metrics.

    Returns dict with keys: mae, rmse, r2
    """
    if cfg is None:
        cfg = NN

    X, y = build_training_dataset(db)
    scaler = load_scaler(cfg["scaler_save_path"])
    X_scaled = scaler.transform(X).astype(np.float32)

    preds = model.predict_batch(X_scaled) / 100.0  # back to [0,1]
    errors = preds - y

    mae  = float(np.abs(errors).mean() * 100)
    rmse = float(np.sqrt((errors ** 2).mean()) * 100)

    # R² = 1 - SS_res / SS_tot
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-9)

    logger.info("Evaluation  MAE=%.2f pts  RMSE=%.2f pts  R²=%.3f", mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}
