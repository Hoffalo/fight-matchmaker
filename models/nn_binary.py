"""
models/nn_binary.py
Binary classification neural network for UFC fight entertainment prediction.

For small-data 12 RFECV features use ``run_twelve_feature_comparison()``;
for PCA-compressed inputs (e.g. 10 components), use ``run_pca10_nn_comparison()``
after ``python -m models.pca_pipeline`` has written ``pca_transformer.pkl``.

Architecture notes:
  - Raw logit output (no sigmoid); BCEWithLogitsLoss or pairwise margin on logits
  - BatchNorm + GELU + Dropout; Kaiming init on Linear layers
"""
from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURE_DIM, SCALE_POS_WEIGHT

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
PCA_TRANSFORMER_PATH = CHECKPOINT_DIR / "pca_transformer.pkl"

NN_12_FEAT_VAL_REF = 0.5991  # best small-data sweep on 12 RFECV features (user-reported)
LOGREG_CV_REF = 0.5945
INPUT_NOISE_STD = 0.05
PAIRWISE_MARGIN = 0.5
COSINE_T_MAX = 200
MAX_EPOCHS_SMALL = 300
PATIENCE_SMALL = 25

LossKind = Literal["bce", "pairwise"]


@dataclass
class BinaryNNConfig:
    input_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (16, 8)
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 32
    epochs: int = MAX_EPOCHS_SMALL
    patience: int = PATIENCE_SMALL
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    pos_weight: float = float(SCALE_POS_WEIGHT)
    cosine_t_max: int = COSINE_T_MAX
    input_noise_std: float = INPUT_NOISE_STD
    grad_clip: float = 1.0
    loss: LossKind = "bce"
    # Populated from data if loss is bce and use_batch_pos_weight True
    use_batch_pos_weight: bool = False


class FightBonusNN(nn.Module):
    """
    Tiny MLP for P(bonus): 12→…→1 logits. Keep ``hidden_dims`` small for n≈338.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: tuple[int, ...] = (16, 8),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
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
        return self.network(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    margin: float = PAIRWISE_MARGIN,
    max_pairs: int = 64,
) -> torch.Tensor | None:
    """Hinge on logits: max(0, margin - (s+ - s-)). Returns None if no mixed batch."""
    y = y.long()
    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (y == 0).nonzero(as_tuple=True)[0]
    if pos_idx.numel() < 1 or neg_idx.numel() < 1:
        return None
    n = min(int(pos_idx.numel()), int(neg_idx.numel()), max_pairs)
    device = logits.device
    ip = pos_idx[torch.randint(0, pos_idx.numel(), (n,), device=device)]
    ine = neg_idx[torch.randint(0, neg_idx.numel(), (n,), device=device)]
    s_pos = logits[ip]
    s_neg = logits[ine]
    return F.relu(margin - (s_pos - s_neg)).mean()


def train_binary_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[BinaryNNConfig] = None,
    verbose: bool = True,
) -> tuple[FightBonusNN, dict]:
    """
    Train FightBonusNN with AdamW, cosine annealing, grad clipping, optional input
    noise (training), early stopping on val ROC-AUC.
    """
    if config is None:
        config = BinaryNNConfig()

    config.input_dim = int(X_train.shape[1])
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    batch_pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
    )
    if config.use_batch_pos_weight:
        pos_w_tensor = batch_pos_weight
        config.pos_weight = float(batch_pos_weight.item())
    else:
        pos_w_tensor = torch.tensor([config.pos_weight], dtype=torch.float32, device=device)

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
        optimiser, T_max=config.cosine_t_max, eta_min=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

    best_val_auc = -1.0
    best_state: Optional[dict] = None
    best_epoch = 0
    patience_left = config.patience

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_auc": [],
        "val_spearman": [],
        "lr": [],
    }

    noise_std = config.input_noise_std

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if noise_std > 0:
                X_b = X_b + noise_std * torch.randn_like(X_b)

            optimiser.zero_grad()
            logits = model(X_b).squeeze(-1)

            if config.loss == "pairwise":
                p_loss = _pairwise_ranking_loss(logits, y_b)
                if p_loss is None:
                    continue
                p_loss.backward()
            else:
                loss = criterion(logits, y_b)
                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimiser.step()
            epoch_loss += (
                p_loss.item() if config.loss == "pairwise" else loss.item()
            )
            n_batches += 1

        if n_batches > 0:
            epoch_loss /= n_batches
        scheduler.step()

        val_loss, val_auc, val_f1, val_spearman = _validate(
            model, val_loader, criterion, device,
        )

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["val_spearman"].append(val_spearman)
        history["lr"].append(scheduler.get_last_lr()[0])

        if verbose and (epoch <= 3 or epoch % 25 == 0 or epoch == config.epochs):
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_loss=%.4f  val_AUC=%.4f  loss=%s",
                epoch, config.epochs, epoch_loss, val_loss, val_auc, config.loss,
            )

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "Early stopping at epoch %d (best val AUC %.4f @ epoch %d)",
                    epoch, best_val_auc, best_epoch,
                )
                break

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
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    logger.info("Checkpoint %s  val AUC=%.4f", ckpt_path, best_val_auc)

    return model, history


def _validate(
    model: FightBonusNN,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
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

    total_loss /= max(len(val_loader), 1)
    logits_cat = torch.cat(all_logits).numpy()
    labels_cat = torch.cat(all_labels).numpy().astype(int)
    probs = _sigmoid_np(logits_cat)

    if len(np.unique(labels_cat)) < 2:
        return total_loss, 0.5, 0.0, 0.0

    auc = roc_auc_score(labels_cat, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels_cat, preds, zero_division=0)
    rho, _ = spearmanr(probs, labels_cat)

    return total_loss, auc, f1, float(rho)


SMALL_DATA_SWEEP: list[dict] = [
    {"hidden_dims": (16, 8), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05, "batch_size": 32, "loss": "bce"},
    {"hidden_dims": (16, 8), "dropout": 0.6, "lr": 5e-4, "weight_decay": 0.05, "batch_size": 64, "loss": "bce"},
    {"hidden_dims": (8,), "dropout": 0.4, "lr": 1e-3, "weight_decay": 0.03, "batch_size": 32, "loss": "bce"},
    {"hidden_dims": (24, 12), "dropout": 0.5, "lr": 5e-4, "weight_decay": 0.05, "batch_size": 32, "loss": "bce"},
    {"hidden_dims": (24, 12), "dropout": 0.6, "lr": 1e-3, "weight_decay": 0.1, "batch_size": 64, "loss": "bce"},
    {"hidden_dims": (16,), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05, "batch_size": 32, "loss": "bce"},
    {"hidden_dims": (32, 16), "dropout": 0.6, "lr": 5e-4, "weight_decay": 0.05, "batch_size": 32, "loss": "bce"},
    {"hidden_dims": (32, 16), "dropout": 0.7, "lr": 1e-4, "weight_decay": 0.1, "batch_size": 64, "loss": "bce"},
    {"hidden_dims": (16, 8), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05, "batch_size": 32, "loss": "pairwise"},
]

# NN on PCA-10 (from ``pca_pipeline`` checkpoint); batch_size defaults to 32 in ``_config_from_sweep_dict``.
PCA10_NN_SWEEP: list[dict] = [
    {"hidden_dims": (16, 8), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05},
    {"hidden_dims": (16, 8), "dropout": 0.6, "lr": 5e-4, "weight_decay": 0.05},
    {"hidden_dims": (16,), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05},
    {"hidden_dims": (8,), "dropout": 0.4, "lr": 1e-3, "weight_decay": 0.03},
    {"hidden_dims": (24, 12), "dropout": 0.5, "lr": 5e-4, "weight_decay": 0.05},
    {"hidden_dims": (32, 16), "dropout": 0.5, "lr": 5e-4, "weight_decay": 0.05},
    {"hidden_dims": (16, 8), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05, "loss": "pairwise"},
    {"hidden_dims": (16,), "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.05, "loss": "pairwise"},
]


def _config_from_sweep_dict(d: dict) -> BinaryNNConfig:
    lk: LossKind = d.get("loss", "bce")  # type: ignore[assignment]
    return BinaryNNConfig(
        hidden_dims=tuple(d["hidden_dims"]),
        dropout=d["dropout"],
        learning_rate=d["lr"],
        weight_decay=d["weight_decay"],
        batch_size=int(d.get("batch_size", 32)),
        epochs=MAX_EPOCHS_SMALL,
        patience=PATIENCE_SMALL,
        cosine_t_max=COSINE_T_MAX,
        pos_weight=float(SCALE_POS_WEIGHT),
        use_batch_pos_weight=False,
        input_noise_std=INPUT_NOISE_STD,
        loss=lk,
    )


def _try_load_xgb_cv_auc() -> Optional[float]:
    p = CHECKPOINT_DIR / "xgb_tuned_metrics.json"
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text()).get("best_cv_auc"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def print_architecture_comparison(
    *,
    nn_val_auc: float,
    nn_params: int,
    xgb_cv_auc: Optional[float] = None,
    logreg_ref: float = LOGREG_CV_REF,
) -> None:
    xgb_str = f"{xgb_cv_auc:.4f}" if xgb_cv_auc is not None else "— (run xgb_tuning + metrics json)"
    print()
    print(f"{'Model':<28} {'Val/CV AUC':<14} {'Params':<12}")
    print(f"{'─' * 28}   {'─' * 12}   {'─' * 8}")
    print(f"{'LogReg (balanced, ref)':<28} {logreg_ref:<14.4f} {'12':<12}")
    print(f"{'XGBoost (tuned, CV)':<28} {xgb_str:<14} {'~100–500':<12}")
    print(f"{'NN (best sweep)':<28} {nn_val_auc:<14.4f} {str(nn_params):<12}")

    ref_tree = xgb_cv_auc if xgb_cv_auc is not None else logreg_ref
    if nn_val_auc < ref_tree - 1e-4:
        print()
        print(
            f"Neural network with {nn_params} parameters could not outperform the stronger "
            "baseline on this split at comparable sample size."
        )
        print(
            "This illustrates the bias–variance tradeoff: with limited training data, the "
            "inductive biases of tree-based and linear models (axis-aligned splits, convex "
            "loss, strong implicit regularization) often beat generic MLPs even with dropout "
            "and weight decay."
        )
    else:
        print()
        print(
            "Despite the small dataset, the neural network achieved competitive validation AUC, "
            "likely because the 12 pre-selected features shrink the input space enough for a "
            "tiny MLP's capacity to be appropriate."
        )


def run_twelve_feature_comparison(
    db_path: str = "data/ufc_matchmaker.db",
    verbose_sweep: bool = False,
) -> dict:
    """
    Train all ``SMALL_DATA_SWEEP`` configs (BCE + one pairwise), pick best val AUC,
    save ``checkpoints/nn_12feat.pt``, print comparison table + narrative.
    """
    from models.data_loader import get_canonical_splits

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    splits = get_canonical_splits(db_path)
    X_tr, y_tr = splits["X_train"], splits["y_train"].astype(np.float32)
    X_va, y_va = splits["X_val"], splits["y_val"].astype(np.float32)

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    best_auc = -1.0
    best_model: Optional[FightBonusNN] = None
    best_cfg: Optional[BinaryNNConfig] = None
    best_hist: Optional[dict] = None
    best_idx = -1

    print(f"\nSweep: {len(SMALL_DATA_SWEEP)} configs  (input_dim={X_tr.shape[1]}, n_train={len(y_tr)})")
    for i, spec in enumerate(SMALL_DATA_SWEEP):
        cfg = _config_from_sweep_dict(spec)
        cfg.input_dim = X_tr.shape[1]
        tag = f"[{i + 1}/{len(SMALL_DATA_SWEEP)}] {spec}"
        logger.info("Run %s", tag)
        model, hist = train_binary_nn(
            X_tr, y_tr, X_va, y_va, config=cfg, verbose=verbose_sweep,
        )
        auc = max(hist["val_auc"]) if hist["val_auc"] else 0.0
        ep = int(np.argmax(hist["val_auc"])) + 1
        print(f"  → val AUC = {auc:.4f}  (best epoch {ep})  loss={cfg.loss}  {cfg.hidden_dims}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_cfg = cfg
            best_hist = hist
            best_idx = i

    assert best_model is not None and best_cfg is not None

    n_params = count_trainable_parameters(best_model)
    out_path = CHECKPOINT_DIR / "nn_12feat.pt"
    torch.save(
        {
            "model_state": best_model.state_dict(),
            "config": {
                "input_dim": best_cfg.input_dim,
                "hidden_dims": best_cfg.hidden_dims,
                "dropout": best_cfg.dropout,
            },
            "best_val_auc": best_auc,
            "sweep_index": best_idx,
            "sweep_spec": SMALL_DATA_SWEEP[best_idx],
        },
        out_path,
    )
    print(f"\nSaved best NN → {out_path}")

    xgb_auc = _try_load_xgb_cv_auc()
    print_architecture_comparison(nn_val_auc=best_auc, nn_params=n_params, xgb_cv_auc=xgb_auc)

    return {
        "best_val_auc": best_auc,
        "best_config": best_cfg,
        "best_model": best_model,
        "best_sweep_index": best_idx,
        "n_parameters": n_params,
        "history": best_hist,
        "checkpoint": str(out_path),
    }


def run_pca10_nn_comparison(
    db_path: str = "data/ufc_matchmaker.db",
    verbose_sweep: bool = False,
    beat_threshold: float = NN_12_FEAT_VAL_REF,
) -> dict:
    """
    Sweep tiny NNs on PCA-transformed train/val (``pca_transformer.pkl``, typically 10-D).

    Saves ``checkpoints/nn_pca10_best.pt`` only if best val AUC **strictly exceeds**
    ``beat_threshold`` (default: 12-feat NN reference ~0.5991).
    """
    from models.data_loader import get_canonical_splits

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not PCA_TRANSFORMER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PCA_TRANSFORMER_PATH}. Run: python -m models.pca_pipeline"
        )

    pca = joblib.load(PCA_TRANSFORMER_PATH)
    n_pc = int(pca.components_.shape[0])
    if n_pc != 10:
        logger.warning(
            "Expected PCA with 10 components for this sweep; got n=%d. Proceeding anyway.",
            n_pc,
        )

    splits = get_canonical_splits(db_path, subset_features=False)
    X_tr = np.asarray(pca.transform(splits["X_train"]), dtype=np.float32)
    X_va = np.asarray(pca.transform(splits["X_val"]), dtype=np.float32)
    y_tr = splits["y_train"].astype(np.float32)
    y_va = splits["y_val"].astype(np.float32)

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    print(
        f"\nPCA-10 NN sweep: {len(PCA10_NN_SWEEP)} configs  "
        f"(input_dim={X_tr.shape[1]}, n_train={len(y_tr)})",
    )
    best_auc = -1.0
    best_model: Optional[FightBonusNN] = None
    best_cfg: Optional[BinaryNNConfig] = None
    best_hist: Optional[dict] = None
    best_idx = -1

    for i, spec in enumerate(PCA10_NN_SWEEP):
        cfg = _config_from_sweep_dict(spec)
        cfg.input_dim = X_tr.shape[1]
        tag = f"[{i + 1}/{len(PCA10_NN_SWEEP)}] {spec}"
        logger.info("Run %s", tag)
        model, hist = train_binary_nn(
            X_tr, y_tr, X_va, y_va, config=cfg, verbose=verbose_sweep,
        )
        auc = max(hist["val_auc"]) if hist["val_auc"] else 0.0
        ep = int(np.argmax(hist["val_auc"])) + 1
        print(f"  → val AUC = {auc:.4f}  (best epoch {ep})  loss={cfg.loss}  {cfg.hidden_dims}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_cfg = cfg
            best_hist = hist
            best_idx = i

    assert best_model is not None and best_cfg is not None and best_hist is not None
    n_params = count_trainable_parameters(best_model)

    print("\n" + "=" * 44)
    print(f"{'12-feat RFECV NN':<22} AUC = {NN_12_FEAT_VAL_REF:.4f}")
    print(f"{'PCA-10 NN':<22} AUC = {best_auc:.4f}")
    print("=" * 44)

    out_pca = CHECKPOINT_DIR / "nn_pca10_best.pt"
    saved: str | None = None
    if best_auc > beat_threshold:
        torch.save(
            {
                "model_state": best_model.state_dict(),
                "config": {
                    "input_dim": best_cfg.input_dim,
                    "hidden_dims": best_cfg.hidden_dims,
                    "dropout": best_cfg.dropout,
                },
                "best_val_auc": best_auc,
                "sweep_index": best_idx,
                "sweep_spec": PCA10_NN_SWEEP[best_idx],
                "pca_n_components": n_pc,
                "beat_threshold": beat_threshold,
            },
            out_pca,
        )
        saved = str(out_pca)
        print(f"\nSaved (PCA-10 NN beat {beat_threshold:.4f}) → {saved}")
    else:
        print(
            f"\nPCA-10 NN did not exceed {beat_threshold:.4f} — not saving {out_pca.name}",
        )

    return {
        "best_val_auc": best_auc,
        "n_pca_components": n_pc,
        "best_config": best_cfg,
        "best_model": best_model,
        "n_parameters": n_params,
        "history": best_hist,
        "checkpoint": saved,
        "reference_12feat_auc": NN_12_FEAT_VAL_REF,
    }


def sweep_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_combos: int = 30,
) -> tuple[BinaryNNConfig, FightBonusNN]:
    """
    Hyperparameter sweep. If ``X_train.shape[1] <= 16``, runs the small-data
    preset sweep (up to ``n_combos`` configs); otherwise samples random large
    architectures (legacy behavior).
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    if X_train.shape[1] <= 16:
        configs = SMALL_DATA_SWEEP[: min(n_combos, len(SMALL_DATA_SWEEP))]
        results: list[tuple[float, BinaryNNConfig, FightBonusNN]] = []
        for spec in configs:
            cfg = _config_from_sweep_dict(spec)
            cfg.input_dim = X_train.shape[1]
            model, history = train_binary_nn(
                X_train, y_train, X_val, y_val, config=cfg, verbose=False,
            )
            auc = max(history["val_auc"]) if history["val_auc"] else 0.0
            results.append((auc, cfg, model))
        results.sort(key=lambda r: r[0], reverse=True)
        print("\n" + "=" * 70)
        print("  SMALL-DATA SWEEP — results")
        print("=" * 70)
        for rank, (auc, cfg, _) in enumerate(results[:10], 1):
            print(f"  {rank}. AUC={auc:.4f}  dims={cfg.hidden_dims}  loss={cfg.loss}")
        print("=" * 70)
        return results[0][1], results[0][2]

    search_space = {
        "hidden_dims": [(256, 128, 64), (128, 64, 32), (64, 32)],
        "dropout": [0.2, 0.3, 0.5],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [32, 64],
    }
    keys = list(search_space.keys())
    all_combos = list(itertools.product(*(search_space[k] for k in keys)))
    random.shuffle(all_combos)
    combos = all_combos[:n_combos]

    results = []
    for i, values in enumerate(combos, 1):
        cfg = BinaryNNConfig(**dict(zip(keys, values)))
        cfg.input_dim = X_train.shape[1]
        model, history = train_binary_nn(
            X_train, y_train, X_val, y_val, config=cfg, verbose=False,
        )
        best_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
        results.append((best_auc, cfg, model))

    results.sort(key=lambda r: r[0], reverse=True)
    print("\n" + "=" * 90)
    print("  HYPERPARAMETER SWEEP — Top 10")
    print("=" * 90)
    for rank, (auc, cfg, _) in enumerate(results[:10], 1):
        print(
            f"  {rank}. {auc:.4f}  dims={cfg.hidden_dims}  drop={cfg.dropout} "
            f"lr={cfg.learning_rate} wd={cfg.weight_decay} bs={cfg.batch_size}"
        )
    print("=" * 90)
    return results[0][1], results[0][2]


def predict_proba(
    model: FightBonusNN,
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device

    if scaler is not None:
        X = scaler.transform(X)

    with torch.no_grad():
        tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tensor).squeeze(-1)
        return _sigmoid_np(logits.cpu().numpy())


def load_binary_nn(path: str) -> FightBonusNN:
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
        "Loaded FightBonusNN from %s  (val AUC=%.4f)",
        path, checkpoint.get("best_val_auc", float("nan")),
    )
    return model


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _generate_placeholder_data() -> dict:
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
    X[pos_idx, 72] += 0.8 + np.random.randn(n_pos) * 0.2
    X[pos_idx, 74] += 0.5
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
        "X_val": X[n_train : n_train + n_val], "y_val": y[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :], "y_test": y[n_train + n_val :],
        "scaler": scaler,
    }


def _load_data() -> dict:
    try:
        from models.data_loader import load_real_data
        data = load_real_data()
        logger.info("Using REAL data from DB")
        return data
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.warning("Real data unavailable (%s), using placeholder.", e)
        return _generate_placeholder_data()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        data = _load_data()
        dim = data["X_train"].shape[1]
        print(f"\n=== Full-feature mode (dim={dim}) — single train + sweep ===")
        cfg = BinaryNNConfig(
            input_dim=dim,
            hidden_dims=(128, 64, 32) if dim > 16 else (16, 8),
            epochs=150 if dim > 16 else MAX_EPOCHS_SMALL,
            patience=15 if dim > 16 else PATIENCE_SMALL,
        )
        model, history = train_binary_nn(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            config=cfg,
        )
        best_epoch = int(np.argmax(history["val_auc"]))
        print(f"Best epoch {best_epoch + 1}  val AUC={history['val_auc'][best_epoch]:.4f}")
        sweep_nn(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            n_combos=10,
        )
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ("pca10", "pca-10", "--pca10"):
        run_pca10_nn_comparison()
    else:
        run_twelve_feature_comparison()
