"""
Full pipeline diagnostic: DB → features → scaler → model.

Run from repo root:
  OMP_NUM_THREADS=1 python -m models.debug_full_pipeline
"""
from __future__ import annotations

import inspect
import json
import sqlite3
import traceback
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import torch

from models.nn_binary import FightBonusNN, predict_proba as nn_predict_proba_fn

print("=" * 70)
print("  UFC MATCHMAKER — FULL PIPELINE DIAGNOSTIC")
print("=" * 70)

bugs_found: list[str] = []
infos: list[str] = []

# Shared state for later checks
vec_bare: np.ndarray | None = None
vec_subset: np.ndarray | None = None
fighters_sample: list = []
db_conn: sqlite3.Connection | None = None
mm = None
vec_mm: np.ndarray | None = None

# ═══════════════════════════════════════════
# CHECK 1: Can the model produce varied predictions on training data?
# ═══════════════════════════════════════════
print("\n[CHECK 1] Model predictions on training/test data")
print("-" * 50)

from models.data_loader import get_canonical_splits

splits = get_canonical_splits()
X_train, y_train = splits["X_train"], splits["y_train"]
X_val, y_val = splits["X_val"], splits["y_val"]
X_test, y_test = splits["X_test"], splits["y_test"]
scaler_from_splits = splits.get("scaler")

print(f"  X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"  X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"  X_train range: [{float(np.min(X_train)):.4f}, {float(np.max(X_train)):.4f}]")
print(f"  X_test range: [{float(np.min(X_test)):.4f}, {float(np.max(X_test)):.4f}]")
print(f"  X_train mean: {float(np.mean(X_train)):.4f}, std: {float(np.std(X_train)):.4f}")
print(
    f"  y_train distribution: {float(y_train.sum()):.0f} positive / {len(y_train)} "
    f"total = {float(y_train.mean()):.3f}",
)

data_is_scaled = bool(abs(float(np.mean(X_train))) < 0.5 and 0.5 < float(np.std(X_train)) < 2.0)
if data_is_scaled:
    print("  → Data appears to be SCALED (mean≈0, std≈1)")
else:
    print("  → Data appears to be RAW (not scaled)")

# NN on random train rows (reference: should vary)
_nn_path = Path("models/checkpoints/nn_12feat.pt")
if _nn_path.is_file():
    ck = torch.load(_nn_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    nn_ref = FightBonusNN(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
    )
    nn_ref.load_state_dict(ck["model_state"])
    nn_ref.eval()
    idx = np.random.RandomState(42).choice(len(X_train), size=min(200, len(X_train)), replace=False)
    train_probs = nn_predict_proba_fn(nn_ref, X_train[idx])
    print(
        f"  NN on 200 random X_train rows: min={train_probs.min():.4f}, max={train_probs.max():.4f}, "
        f"mean={train_probs.mean():.4f}, n_unique≈{len(np.unique(np.round(train_probs, 4)))}",
    )
    if (train_probs >= 0.999).mean() > 0.5:
        bugs_found.append(
            "NN outputs ~1.0 for most scaled training rows — model/scaler/weights issue, not just matchmaker",
        )
    if len(np.unique(np.round(train_probs, 3))) < 3:
        bugs_found.append("NN produces almost no diversity on training subsample")

# ═══════════════════════════════════════════
# CHECK 2: XGBoost pipeline structure
# ═══════════════════════════════════════════
print("\n[CHECK 2] XGBoost pipeline structure")
print("-" * 50)

xgb_path = Path("models/checkpoints/xgb_tuned_12feat.pkl")
pipeline_has_scaler = False
xgb_pipeline = None

if not xgb_path.is_file():
    print(f"  ERROR: {xgb_path} not found!")
    bugs_found.append("XGBoost checkpoint file missing")
else:
    xgb_pipeline = joblib.load(xgb_path)
    print(f"  Type: {type(xgb_pipeline)}")

    if hasattr(xgb_pipeline, "steps"):
        print(
            f"  Pipeline steps: {[(name, type(step).__name__) for name, step in xgb_pipeline.steps]}",
        )
        for name, step in xgb_pipeline.steps:
            ln = name.lower()
            if "scaler" in ln or "standard" in type(step).__name__.lower():
                pipeline_has_scaler = True
                print(f"  → Pipeline HAS scaler-like step '{name}' ({type(step).__name__})")
    elif hasattr(xgb_pipeline, "predict_proba"):
        print(f"  → Bare model (no pipeline), type: {type(xgb_pipeline).__name__}")
    else:
        print(f"  → Unknown type: {type(xgb_pipeline)}")
        bugs_found.append(f"XGBoost checkpoint is unexpected type: {type(xgb_pipeline)}")

    try:
        probs_wrong = xgb_pipeline.predict_proba(X_test)[:, 1]
        print(
            f"  Predictions on scaled X_test (WRONG if pipeline expects raw): "
            f"min={probs_wrong.min():.4f}, max={probs_wrong.max():.4f}",
        )
        if (probs_wrong >= 0.999).all() or (probs_wrong <= 0.001).all():
            bugs_found.append(
                "XGBoost on scaled X_test yields all-saturated probabilities — likely wrong input scale",
            )
    except Exception as e:
        print(f"  predict_proba on scaled X_test failed: {e}")

    try:
        X_test_raw = scaler_from_splits.inverse_transform(X_test)
        probs = xgb_pipeline.predict_proba(X_test_raw)[:, 1]
        print(
            f"  Predictions on UNSCALED (inverse) X_test: min={probs.min():.4f}, "
            f"max={probs.max():.4f}, mean={probs.mean():.4f}",
        )
        print(f"  All ≥0.999? {bool((probs >= 0.999).all())}")
        print(f"  All ≤0.001? {bool((probs <= 0.001).all())}")
        print(f"  Unique values (rounded 4dp): {len(np.unique(np.round(probs, 4)))}")
        if (probs >= 0.999).all() or (probs <= 0.001).all():
            bugs_found.append("XGBoost predicts all same tail value on correctly scaled raw test")
        elif len(np.unique(np.round(probs, 2))) < 5:
            bugs_found.append(f"XGBoost produces very few unique predictions ({len(np.unique(np.round(probs, 2)))})")
    except Exception as e:
        print(f"  ERROR predicting on raw X_test: {e}")
        bugs_found.append(f"XGBoost can't predict on inverse-scaled X_test: {e}")

# ═══════════════════════════════════════════
# CHECK 3: Scaler alignment
# ═══════════════════════════════════════════
print("\n[CHECK 3] Scaler alignment")
print("-" * 50)

scaler_path = Path("models/checkpoints/scaler_12feat.pkl")
scaler_standalone = None
if scaler_path.is_file():
    scaler_standalone = joblib.load(scaler_path)
    print(f"  Standalone scaler n_features: {scaler_standalone.n_features_in_}")
    print(f"  Standalone scaler mean (first 5): {scaler_standalone.mean_[:5]}")
    print(f"  Standalone scaler scale (first 5): {scaler_standalone.scale_[:5]}")

    if xgb_pipeline is not None and hasattr(xgb_pipeline, "named_steps"):
        pipe_scaler = xgb_pipeline.named_steps.get("scaler")
        if pipe_scaler is not None:
            print(f"  Pipeline scaler n_features: {pipe_scaler.n_features_in_}")
            print(f"  Pipeline scaler mean (first 5): {pipe_scaler.mean_[:5]}")
            same_mean = np.allclose(scaler_standalone.mean_, pipe_scaler.mean_)
            same_scale = np.allclose(scaler_standalone.scale_, pipe_scaler.scale_)
            if same_mean and same_scale:
                print("  → Standalone scaler matches pipeline scaler (mean & scale)")
            else:
                print("  → Standalone scaler DIFFERS from pipeline scaler")
                bugs_found.append(
                    "scaler_12feat.pkl parameters differ from XGB pipeline's internal scaler",
                )

    # Match training scaler vs saved
    if scaler_from_splits is not None:
        if np.allclose(scaler_from_splits.mean_, scaler_standalone.mean_) and np.allclose(
            scaler_from_splits.scale_, scaler_standalone.scale_
        ):
            print("  → get_canonical_splits scaler matches scaler_12feat.pkl")
        else:
            print("  → get_canonical_splits scaler DIFFERS from scaler_12feat.pkl")
            bugs_found.append(
                "Data loader scaler ≠ scaler_12feat.pkl — matchmaker may use wrong z-scores vs training",
            )

    print("\n  ⚠️  DOUBLE SCALING CHECK:")
    print(f"  Pipeline has scaler: {pipeline_has_scaler}")
    print(f"  get_canonical_splits returns scaled matrices: {data_is_scaled}")
    if pipeline_has_scaler and data_is_scaled:
        infos.append(
            "INFO: Default MatchmakerV2 uses XGB pipeline on raw 12-D rows; "
            "backend='nn' uses scaler_12feat + FightBonusNN (separate from XGB's internal scaler).",
        )
else:
    print(f"  Standalone scaler not found at {scaler_path}")
    bugs_found.append("scaler_12feat.pkl missing")

# ═══════════════════════════════════════════
# CHECK 4: Feature names and ordering
# ═══════════════════════════════════════════
print("\n[CHECK 4] Feature names and ordering")
print("-" * 50)

from models.feature_engineering import ALL_FEATURE_NAMES
from models.pipeline_config import SELECTED_FEATURES

_sf = SELECTED_FEATURES or []
print(f"  SELECTED_FEATURES ({len(_sf)}): {_sf}")
print(f"  ALL_FEATURE_NAMES count: {len(ALL_FEATURE_NAMES)}")

for feat in _sf:
    if feat in ALL_FEATURE_NAMES:
        idx = ALL_FEATURE_NAMES.index(feat)
        print(f"    {feat} → index {idx} in ALL_FEATURE_NAMES ✓")
    else:
        print(f"    {feat} → NOT FOUND in ALL_FEATURE_NAMES ✗")
        bugs_found.append(f"Selected feature '{feat}' not found in ALL_FEATURE_NAMES")

# ═══════════════════════════════════════════
# CHECK 5: build_full_matchup_vector output
# ═══════════════════════════════════════════
print("\n[CHECK 5] build_full_matchup_vector output")
print("-" * 50)

from models.feature_engineering import build_full_matchup_vector, subset_full_feature_vector

db_conn = sqlite3.connect(Path("data/ufc_matchmaker.db"))
db_conn.row_factory = sqlite3.Row
fighters_db = db_conn.execute(
    """
    SELECT f.* FROM fighters f
    WHERE EXISTS (SELECT 1 FROM fight_stats fs WHERE fs.fighter_id = f.id)
    LIMIT 2
    """,
).fetchall()

if len(fighters_db) < 2:
    print("  ERROR: Can't find 2 fighters with fight_stats")
    bugs_found.append("Database has fewer than 2 fighters with fight_stats rows")
else:
    fighters_sample = [dict(fighters_db[0]), dict(fighters_db[1])]
    fa_raw = fighters_sample[0]
    fb_raw = fighters_sample[1]
    print(f"  Fighter A: {fa_raw.get('name', 'unknown')} (id={fa_raw.get('id')})")
    print(f"  Fighter B: {fb_raw.get('name', 'unknown')} (id={fb_raw.get('id')})")
    print(f"  Fighter A keys (sample): {sorted(fa_raw.keys())[:18]}...")

    try:
        vec_bare = build_full_matchup_vector(fa_raw, fb_raw)
        vec_bare = np.asarray(vec_bare, dtype=np.float64)
        print(f"  Bare vector (no attachments) length: {len(vec_bare)}")
        print(f"  Bare vector values (first 10): {vec_bare[:10]}")
        nan_c = int(np.isnan(vec_bare).sum())
        print(f"  Bare vector NaN count: {nan_c}")
        if nan_c:
            bugs_found.append("build_full_matchup_vector produces NaNs without rolling/context attachments")

        if len(vec_bare) != len(ALL_FEATURE_NAMES):
            bugs_found.append(
                f"build_full_matchup_vector returns length {len(vec_bare)}, expected {len(ALL_FEATURE_NAMES)}",
            )
    except Exception as e:
        print(f"  ERROR building bare vector: {e}")
        bugs_found.append(f"build_full_matchup_vector fails without attachments: {e}")

# ═══════════════════════════════════════════
# CHECK 6: Rolling features
# ═══════════════════════════════════════════
print("\n[CHECK 6] Rolling features")
print("-" * 50)

from models.rolling_features import compute_rolling_features, get_fighter_fight_history

if len(fighters_sample) >= 1 and db_conn is not None:
    fa_id = int(fighters_sample[0]["id"])
    try:
        history = get_fighter_fight_history(fa_id, db_conn)
        print(f"  Fighter A fight history rows: {len(history)}")
        if len(history) >= 2:
            asof = str(history[0].get("event_date") or "")
            rolling = compute_rolling_features(
                history,
                fa_id,
                asof,
                n_recent=5,
                career_fallback=fighters_sample[0],
            )
            rolling = np.asarray(rolling)
            print(f"  Rolling features length: {rolling.shape}")
            print(f"  Rolling values (first 8): {rolling[:8]}")
            print(f"  Rolling NaN count: {np.isnan(rolling).sum()}")
            print(f"  Rolling all zeros? {bool((rolling == 0).all())}")
        else:
            print("  Not enough fight history for full rolling (need >=2 prior fights in DB)")
    except Exception as e:
        print(f"  ERROR computing rolling features: {e}")
        bugs_found.append(f"Rolling features computation fails: {e}")

# ═══════════════════════════════════════════
# CHECK 7: subset_full_feature_vector
# ═══════════════════════════════════════════
print("\n[CHECK 7] subset_full_feature_vector")
print("-" * 50)

if vec_bare is not None:
    try:
        vec_subset = subset_full_feature_vector(vec_bare, list(_sf), list(ALL_FEATURE_NAMES))
        vec_subset = np.asarray(vec_subset, dtype=np.float64)
        print(f"  Subset vector length: {len(vec_subset)}")
        print(f"  Subset vector: {vec_subset}")
        print(f"  Expected length: {len(_sf)}")
        if len(vec_subset) != len(_sf):
            bugs_found.append(
                f"subset returns length {len(vec_subset)}, expected {len(_sf)}",
            )
        for i, feat_name in enumerate(_sf):
            if feat_name in ALL_FEATURE_NAMES:
                full_idx = ALL_FEATURE_NAMES.index(feat_name)
                expected_val = float(vec_bare[full_idx])
                actual_val = float(vec_subset[i])
                ok = abs(expected_val - actual_val) < 1e-5
                tag = "✓" if ok else f"✗ (expected {expected_val}, got {actual_val})"
                print(
                    f"    {feat_name}: full[{full_idx}]={expected_val:.4f}, subset[{i}]={actual_val:.4f} {tag}",
                )
                if not ok:
                    bugs_found.append(
                        f"Feature {feat_name} wrong in subset: expected {expected_val}, got {actual_val}",
                    )
    except Exception as e:
        print(f"  ERROR in subset: {e}")
        bugs_found.append(f"subset_full_feature_vector fails: {e}")

# ═══════════════════════════════════════════
# CHECK 8: MatchmakerV2 initialization
# ═══════════════════════════════════════════
print("\n[CHECK 8] MatchmakerV2 initialization")
print("-" * 50)

try:
    from models.matchmaker_v2 import MatchmakerV2

    init_src = inspect.getsource(MatchmakerV2.__init__)
    print(f"  __init__ excerpt:\n{init_src[:700]}...\n")

    if "FightBonusNN" in init_src:
        infos.append("MatchmakerV2 supports backend='nn' (FightBonusNN) and backend='xgb' (default pipeline)")
    if "xgb_tuned" in init_src or "xgb_pipeline" in init_src:
        infos.append("Default checkpoint is xgb_tuned_12feat.pkl (production)")

    mm = MatchmakerV2(db_path="data/ufc_matchmaker.db")
    print(f"  MatchmakerV2 initialized (backend={mm.backend})")
    if mm.backend == "xgb":
        print(f"  XGB pipeline: {type(mm.xgb_pipeline).__name__}")
    else:
        print(f"  NN model: {type(mm.model).__name__}")
    print(f"  Has scaler (NN only): {mm.scaler is not None}")
    print(f"  Fighters loaded: {len(mm.fighters)}")
    if hasattr(mm, "rolling_cache"):
        non_none = sum(1 for v in mm.rolling_cache.values() if v is not None)
        print(f"  Rolling cache non-None: {non_none} / {len(mm.rolling_cache)}")

except Exception as e:
    print(f"  ERROR initializing MatchmakerV2: {e}")
    traceback.print_exc()
    bugs_found.append(f"MatchmakerV2 fails to initialize: {e}")

# ═══════════════════════════════════════════
# CHECK 9: End-to-end matchmaker prediction trace
# ═══════════════════════════════════════════
print("\n[CHECK 9] End-to-end prediction trace")
print("-" * 50)

raw_12: np.ndarray | None = None
vec_scaled_row: np.ndarray | None = None

if mm is not None:
    try:
        active = mm._get_active_fighters("Lightweight")[:6]
        if len(active) < 2:
            active = mm._get_active_fighters("Welterweight")[:6]
        if len(active) < 2:
            fids = list(mm.fighters.keys())[:6]
            active = [mm.fighters[i] for i in fids]

        # DBfighter dicts use 'id'
        fa_id = int(active[0]["id"])
        fb_id = int(active[1]["id"])
        fa_name = mm.fighters[fa_id].get("name", fa_id)
        fb_name = mm.fighters[fb_id].get("name", fb_id)
        print(f"  Testing: {fa_name} vs {fb_name}")

        raw_12 = mm._raw_subvector(fa_id, fb_id)
        print(f"  _raw_subvector (12): {raw_12}")
        print(f"  raw range: [{raw_12.min():.4f}, {raw_12.max():.4f}]")

        vec_scaled_row = scaler_from_splits.transform(raw_12.reshape(1, -1)).ravel()
        print(f"  canonical-scaler(raw) first 6: {vec_scaled_row[:6]}")

        result = mm.score_matchup(fa_id, fb_id)
        print(f"  score_matchup probability: {result['probability']}")

        if result["probability"] >= 0.99 and mm.backend == "nn":
            bugs_found.append(
                f"NN backend: score_matchup ≥0.99 for ({fa_name} vs {fb_name}) — known sigmoid saturation; "
                "use backend='xgb' for calibrated probabilities",
            )
        if result["probability"] >= 0.995 and mm.backend == "xgb":
            bugs_found.append(
                f"XGB backend still ≥0.995 for ({fa_name} vs {fb_name}) — investigate features",
            )

        print("\n  Testing several matchups (same division pool):")
        all_probs: list[float] = []
        for fa, fb in list(combinations(active[:5], 2))[:5]:
            try:
                a_id = int(fa["id"])
                b_id = int(fb["id"])
                r = mm.score_matchup(a_id, b_id)
                print(f"    {r['fighter_a']} vs {r['fighter_b']}: {r['probability']:.4f}")
                all_probs.append(float(r["probability"]))
            except Exception as ex:
                print(f"    Error: {ex}")

        if len(all_probs) >= 2 and len(set(round(p, 4) for p in all_probs)) == 1:
            bugs_found.append(
                f"Multiple distinct matchups yield identical probability {all_probs[0]:.4f}",
            )

    except Exception as e:
        print(f"  ERROR in prediction trace: {e}")
        traceback.print_exc()
        bugs_found.append(f"Prediction trace failed: {e}")

# ═══════════════════════════════════════════
# CHECK 10: Canonical-scaled matchmaker vector vs training
# ═══════════════════════════════════════════
print("\n[CHECK 10] Canonical scaler(raw) vs training distribution")
print("-" * 50)

if mm is not None and raw_12 is not None and vec_scaled_row is not None:
    Xtr = np.asarray(X_train, dtype=np.float64)
    for i, feat in enumerate(_sf):
        if i >= vec_scaled_row.shape[0] or i >= Xtr.shape[1]:
            break
        train_mean = float(Xtr[:, i].mean())
        train_std = float(Xtr[:, i].std())
        mm_val = float(vec_scaled_row[i])
        z_score = (mm_val - train_mean) / train_std if train_std > 1e-12 else 0.0
        flag = ""
        if abs(z_score) > 5:
            flag = " ⚠️ TAIL (compare to train max for this column)"
        elif abs(z_score) > 3:
            flag = " ⚠️ OUTLIER"
        print(
            f"    {feat:<38} mm={mm_val:>8.4f}  train_mean={train_mean:>8.4f}  z={z_score:>6.2f}{flag}",
        )

# ═══════════════════════════════════════════
# CHECK 11: Hypothetical — BatchNorm running stats vs batch=1
# ═══════════════════════════════════════════
print("\n[CHECK 11] NN module state (BatchNorm running stats)")
print("-" * 50)
if _nn_path.is_file():
    ck = torch.load(_nn_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = FightBonusNN(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
    )
    m.load_state_dict(ck["model_state"])
    m.eval()
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            rm = float(mod.running_mean.detach().abs().mean())
            rv = float(mod.running_var.detach().mean())
            eps = float(mod.eps)
            print(f"  {name}: running_mean|mean|={rm:.4f}, running_var_mean={rv:.6f}, eps={eps}")
            if rv < 1e-5:
                bugs_found.append(
                    f"BatchNorm1d '{name}' has near-zero running_var ({rv}) — inference can explode",
                )

# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("  INFO / NOTES")
print("=" * 70)
for i, note in enumerate(infos, 1):
    print(f"  {i}. {note}")
if not infos:
    print("  (none)")

print("\n" + "=" * 70)
print("  BUG SUMMARY")
print("=" * 70)
# Dedup preserve order
seen: set[str] = set()
deduped: list[str] = []
for b in bugs_found:
    if b not in seen:
        seen.add(b)
        deduped.append(b)

if deduped:
    for i, bug in enumerate(deduped, 1):
        print(f"  {i}. {bug}")
    print(f"\n  Total bugs found: {len(deduped)}")
else:
    print("  No bugs flagged by automated checks.")

print("=" * 70)

summary_path = Path("outputs/debug_full_pipeline_summary.json")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps({"infos": infos, "bugs_found": deduped}, indent=2),
    encoding="utf-8",
)
print(f"\nWrote {summary_path}")

if db_conn is not None:
    db_conn.close()

print("\nRun: OMP_NUM_THREADS=1 python -m models.debug_full_pipeline")
