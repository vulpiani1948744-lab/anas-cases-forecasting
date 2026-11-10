"""
08 - Ensemble finale: LightGBM + CatBoost + baseline.

Pesi blend cercati su validation 2023.
Output: output/submission_v3.txt (candidata finale).
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica
df = pd.read_csv(OUT_DIR / "dataset_arricchito.csv", parse_dates=["mese_data"])
df["tipo_strada"] = df["tipo_strada"].astype("category")

mask_obs = df["num_richieste"].notna() & (df["split"] == "train")
train_full = df[mask_obs].copy()
is_train = train_full["mese_data"] < "2023-01-01"
is_val = (train_full["mese_data"] >= "2023-01-01") & (train_full["mese_data"] < "2024-01-01")
train_set = train_full[is_train]
val_set = train_full[is_val]
test_set = df[df["split"] == "test"].copy()

drop_cols = ["Strada", "mese_data", "num_richieste", "split", "test_order",
             "trend", "n_eventi_storici"]
feat_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]
if "tipo_strada" not in feat_cols:
    feat_cols.append("tipo_strada")

X_train = train_set[feat_cols]
X_val = val_set[feat_cols]
X_test = test_set[feat_cols]
y_train_raw = train_set["num_richieste"].values
y_val_raw = val_set["num_richieste"].values
y_train = np.log1p(y_train_raw)
y_val = np.log1p(y_val_raw)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# %% LightGBM
print("Training LightGBM...")
params_lgb = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=31, max_depth=6, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.5, verbose=-1, seed=42,
)
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=["tipo_strada"])
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=["tipo_strada"], reference=dtrain)
lgbm = lgb.train(
    params_lgb, dtrain, num_boost_round=3000,
    valid_sets=[dval], valid_names=["val"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
pred_val_lgb = np.clip(np.expm1(lgbm.predict(X_val, num_iteration=lgbm.best_iteration)), 0, None)
pred_test_lgb = np.clip(np.expm1(lgbm.predict(X_test, num_iteration=lgbm.best_iteration)), 0, None)
print(f"  RMSLE val LGBM: {rmsle(y_val_raw, pred_val_lgb):.4f}")

# %% CatBoost
print("\nTraining CatBoost...")
# CatBoost gestisce categoriche da indice
cat_idx = [feat_cols.index("tipo_strada")]
# riempi NaN nelle categoriche con stringa
X_train_cb = X_train.copy()
X_val_cb = X_val.copy()
X_test_cb = X_test.copy()
X_train_cb["tipo_strada"] = X_train_cb["tipo_strada"].astype(str)
X_val_cb["tipo_strada"] = X_val_cb["tipo_strada"].astype(str)
X_test_cb["tipo_strada"] = X_test_cb["tipo_strada"].astype(str)

cb = CatBoostRegressor(
    iterations=3000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3, random_seed=42, verbose=0,
    early_stopping_rounds=100, cat_features=cat_idx,
    loss_function="RMSE",
)
cb.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val))
pred_val_cb = np.clip(np.expm1(cb.predict(X_val_cb)), 0, None)
pred_test_cb = np.clip(np.expm1(cb.predict(X_test_cb)), 0, None)
print(f"  RMSLE val CB  : {rmsle(y_val_raw, pred_val_cb):.4f}")

# %% baseline no leakage
def build_profile(cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    pb = h.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_base"})
    gm = h.groupby("mese")["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_global"})
    return pb, gm, h["num_richieste"].mean()
def apply_profile(d, pb, gm, gmean):
    b = d.merge(pb, on=["Strada", "mese"], how="left").merge(gm, on="mese", how="left")
    return b["pred_base"].fillna(b["pred_global"]).fillna(gmean).values

pb_val, gm_val, gmean_val = build_profile("2023-01-01")
pred_val_base = apply_profile(val_set, pb_val, gm_val, gmean_val)
pb_test, gm_test, gmean_test = build_profile("2024-01-01")
pred_test_base = apply_profile(test_set, pb_test, gm_test, gmean_test)
print(f"  RMSLE val base: {rmsle(y_val_raw, pred_val_base):.4f}")

# %% ottimizza pesi blend (geometric on log-scale) via grid + minimize
print("\nOttimizzazione pesi blend (log-space)...")
def blended_rmsle(weights):
    w_lgb, w_cb, w_base = weights
    s = w_lgb + w_cb + w_base
    if s <= 0:
        return 1e9
    w_lgb, w_cb, w_base = w_lgb/s, w_cb/s, w_base/s
    log_pred = (w_lgb * np.log1p(pred_val_lgb) +
                w_cb * np.log1p(pred_val_cb) +
                w_base * np.log1p(pred_val_base))
    return rmsle(y_val_raw, np.expm1(log_pred))

# grid search prima per partenza, poi minimize
best = (None, 1e9)
for wl in np.arange(0, 1.01, 0.1):
    for wc in np.arange(0, 1.01-wl + 1e-9, 0.1):
        wb = 1 - wl - wc
        if wb < -1e-9: continue
        s = blended_rmsle([wl, wc, max(wb, 0)])
        if s < best[1]:
            best = ([wl, wc, max(wb, 0)], s)

# raffinamento
res = minimize(blended_rmsle, best[0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5})
w_lgb, w_cb, w_base = res.x / res.x.sum()
score_final = blended_rmsle(res.x)

print(f"Pesi ottimi: LGBM={w_lgb:.2f}, CB={w_cb:.2f}, Base={w_base:.2f}")
print(f"RMSLE val finale: {score_final:.4f}")

# %% blend conservativo: vincolo non-negativo + normalizzato (più safe contro overfit della val)
print("\nBlend conservativo (pesi ≥ 0, sum=1):")
def safe_rmsle(weights):
    w = np.clip(weights, 0, None)
    if w.sum() == 0:
        return 1e9
    w = w / w.sum()
    log_pred = (w[0] * np.log1p(pred_val_lgb) +
                w[1] * np.log1p(pred_val_cb) +
                w[2] * np.log1p(pred_val_base))
    return rmsle(y_val_raw, np.expm1(log_pred))

best_safe = (None, 1e9)
for wl in np.arange(0, 1.01, 0.05):
    for wc in np.arange(0, 1.01-wl, 0.05):
        wb = 1 - wl - wc
        if wb < -1e-9: continue
        s = safe_rmsle([wl, wc, max(wb, 0)])
        if s < best_safe[1]:
            best_safe = ([wl, wc, max(wb, 0)], s)
w_safe = np.array(best_safe[0])
score_safe = best_safe[1]
print(f"  Pesi safe: LGBM={w_safe[0]:.2f}, CB={w_safe[1]:.2f}, Base={w_safe[2]:.2f} → RMSLE={score_safe:.4f}")

# %% submission v3
log_pred_test = (w_lgb * np.log1p(pred_test_lgb) +
                 w_cb * np.log1p(pred_test_cb) +
                 w_base * np.log1p(pred_test_base))
pred_test_final = np.clip(np.expm1(log_pred_test), 0, None)

test_out = test_set.assign(pred=pred_test_final).sort_values("test_order")
submission = test_out[["pred"]].rename(columns={"pred": "num_richieste"})
sub_path = OUT_DIR / "submission_v3_aggressive.txt"
submission.to_csv(sub_path, index=False)

# submission "safe" con pesi non-negativi
log_pred_safe = (w_safe[0] * np.log1p(pred_test_lgb) +
                 w_safe[1] * np.log1p(pred_test_cb) +
                 w_safe[2] * np.log1p(pred_test_base))
pred_test_safe = np.clip(np.expm1(log_pred_safe), 0, None)
test_out2 = test_set.assign(pred=pred_test_safe).sort_values("test_order")
sub_safe = test_out2[["pred"]].rename(columns={"pred": "num_richieste"})
sub_safe.to_csv(OUT_DIR / "submission_v3_safe.txt", index=False)
print(f"\n→ salvato submission_v3_safe.txt (pesi non-negativi)")

print("\n" + "=" * 60)
print("PROGRESSO RMSLE")
print("=" * 60)
print(f"Baseline pura       : 0.7606")
print(f"LGBM v1 + blend     : 0.7392")
print(f"LGBM v2 + blend     : 0.7389")
print(f"Ensemble v3 (LGBM+CB+Base): {score_final:.4f}")
print(f"\n→ salvato {sub_path}")
print(f"Stats: min={submission['num_richieste'].min():.2f}, max={submission['num_richieste'].max():.2f}, mean={submission['num_richieste'].mean():.2f}, NaN={submission['num_richieste'].isna().sum()}")
