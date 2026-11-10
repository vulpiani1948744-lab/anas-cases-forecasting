"""
13 - Training v3: ensemble LGBM + CatBoost + baseline su dataset_v2_meteo (con meteo + provincia).
Output: v2/output/submission_v3_meteo.txt (e _safe / _aggressive)
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"  # dati grezzi della competizione (non versionati)

# %% carica
df = pd.read_csv(V2_OUT / "dataset_v2_meteo.csv", parse_dates=["mese_data"])
for c in ["tipo_strada", "regione", "Provincia"]:
    df[c] = df[c].astype("category")

print(f"Dataset v2 meteo: {df.shape}")

# %% split
mask_obs = df["num_richieste"].notna() & (df["split"] == "train")
train_full = df[mask_obs].copy()
is_train = train_full["mese_data"] < "2023-01-01"
is_val = (train_full["mese_data"] >= "2023-01-01") & (train_full["mese_data"] < "2024-01-01")
train_set = train_full[is_train]
val_set = train_full[is_val]
test_set = df[df["split"] == "test"].copy()

drop_cols = ["Strada", "mese_data", "num_richieste", "split", "test_order",
             "trend", "n_eventi_storici", "codice_strada"]
feat_cols = [c for c in df.columns if c not in drop_cols]
# CatBoost categorical features
cat_cols = ["tipo_strada", "regione", "Provincia"]
print(f"# feature: {len(feat_cols)} (cat: {cat_cols})")

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
print("\nTraining LightGBM...")
params_lgb = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=31, max_depth=6, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.5, verbose=-1, seed=42,
)
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
lgbm = lgb.train(
    params_lgb, dtrain, num_boost_round=3000,
    valid_sets=[dval], valid_names=["val"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
pred_val_lgb = np.clip(np.expm1(lgbm.predict(X_val, num_iteration=lgbm.best_iteration)), 0, None)
pred_test_lgb = np.clip(np.expm1(lgbm.predict(X_test, num_iteration=lgbm.best_iteration)), 0, None)
score_lgb = rmsle(y_val_raw, pred_val_lgb)
print(f"  RMSLE val LGBM: {score_lgb:.4f}")

# %% CatBoost
print("\nTraining CatBoost...")
X_tr_cb = X_train.copy()
X_va_cb = X_val.copy()
X_te_cb = X_test.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("__NA__")
    X_va_cb[c] = X_va_cb[c].astype(str).fillna("__NA__")
    X_te_cb[c] = X_te_cb[c].astype(str).fillna("__NA__")

cb = CatBoostRegressor(
    iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
    random_seed=42, verbose=0, early_stopping_rounds=100,
    cat_features=[feat_cols.index(c) for c in cat_cols],
    loss_function="RMSE",
)
cb.fit(X_tr_cb, y_train, eval_set=(X_va_cb, y_val))
pred_val_cb = np.clip(np.expm1(cb.predict(X_va_cb)), 0, None)
pred_test_cb = np.clip(np.expm1(cb.predict(X_te_cb)), 0, None)
score_cb = rmsle(y_val_raw, pred_val_cb)
print(f"  RMSLE val CB  : {score_cb:.4f}")

# %% baseline no-leakage
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
score_base = rmsle(y_val_raw, pred_val_base)
print(f"  RMSLE val base: {score_base:.4f}")

# %% blend safe
def safe_rmsle(weights):
    w = np.clip(weights, 0, None)
    if w.sum() == 0: return 1e9
    w = w / w.sum()
    log_pred = (w[0] * np.log1p(pred_val_lgb) + w[1] * np.log1p(pred_val_cb) + w[2] * np.log1p(pred_val_base))
    return rmsle(y_val_raw, np.expm1(log_pred))

best = (None, 1e9)
for wl in np.arange(0, 1.01, 0.05):
    for wc in np.arange(0, 1.01-wl, 0.05):
        wb = 1 - wl - wc
        if wb < -1e-9: continue
        s = safe_rmsle([wl, wc, max(wb, 0)])
        if s < best[1]:
            best = ([wl, wc, max(wb, 0)], s)
w = np.array(best[0]); score_blend = best[1]
print(f"\nBlend safe: LGBM={w[0]:.2f}, CB={w[1]:.2f}, Base={w[2]:.2f} → RMSLE={score_blend:.4f}")

# %% submission
log_pred_test = (w[0] * np.log1p(pred_test_lgb) + w[1] * np.log1p(pred_test_cb) + w[2] * np.log1p(pred_test_base))
pred_test_final = np.clip(np.expm1(log_pred_test), 0, None)
test_out = test_set.assign(pred=pred_test_final).sort_values("test_order")
preds = test_out["pred"].values

# scrivi nel formato dell'esempio: no header, CRLF, una riga per valore
sub_path = V2_OUT / "submission_v3_meteo.txt"
content = "\r\n".join(f"{v}" for v in preds) + "\r\n"
sub_path.write_text(content)
print(f"→ salvato {sub_path}")
print(f"Righe: {len(preds)} (atteso 15048)")

# feature importance CatBoost (di solito più informativa per ensemble)
fi = pd.DataFrame({"f": feat_cols, "imp": cb.feature_importances_}).sort_values("imp", ascending=False)
print("\nTop 20 feature importance (CatBoost):")
print(fi.head(20).to_string(index=False))

print("\n" + "=" * 60)
print(f"PROGRESSO RMSLE")
print("=" * 60)
print(f"Baseline pura       : 0.7606")
print(f"v1 ensemble safe    : 0.7368")
print(f"v3 meteo ensemble   : {score_blend:.4f}")
