"""
20 - Training v6: dataset con feature ACI incidenti (n_incidenti_mese, lag1, roll3, roll12)
+ multi-seed ensemble LightGBM + CatBoost + baseline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"

BASELINE_SAFE = 0.7368
THRESHOLD = 0.005

# %% carica
df = pd.read_csv(V2_OUT / "dataset_v6_aci.csv", parse_dates=["mese_data"])
for c in ["tipo_strada", "regione", "Provincia"]:
    df[c] = df[c].astype("category")
print(f"Dataset v6: {df.shape}")

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
cat_cols = ["tipo_strada", "regione", "Provincia"]
print(f"# feature: {len(feat_cols)}")

X_train = train_set[feat_cols]; X_val = val_set[feat_cols]; X_test = test_set[feat_cols]
y_train_raw = train_set["num_richieste"].values
y_val_raw = val_set["num_richieste"].values
y_train = np.log1p(y_train_raw); y_val = np.log1p(y_val_raw)

def rmsle(yt, yp):
    yp = np.clip(yp, 0, None)
    return np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt))**2))

# %% multi-seed LGBM (5 seed)
SEEDS = [42, 7, 123, 2024, 999]
print(f"\nMulti-seed LGBM...")
pred_val_lgb = np.zeros(len(val_set)); pred_test_lgb = np.zeros(len(test_set))
for s in SEEDS:
    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=31, max_depth=6, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  lambda_l1=0.1, lambda_l2=0.5, verbose=-1, seed=s)
    dtr = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], valid_names=["val"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred_val_lgb += np.clip(np.expm1(m.predict(X_val, num_iteration=m.best_iteration)), 0, None)
    pred_test_lgb += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None)
pred_val_lgb /= len(SEEDS); pred_test_lgb /= len(SEEDS)
print(f"  RMSLE val LGBM: {rmsle(y_val_raw, pred_val_lgb):.4f}")

# %% multi-seed CatBoost
print(f"\nMulti-seed CatBoost...")
X_tr = X_train.copy(); X_va = X_val.copy(); X_te = X_test.copy()
for c in cat_cols:
    X_tr[c] = X_tr[c].astype(str).fillna("__NA__")
    X_va[c] = X_va[c].astype(str).fillna("__NA__")
    X_te[c] = X_te[c].astype(str).fillna("__NA__")
cat_idx = [feat_cols.index(c) for c in cat_cols]
pred_val_cb = np.zeros(len(val_set)); pred_test_cb = np.zeros(len(test_set))
for s in SEEDS:
    m = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          random_seed=s, verbose=0, early_stopping_rounds=100,
                          cat_features=cat_idx, loss_function="RMSE")
    m.fit(X_tr, y_train, eval_set=(X_va, y_val))
    pred_val_cb += np.clip(np.expm1(m.predict(X_va)), 0, None)
    pred_test_cb += np.clip(np.expm1(m.predict(X_te)), 0, None)
pred_val_cb /= len(SEEDS); pred_test_cb /= len(SEEDS)
print(f"  RMSLE val CB  : {rmsle(y_val_raw, pred_val_cb):.4f}")

# %% baseline no-leakage
def build_profile(cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    pb = h.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_base"})
    gm = h.groupby("mese")["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_global"})
    return pb, gm, h["num_richieste"].mean()
def apply_profile(d, pb, gm, gmean):
    b = d.merge(pb, on=["Strada", "mese"], how="left").merge(gm, on="mese", how="left")
    return b["pred_base"].fillna(b["pred_global"]).fillna(gmean).values

pb_v, gm_v, gmean_v = build_profile("2023-01-01")
pred_val_base = apply_profile(val_set, pb_v, gm_v, gmean_v)
pb_t, gm_t, gmean_t = build_profile("2024-01-01")
pred_test_base = apply_profile(test_set, pb_t, gm_t, gmean_t)
print(f"  RMSLE val base: {rmsle(y_val_raw, pred_val_base):.4f}")

# %% blend safe
def safe_rmsle(w):
    w = np.clip(w, 0, None)
    if w.sum() == 0: return 1e9
    w = w / w.sum()
    log_pred = w[0]*np.log1p(pred_val_lgb) + w[1]*np.log1p(pred_val_cb) + w[2]*np.log1p(pred_val_base)
    return rmsle(y_val_raw, np.expm1(log_pred))
best = (None, 1e9)
for wl in np.arange(0, 1.01, 0.05):
    for wc in np.arange(0, 1.01-wl, 0.05):
        wb = 1 - wl - wc
        if wb < -1e-9: continue
        s = safe_rmsle([wl, wc, max(wb, 0)])
        if s < best[1]: best = ([wl, wc, max(wb, 0)], s)
w = np.array(best[0]); score = best[1]
print(f"\nBlend safe v6: LGBM={w[0]:.2f}, CB={w[1]:.2f}, Base={w[2]:.2f} → RMSLE val = {score:.4f}")

# %% submission
log_pred_test = w[0]*np.log1p(pred_test_lgb) + w[1]*np.log1p(pred_test_cb) + w[2]*np.log1p(pred_test_base)
pred_test_final = np.clip(np.expm1(log_pred_test), 0, None)
test_out = test_set.assign(pred=pred_test_final).sort_values("test_order")
preds = test_out["pred"].values

sub_path = V2_OUT / "submission_v6_aci.txt"
content = "\r\n".join(f"{v}" for v in preds) + "\r\n"
sub_path.write_text(content)
print(f"→ salvato {sub_path}")

# feature importance ACI specifiche
import lightgbm as lgb_check
m_last = m
fi_cb = pd.DataFrame({"f": feat_cols, "imp": m_last.feature_importances_}).sort_values("imp", ascending=False)
print("\nTop 25 feature importance (CatBoost ultimo seed):")
print(fi_cb.head(25).to_string(index=False))

# evidenzia rank delle feature ACI
aci_feats = ["n_incidenti_mese", "n_incidenti_lag1", "n_incidenti_roll3", "n_incidenti_roll12"]
print("\nRank feature ACI:")
for f in aci_feats:
    rank = (fi_cb["f"] == f).idxmax() if (fi_cb["f"] == f).any() else None
    if rank is not None:
        pos = fi_cb.reset_index().index[fi_cb.reset_index()["f"] == f].tolist()[0]
        print(f"  {f}: rank #{pos+1} (imp={fi_cb[fi_cb['f']==f]['imp'].iloc[0]:.4f})")

print("\n" + "=" * 60)
print("CONFRONTO FINALE")
print("=" * 60)
print(f"v3_safe (leaderboard reference): {BASELINE_SAFE:.4f}")
print(f"v6 con ACI incidenti           : {score:.4f}")
delta = BASELINE_SAFE - score
print(f"Delta: {delta:+.4f} ({'MIGLIORA' if delta > 0 else 'PEGGIORA'})")
print(f"Soglia 'reale' (Δ ≥ {THRESHOLD}): {'✓ SUPERATA' if delta >= THRESHOLD else '✗ sotto soglia'}")
