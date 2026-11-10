"""
22 - Training v7: ensemble + totale annuale feature + rescaling sui totali.

Pipeline:
1. Multi-seed LGBM + CatBoost + baseline (come v6)
2. Aggiunge feature tot_anno_corrente, tot_2024/25 ecc.
3. Dopo la predizione, **rescala** le predizioni mensili 2024 e 2025 per ogni strada
   in modo che sum(pred_mese) = tot_anno_strada.

Genera due submission:
- submission_v7_norescale.txt : solo feature, niente rescaling
- submission_v7_rescaled.txt  : con rescaling (questo dovrebbe essere il migliore)
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
df = pd.read_csv(V2_OUT / "dataset_v7_totali.csv", parse_dates=["mese_data"])
for c in ["tipo_strada", "regione", "Provincia"]:
    df[c] = df[c].astype("category")
print(f"Dataset v7: {df.shape}")

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

# %% multi-seed LGBM
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
m_last_cb = None
for s in SEEDS:
    m = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          random_seed=s, verbose=0, early_stopping_rounds=100,
                          cat_features=cat_idx, loss_function="RMSE")
    m.fit(X_tr, y_train, eval_set=(X_va, y_val))
    pred_val_cb += np.clip(np.expm1(m.predict(X_va)), 0, None)
    pred_test_cb += np.clip(np.expm1(m.predict(X_te)), 0, None)
    m_last_cb = m
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

# %% blend
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
print(f"\nBlend safe v7: LGBM={w[0]:.2f}, CB={w[1]:.2f}, Base={w[2]:.2f} → RMSLE val (NO rescale) = {score:.4f}")

# %% predizione test senza rescale
log_pred_test = w[0]*np.log1p(pred_test_lgb) + w[1]*np.log1p(pred_test_cb) + w[2]*np.log1p(pred_test_base)
pred_test_norescale = np.clip(np.expm1(log_pred_test), 0, None)

# %% RESCALING: per ogni strada, riallinea sum(2024 mesi) = tot_2024 e sum(2025 mesi) = tot_2025
print("\n" + "=" * 60)
print("RESCALING per totali noti")
print("=" * 60)

test_set_rescale = test_set.assign(pred_raw=pred_test_norescale).copy()
# scala per (Strada, anno)
test_set_rescale["anno"] = test_set_rescale["mese_data"].dt.year
test_set_rescale["target_tot"] = np.where(
    test_set_rescale["anno"] == 2024, test_set_rescale["tot_2024_strada"],
    test_set_rescale["tot_2025_strada"]
)

# somma per (Strada, anno) della predizione raw
sums_raw = test_set_rescale.groupby(["Strada", "anno"])["pred_raw"].sum().reset_index().rename(columns={"pred_raw": "sum_raw"})
test_set_rescale = test_set_rescale.merge(sums_raw, on=["Strada", "anno"])

# fattore di scaling: target_tot / sum_raw (se sum_raw > 0)
# se target_tot è NaN (strade non in road_metadata), lascia raw
# se sum_raw è 0 (raro), spalma uniforme su 12 mesi
def rescale_row(row):
    if pd.isna(row["target_tot"]):
        return row["pred_raw"]
    if row["sum_raw"] > 0:
        return row["pred_raw"] * (row["target_tot"] / row["sum_raw"])
    return row["target_tot"] / 12  # spalma uniforme

test_set_rescale["pred_rescaled"] = test_set_rescale.apply(rescale_row, axis=1)
test_set_rescale["pred_rescaled"] = np.clip(test_set_rescale["pred_rescaled"], 0, None)

print(f"Righe con rescaling applicato: {test_set_rescale['target_tot'].notna().sum()}")
print(f"Stats pred raw       : mean={test_set_rescale['pred_raw'].mean():.2f}, max={test_set_rescale['pred_raw'].max():.2f}")
print(f"Stats pred rescaled  : mean={test_set_rescale['pred_rescaled'].mean():.2f}, max={test_set_rescale['pred_rescaled'].max():.2f}")

# soft rescaling: blend tra raw e rescaled (log-space)
# alpha=0 → pred_raw; alpha=1 → pred_rescaled; alpha=0.5 → meta strada
for alpha in [0.5, 0.75]:
    log_blend = (1 - alpha) * np.log1p(test_set_rescale["pred_raw"]) + alpha * np.log1p(test_set_rescale["pred_rescaled"])
    test_set_rescale[f"pred_soft_{int(alpha*100)}"] = np.clip(np.expm1(log_blend), 0, None)

# %% submissions
for col, name in [("pred_raw", "submission_v7_norescale.txt"),
                  ("pred_rescaled", "submission_v7_rescaled.txt"),
                  ("pred_soft_50", "submission_v7_soft50.txt"),
                  ("pred_soft_75", "submission_v7_soft75.txt")]:
    test_out = test_set_rescale.sort_values("test_order")
    preds = test_out[col].values
    sub_path = V2_OUT / name
    content = "\r\n".join(f"{v}" for v in preds) + "\r\n"
    sub_path.write_text(content)
    print(f"→ {name}: mean={np.mean(preds):.2f}, max={np.max(preds):.2f}")

print("\n" + "=" * 60)
print("CONFRONTO FINALE")
print("=" * 60)
print(f"v3_safe (leaderboard reference): {BASELINE_SAFE:.4f}")
print(f"v7 no-rescale (val 2023)       : {score:.4f}  Δ={BASELINE_SAFE-score:+.4f}")
print(f"  (rescale: il guadagno reale si vede solo sul test 2024-25, non sulla val 2023)")
print(f"\nFeature importance Top 10 (CatBoost):")
fi = pd.DataFrame({"f": feat_cols, "imp": m_last_cb.feature_importances_}).sort_values("imp", ascending=False)
print(fi.head(15).to_string(index=False))
