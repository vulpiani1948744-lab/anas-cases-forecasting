"""
07 - Training v2 sul dataset arricchito (con COVID dettagliato + incidenti).

Stessa pipeline di 05, ma su dataset_arricchito.csv. Genera submission_v2.txt.
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica
df = pd.read_csv(OUT_DIR / "dataset_arricchito.csv", parse_dates=["mese_data"])
df["tipo_strada"] = df["tipo_strada"].astype("category")
print(f"dataset_arricchito: {df.shape}")

# %% split
mask_obs = df["num_richieste"].notna() & (df["split"] == "train")
train_full = df[mask_obs].copy()
is_train = train_full["mese_data"] < "2023-01-01"
is_val = (train_full["mese_data"] >= "2023-01-01") & (train_full["mese_data"] < "2024-01-01")

train_set = train_full[is_train]
val_set = train_full[is_val]
test_set = df[df["split"] == "test"].copy()

# rimuovi colonne con distribution shift evidente
drop_cols = ["Strada", "mese_data", "num_richieste", "split", "test_order",
             "trend", "n_eventi_storici"]
feat_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]
# il dataset salvato in CSV può avere tipo_strada come stringa
if "tipo_strada" not in feat_cols:
    feat_cols.append("tipo_strada")

print(f"# feature: {len(feat_cols)}")

X_train = train_set[feat_cols]
X_val = val_set[feat_cols]
X_test = test_set[feat_cols]
y_train = np.log1p(train_set["num_richieste"].values)
y_val = np.log1p(val_set["num_richieste"].values)

# %% LightGBM regolarizzato
params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.5,
    verbose=-1,
    seed=42,
)

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=["tipo_strada"])
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=["tipo_strada"], reference=dtrain)

print("Training LightGBM v2...")
model = lgb.train(
    params, dtrain, num_boost_round=3000,
    valid_sets=[dtrain, dval], valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

# %% metrica RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

pred_val = np.clip(np.expm1(model.predict(X_val, num_iteration=model.best_iteration)), 0, None)
score_lgbm = rmsle(val_set["num_richieste"].values, pred_val)
print(f"\nRMSLE val (LGBM v2 puro): {score_lgbm:.4f}")

# %% baseline no leakage
def build_profile(cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    pb = h.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_base"})
    gm = h.groupby("mese")["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_global"})
    return pb, gm, h["num_richieste"].mean()

def apply_profile(df_in, pb, gm, gmean):
    b = df_in.merge(pb, on=["Strada", "mese"], how="left").merge(gm, on="mese", how="left")
    return b["pred_base"].fillna(b["pred_global"]).fillna(gmean).values

pb_val, gm_val, gmean_val = build_profile("2023-01-01")
pred_val_base = apply_profile(val_set, pb_val, gm_val, gmean_val)
score_base = rmsle(val_set["num_richieste"].values, pred_val_base)
print(f"RMSLE val (baseline)    : {score_base:.4f}")

pb_test, gm_test, gmean_test = build_profile("2024-01-01")
pred_test_base = apply_profile(test_set, pb_test, gm_test, gmean_test)

# %% blend search
pred_test_lgbm = np.clip(np.expm1(model.predict(X_test, num_iteration=model.best_iteration)), 0, None)

print("\nBlend search:")
best_w, best_s = 1.0, score_lgbm
for w in np.arange(0.0, 1.01, 0.05):
    blend_log = w * np.log1p(pred_val) + (1 - w) * np.log1p(pred_val_base)
    s = rmsle(val_set["num_richieste"].values, np.expm1(blend_log))
    if s < best_s:
        best_s, best_w = s, w
print(f"Miglior peso w_lgbm = {best_w:.2f} → RMSLE val = {best_s:.4f}")

# %% submission
final_log = best_w * np.log1p(pred_test_lgbm) + (1 - best_w) * np.log1p(pred_test_base)
pred_test_final = np.clip(np.expm1(final_log), 0, None)

test_out = test_set.assign(pred=pred_test_final).sort_values("test_order")
submission = test_out[["pred"]].rename(columns={"pred": "num_richieste"})
sub_path = OUT_DIR / "submission_v2.txt"
submission.to_csv(sub_path, index=False)

# top feature importance
fi = pd.DataFrame({"f": feat_cols, "imp": model.feature_importance("gain")}).sort_values("imp", ascending=False)
print("\nTop 20 feature importance:")
print(fi.head(20).to_string(index=False))

print(f"\n→ salvato {sub_path}")
print(f"Stats: min={submission['num_richieste'].min():.2f}, max={submission['num_richieste'].max():.2f}, mean={submission['num_richieste'].mean():.2f}")

print("\n" + "=" * 60)
print(f"PROGRESSO RMSLE")
print("=" * 60)
print(f"Baseline pura       : 0.7606")
print(f"LGBM v1 + blend 0.4 : 0.7392")
print(f"LGBM v2 puro        : {score_lgbm:.4f}")
print(f"LGBM v2 + blend     : {best_s:.4f}   ({(best_s - 0.7392)/0.7392*100:+.1f}% vs v1)")
