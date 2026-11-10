"""
05 - Dataset finale + training LightGBM + submission.

Pipeline:
1. Carica dataset_base + profilo_strada (con fallback per strade nuove).
2. Aggiunge lag stabili sul target: lag_12, lag_24, rolling_mean_12, rolling_max_12,
   same_3m_lastyear. Niente lag short-term per evitare predizione iterativa sul test.
3. Aggiunge un indicatore COVID grezzo (dummy 2020-03 → 2022-04) come prima approssimazione.
4. Train LightGBM su log1p(target), early stopping su validation 2023.
5. Genera submission_lgbm.txt.

Output:
- output/dataset_finale.csv
- output/submission_lgbm.txt
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica
df = pd.read_csv(OUT_DIR / "dataset_base.csv", parse_dates=["mese_data"])
profilo = pd.read_csv(OUT_DIR / "profilo_strada.csv")

print(f"dataset_base: {df.shape}")
print(f"profilo     : {profilo.shape}")

# %% merge profilo
df = df.merge(profilo, on="Strada", how="left")

# fallback per strade nuove (mai viste in train): media del profilo
profilo_means = profilo.select_dtypes("number").mean()
for col in profilo_means.index:
    df[col] = df[col].fillna(profilo_means[col])

# flag strada nuova (può essere informativa)
df["is_strada_nuova"] = df["n_eventi_storici"].isna().astype(int)

# %% lag stabili sul target (calcolati per Strada, ordinati per mese)
print("\nCalcolo lag stabili sul target...")
df = df.sort_values(["Strada", "mese_data"]).reset_index(drop=True)

def add_lag_features(g):
    s = g["num_richieste"]
    g["lag_12"] = s.shift(12)
    g["lag_24"] = s.shift(24)
    # rolling sui 12 mesi precedenti (chiuso): media e max degli ultimi 12 mesi
    g["roll_mean_12"] = s.shift(1).rolling(12, min_periods=3).mean()
    g["roll_max_12"] = s.shift(1).rolling(12, min_periods=3).max()
    g["roll_std_12"] = s.shift(1).rolling(12, min_periods=3).std()
    # media degli stessi 3 mesi anno scorso (es. per 2024-02: media di 2023-01,02,03)
    g["same_3m_lastyear"] = (s.shift(11) + s.shift(12) + s.shift(13)) / 3
    return g

df = df.groupby("Strada", group_keys=False).apply(add_lag_features, include_groups=True)

# %% dummy COVID grezza (fallback in attesa dei dati zone colorate)
df["is_covid_strict"] = (
    (df["mese_data"] >= "2020-03-01") & (df["mese_data"] <= "2020-06-01")
).astype(int)
df["is_covid_period"] = (
    (df["mese_data"] >= "2020-03-01") & (df["mese_data"] <= "2022-04-01")
).astype(int)
df["mesi_da_covid_end"] = np.where(
    df["mese_data"] >= "2022-05-01",
    ((df["mese_data"] - pd.Timestamp("2022-04-01")).dt.days / 30.4).clip(lower=0),
    0,
)

# %% encode categoriche
df["tipo_strada"] = df["tipo_strada"].astype("category")

# %% split train/val/test
mask_obs = df["num_richieste"].notna() & (df["split"] == "train")
train_full = df[mask_obs].copy()
is_train = train_full["mese_data"] < "2023-01-01"
is_val = (train_full["mese_data"] >= "2023-01-01") & (train_full["mese_data"] < "2024-01-01")

train_set = train_full[is_train]
val_set = train_full[is_val]
test_set = df[df["split"] == "test"].copy()

print(f"\nTrain : {len(train_set)} righe")
print(f"Val   : {len(val_set)} righe")
print(f"Test  : {len(test_set)} righe")

# %% feature columns
drop_cols = ["Strada", "mese_data", "num_richieste", "split", "test_order"]
feat_cols = [c for c in df.columns if c not in drop_cols]
print(f"\n# features: {len(feat_cols)}")

# %% target log1p
y_train = np.log1p(train_set["num_richieste"].values)
y_val = np.log1p(val_set["num_richieste"].values)
X_train = train_set[feat_cols]
X_val = val_set[feat_cols]
X_test = test_set[feat_cols]

# %% rimuovo feature che cambiano molto tra train e test (distribution shift)
shift_cols = ["trend", "n_eventi_storici"]
for c in shift_cols:
    if c in feat_cols:
        feat_cols.remove(c)
X_train = train_set[feat_cols]
X_val = val_set[feat_cols]
X_test = test_set[feat_cols]
print(f"# features dopo rimozione shift: {len(feat_cols)}")

# %% LightGBM con regolarizzazione più alta
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

print("\nTraining LightGBM...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=80),
        lgb.log_evaluation(period=100),
    ],
)

# %% metrica RMSLE sulla validation
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

pred_val_log = model.predict(X_val, num_iteration=model.best_iteration)
pred_val = np.expm1(pred_val_log)
pred_val = np.clip(pred_val, 0, None)
score_val = rmsle(val_set["num_richieste"].values, pred_val)
print(f"\nRMSLE validation 2023 (LightGBM): {score_val:.4f}")
print(f"(Baseline naive era: 0.7606)")

# %% feature importance
fi = pd.DataFrame({
    "feature": feat_cols,
    "imp": model.feature_importance(importance_type="gain"),
}).sort_values("imp", ascending=False)
print("\nTop 15 feature importance (gain):")
print(fi.head(15).to_string(index=False))

# %% predizione test (LightGBM puro)
pred_test_log = model.predict(X_test, num_iteration=model.best_iteration)
pred_test_lgbm = np.clip(np.expm1(pred_test_log), 0, None)

# %% baseline (media strada × mese del calendario)
# IMPORTANTE: usa cutoff diversi per evitare leakage
#  - per valutare sul val 2023: usa solo dati pre-2023
#  - per predire sul test 2024-2025: usa tutti i dati pre-2024

def build_profile(cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    pb = h.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_base"})
    gm = h.groupby("mese")["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_global"})
    return pb, gm, h["num_richieste"].mean()

def apply_profile(df_in, pb, gm, gmean):
    b = df_in.merge(pb, on=["Strada", "mese"], how="left").merge(gm, on="mese", how="left")
    return b["pred_base"].fillna(b["pred_global"]).fillna(gmean).values

# baseline per validation (cutoff = 2023-01)
pb_val, gm_val, gmean_val = build_profile("2023-01-01")
pred_val_base = apply_profile(val_set, pb_val, gm_val, gmean_val)
score_val_base = rmsle(val_set["num_richieste"].values, pred_val_base)
print(f"\nRMSLE baseline (no leakage) su val 2023: {score_val_base:.4f}")

# baseline per test (cutoff = 2024-01, può usare anche il 2023 legittimamente)
pb_test, gm_test, gmean_test = build_profile("2024-01-01")
pred_test_base = apply_profile(test_set, pb_test, gm_test, gmean_test)

# %% scelta peso blend (geometric mean su log-scale)
print("\nBlend LightGBM × Baseline (cerco peso ottimo su val):")
best_w, best_s = 1.0, score_val
for w in np.arange(0.0, 1.01, 0.1):
    blend_log = w * np.log1p(pred_val) + (1 - w) * np.log1p(pred_val_base)
    blend = np.expm1(blend_log)
    s = rmsle(val_set["num_richieste"].values, blend)
    print(f"  w_lgbm={w:.1f}: RMSLE={s:.4f}")
    if s < best_s:
        best_s, best_w = s, w

print(f"\nMiglior blend: w_lgbm={best_w:.1f}, RMSLE val = {best_s:.4f}")

# applica blend su test
final_log = best_w * np.log1p(pred_test_lgbm) + (1 - best_w) * np.log1p(pred_test_base)
pred_test_final = np.clip(np.expm1(final_log), 0, None)

test_set = test_set.assign(pred=pred_test_final).sort_values("test_order")
submission = test_set[["pred"]].rename(columns={"pred": "num_richieste"})

sub_path = OUT_DIR / "submission_lgbm.txt"
submission.to_csv(sub_path, index=False)
print(f"\n→ salvato {sub_path}")
print(f"Righe: {len(submission)} (devono essere 15048)")
print(f"Stats: min={submission['num_richieste'].min():.2f}, max={submission['num_richieste'].max():.2f}, mean={submission['num_richieste'].mean():.2f}")

# salva anche dataset finale per ispezione
df.to_csv(OUT_DIR / "dataset_finale.csv", index=False)
print(f"→ salvato {OUT_DIR / 'dataset_finale.csv'}")
