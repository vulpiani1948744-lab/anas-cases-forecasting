"""
14 - Ablation studies su dataset_v2_meteo:
- A: escludi righe 2016 dal training
- B: IterativeImputer (MICE-like) per i lag NaN

Confronto con baseline corrente: v3 meteo = 0.7359 val RMSLE
Tiene la submission solo se MIGLIORA la baseline.
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

# attiva IterativeImputer (sperimentale)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"

# %% carica
df = pd.read_csv(V2_OUT / "dataset_v2_meteo.csv", parse_dates=["mese_data"])
for c in ["tipo_strada", "regione", "Provincia"]:
    df[c] = df[c].astype("category")

drop_cols = ["Strada", "mese_data", "num_richieste", "split", "test_order",
             "trend", "n_eventi_storici", "codice_strada"]
feat_cols = [c for c in df.columns if c not in drop_cols]
cat_cols = ["tipo_strada", "regione", "Provincia"]

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def build_profile(cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    pb = h.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_base"})
    gm = h.groupby("mese")["num_richieste"].mean().reset_index().rename(columns={"num_richieste": "pred_global"})
    return pb, gm, h["num_richieste"].mean()
def apply_profile(d, pb, gm, gmean):
    b = d.merge(pb, on=["Strada", "mese"], how="left").merge(gm, on="mese", how="left")
    return b["pred_base"].fillna(b["pred_global"]).fillna(gmean).values

def train_and_eval(train_set, val_set, test_set, label=""):
    """Allena LGBM+CB+Baseline e ritorna RMSLE val + predizione test."""
    X_train = train_set[feat_cols]; y_train = np.log1p(train_set["num_richieste"].values)
    X_val   = val_set[feat_cols];   y_val   = np.log1p(val_set["num_richieste"].values)
    X_test  = test_set[feat_cols]

    # LGBM
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)
    params_lgb = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=31, max_depth=6, min_child_samples=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.5, verbose=-1, seed=42,
    )
    lgbm = lgb.train(params_lgb, dtrain, 3000, valid_sets=[dval], valid_names=["val"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred_val_lgb = np.clip(np.expm1(lgbm.predict(X_val, num_iteration=lgbm.best_iteration)), 0, None)
    pred_test_lgb = np.clip(np.expm1(lgbm.predict(X_test, num_iteration=lgbm.best_iteration)), 0, None)

    # CatBoost
    X_tr_cb = X_train.copy(); X_va_cb = X_val.copy(); X_te_cb = X_test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("__NA__")
        X_va_cb[c] = X_va_cb[c].astype(str).fillna("__NA__")
        X_te_cb[c] = X_te_cb[c].astype(str).fillna("__NA__")
    cb = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                           random_seed=42, verbose=0, early_stopping_rounds=100,
                           cat_features=[feat_cols.index(c) for c in cat_cols],
                           loss_function="RMSE")
    cb.fit(X_tr_cb, y_train, eval_set=(X_va_cb, y_val))
    pred_val_cb = np.clip(np.expm1(cb.predict(X_va_cb)), 0, None)
    pred_test_cb = np.clip(np.expm1(cb.predict(X_te_cb)), 0, None)

    # baseline
    pb_v, gm_v, gmean_v = build_profile("2023-01-01")
    pred_val_base = apply_profile(val_set, pb_v, gm_v, gmean_v)
    pb_t, gm_t, gmean_t = build_profile("2024-01-01")
    pred_test_base = apply_profile(test_set, pb_t, gm_t, gmean_t)
    y_val_raw = val_set["num_richieste"].values

    # blend safe
    def safe_rmsle(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 1e9
        w = w / w.sum()
        log_pred = w[0] * np.log1p(pred_val_lgb) + w[1] * np.log1p(pred_val_cb) + w[2] * np.log1p(pred_val_base)
        return rmsle(y_val_raw, np.expm1(log_pred))
    best = (None, 1e9)
    for wl in np.arange(0, 1.01, 0.05):
        for wc in np.arange(0, 1.01-wl, 0.05):
            wb = 1 - wl - wc
            if wb < -1e-9: continue
            s = safe_rmsle([wl, wc, max(wb, 0)])
            if s < best[1]: best = ([wl, wc, max(wb, 0)], s)
    w = np.array(best[0])
    log_pred_test = w[0]*np.log1p(pred_test_lgb) + w[1]*np.log1p(pred_test_cb) + w[2]*np.log1p(pred_test_base)
    pred_test = np.clip(np.expm1(log_pred_test), 0, None)
    print(f"[{label}] LGBM={best[0][0]:.2f} CB={best[0][1]:.2f} Base={best[0][2]:.2f} → RMSLE val = {best[1]:.4f}")
    return best[1], pred_test

# %% split helper
mask_obs = df["num_richieste"].notna() & (df["split"] == "train")
train_full = df[mask_obs].copy()
test_set = df[df["split"] == "test"].copy()

is_val = (train_full["mese_data"] >= "2023-01-01") & (train_full["mese_data"] < "2024-01-01")
val_set = train_full[is_val]

BASELINE_TO_BEAT = 0.7359

# =========================================================
# Ablation A: escludi 2016
# =========================================================
print("\n" + "=" * 60)
print("ABLATION A: escludi righe 2016 dal training")
print("=" * 60)
train_A = train_full[(train_full["mese_data"] >= "2017-01-01") & (train_full["mese_data"] < "2023-01-01")]
print(f"Train A: {len(train_A)} righe (vs base {(train_full['mese_data'] < '2023-01-01').sum()})")
score_A, pred_A = train_and_eval(train_A, val_set, test_set, "A_no2016")

# =========================================================
# Ablation B: IterativeImputer per i lag
# =========================================================
print("\n" + "=" * 60)
print("ABLATION B: IterativeImputer per i lag")
print("=" * 60)

# imputiamo solo le colonne lag/rolling che hanno NaN (no target, no categoriche)
lag_cols = ["lag_12", "lag_24", "roll_mean_12", "roll_max_12", "roll_std_12", "same_3m_lastyear"]

# feature ausiliarie per l'imputer (numeric only, no NaN)
aux_cols = ["anno", "mese", "sin_mese", "cos_mese", "is_estate", "is_inverno",
            "n_festivi", "n_weekend", "n_vacanze_scolastiche",
            "media_storica_mensile", "std_storica_mensile", "mesi_attivi", "pct_mesi_attivi",
            "lunghezza_km", "quota_pesanti", "consistenza_gg_media", "leggeri_media",
            "covid_intensity"]
# usa solo quelle effettivamente presenti
aux_cols = [c for c in aux_cols if c in df.columns]

X_for_impute = df[aux_cols + lag_cols].copy()
# riempie NaN delle ausiliarie con la media globale prima dell'imputer (servono input completi)
for c in aux_cols:
    X_for_impute[c] = X_for_impute[c].fillna(X_for_impute[c].median())

print(f"Righe da imputare (con almeno un lag NaN): {X_for_impute[lag_cols].isna().any(axis=1).sum()}")

imp = IterativeImputer(
    estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=42),
    max_iter=5, random_state=42, verbose=0,
)
X_imp = imp.fit_transform(X_for_impute)
X_imp = pd.DataFrame(X_imp, columns=aux_cols + lag_cols, index=df.index)

# i lag possono diventare negativi → clip a 0
for c in lag_cols:
    X_imp[c] = X_imp[c].clip(lower=0)

# copia i lag imputati nel df
df_B = df.copy()
for c in lag_cols:
    df_B[c] = X_imp[c].values

# rebuild split
mask_obs_B = df_B["num_richieste"].notna() & (df_B["split"] == "train")
train_full_B = df_B[mask_obs_B]
train_B = train_full_B[train_full_B["mese_data"] < "2023-01-01"]
val_B = train_full_B[(train_full_B["mese_data"] >= "2023-01-01") & (train_full_B["mese_data"] < "2024-01-01")]
test_B = df_B[df_B["split"] == "test"]

for c in ["tipo_strada", "regione", "Provincia"]:
    train_B[c] = train_B[c].astype("category")
    val_B[c] = val_B[c].astype("category")
    test_B[c] = test_B[c].astype("category")

score_B, pred_B = train_and_eval(train_B, val_B, test_B, "B_mice_lag")

# =========================================================
# Confronto finale e selezione
# =========================================================
print("\n" + "=" * 60)
print("CONFRONTO FINALE")
print("=" * 60)
print(f"Baseline v3 meteo (da battere): {BASELINE_TO_BEAT:.4f}")
print(f"Ablation A (no 2016)         : {score_A:.4f} ({'✓ MIGLIORA' if score_A < BASELINE_TO_BEAT else '✗ peggiora'})")
print(f"Ablation B (MICE lag)        : {score_B:.4f} ({'✓ MIGLIORA' if score_B < BASELINE_TO_BEAT else '✗ peggiora'})")

# salva submission del MIGLIORE solo se batte la baseline
best_score = min(BASELINE_TO_BEAT, score_A, score_B)
if best_score < BASELINE_TO_BEAT:
    if score_A < score_B:
        winner_pred, winner_name = pred_A, "submission_v4_no2016"
    else:
        winner_pred, winner_name = pred_B, "submission_v4_miceLag"
    test_out = test_set.assign(pred=winner_pred).sort_values("test_order")
    preds = test_out["pred"].values
    content = "\r\n".join(f"{v}" for v in preds) + "\r\n"
    out_path = V2_OUT / f"{winner_name}.txt"
    out_path.write_text(content)
    print(f"\n→ MIGLIORAMENTO! Salvato {out_path}")
    print(f"  RMSLE val: {best_score:.4f}")
else:
    print("\n→ Nessuna ablation migliora. Submission corrente resta submission_v3_meteo.txt")
