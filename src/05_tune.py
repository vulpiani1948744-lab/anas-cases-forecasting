"""
04 - Tuning iperparametri con Optuna (OPZIONALE).

Cerca i parametri ottimi di LightGBM e CatBoost minimizzando l'RMSLE sulla
validazione (anno VAL_YEAR). Salva:
  - output/best_lgbm_params.json
  - output/best_cb_params.json

Se non lo esegui, 05_train.py usa i DEFAULT in config.py (gia' tunati).
Uso: python 04_tune.py            (40 trial LGBM + 20 CatBoost)
     python 04_tune.py 60 30      (trial personalizzati)
"""
import sys
import json
import numpy as np
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
import config as C
from _common import load_dataset, feature_cols, splits, cat_for_catboost

n_lgbm = int(sys.argv[1]) if len(sys.argv) > 1 else 40
n_cb = int(sys.argv[2]) if len(sys.argv) > 2 else 20

df = load_dataset()
feats = feature_cols(df)
cats = [c for c in C.CAT_COLS if c in feats]
tr, va, _ = splits(df)
Xtr, Xva = tr[feats], va[feats]
# target a conteggi grezzi (loss Tweedie -> nessun log transform)
ytr = tr["num_richieste"].values
yva_raw = va["num_richieste"].values
yva = yva_raw
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"# feature: {len(feats)} | train: {len(tr):,} | val: {len(va):,}")

# --------------------------------------------------------------- LightGBM (Tweedie)
def obj_lgbm(t):
    p = dict(objective="tweedie", metric="tweedie",
             tweedie_variance_power=C.TWEEDIE_VP, verbose=-1, seed=42,
             learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
             num_leaves=t.suggest_int("num_leaves", 15, 127),
             max_depth=t.suggest_int("max_depth", 3, 12),
             min_child_samples=t.suggest_int("min_child_samples", 5, 200),
             feature_fraction=t.suggest_float("feature_fraction", 0.5, 1.0),
             bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 1.0),
             bagging_freq=t.suggest_int("bagging_freq", 1, 10),
             lambda_l1=t.suggest_float("lambda_l1", 1e-3, 10, log=True),
             lambda_l2=t.suggest_float("lambda_l2", 1e-3, 10, log=True),
             min_split_gain=t.suggest_float("min_split_gain", 0, 1))
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr)
    mdl = lgb.train(p, dtr, 3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pred = np.clip(mdl.predict(Xva, num_iteration=mdl.best_iteration), 0, None)
    return C.rmsle(yva_raw, pred)

print(f"\n>>> Optuna LightGBM ({n_lgbm} trial)...")
s1 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
s1.optimize(obj_lgbm, n_trials=n_lgbm)
json.dump(s1.best_params, open(C.OUT / "best_lgbm_params.json", "w"), indent=2)
print(f"  best RMSLE: {s1.best_value:.4f} -> best_lgbm_params.json")

# --------------------------------------------------------------- CatBoost
Xtr_c = cat_for_catboost(Xtr, cats)
Xva_c = cat_for_catboost(Xva, cats)
cat_idx = [feats.index(c) for c in cats]

def obj_cb(t):
    p = dict(iterations=3000, random_seed=42, verbose=0, early_stopping_rounds=100,
             loss_function=f"Tweedie:variance_power={C.TWEEDIE_VP}", cat_features=cat_idx,
             learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
             depth=t.suggest_int("depth", 4, 10),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
             random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
             bagging_temperature=t.suggest_float("bagging_temperature", 0, 1),
             border_count=t.suggest_int("border_count", 32, 255),
             min_data_in_leaf=t.suggest_int("min_data_in_leaf", 5, 200))
    mdl = CatBoostRegressor(**p)
    mdl.fit(Xtr_c, ytr, eval_set=(Xva_c, yva))
    return C.rmsle(yva_raw, np.clip(mdl.predict(Xva_c), 0, None))

print(f"\n>>> Optuna CatBoost ({n_cb} trial)...")
s2 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
s2.optimize(obj_cb, n_trials=n_cb)
json.dump(s2.best_params, open(C.OUT / "best_cb_params.json", "w"), indent=2)
print(f"  best RMSLE: {s2.best_value:.4f} -> best_cb_params.json")
