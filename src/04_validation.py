"""
04 - Validazione walk-forward out-of-time + scelta della loss function.

Pilastro statistico: nelle serie storiche il tempo scorre in una direzione.
Vietato il K-Fold casuale (userebbe il futuro per prevedere il passato).
Qui validiamo OUT-OF-TIME, addestrando sul passato e validando sull'anno dopo:
  fold 1: train <=2020  -> val 2021
  fold 2: train <=2021  -> val 2022
  fold 3: train <=2022  -> val 2023
Questo mima esattamente la produzione (prevedere un futuro mai visto).

Contemporaneamente confrontiamo tre loss function su LightGBM per giustificare
la scelta del modello nel report:
  - regression  : MSE su log1p(y)   (proxy classico dell'RMSLE)
  - poisson     : conteggi, equidispersione
  - tweedie     : conteggi zero-inflated e sovradispersi (compound Poisson-Gamma)

Metrica riportata: RMSLE (metrica del contest) + devianza di Poisson, per fold.
Output: output/validation_walkforward.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import config as C
from _common import load_dataset, feature_cols, oot_fold

df = load_dataset()
feats = feature_cols(df)
cats = [c for c in C.CAT_COLS if c in feats]

LOSSES = {
    "regression_logy": dict(objective="regression", metric="rmse", _logy=True),
    "poisson":         dict(objective="poisson", metric="poisson", _logy=False),
    "tweedie":         dict(objective="tweedie", metric="tweedie",
                            tweedie_variance_power=C.TWEEDIE_VP, _logy=False),
}

rows = []
for end_year, val_year in C.WALK_FORWARD:
    tr, va = oot_fold(df, end_year, val_year)
    Xtr, Xva = tr[feats], va[feats]
    yva_raw = va["num_richieste"].values
    for name, cfg in LOSSES.items():
        cfg = dict(cfg)
        logy = cfg.pop("_logy")
        ytr = np.log1p(tr["num_richieste"].values) if logy else tr["num_richieste"].values
        yva = np.log1p(yva_raw) if logy else yva_raw
        params = {**C.DEFAULT_LGBM, **cfg, "verbose": -1, "seed": 42}
        dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats)
        dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr)
        m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pred = m.predict(Xva, num_iteration=m.best_iteration)
        if logy:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        rows.append({
            "fold": f"train<={end_year}->val{val_year}", "loss": name,
            "n_iter": m.best_iteration,
            "rmsle": round(C.rmsle(yva_raw, pred), 4),
            "poisson_dev": round(C.poisson_deviance(yva_raw, pred), 4),
        })
        print(f"  [{val_year}] {name:16s} RMSLE={rows[-1]['rmsle']:.4f} "
              f"PoisDev={rows[-1]['poisson_dev']:.4f}")

res = pd.DataFrame(rows)
res.to_csv(C.OUT / "validation_walkforward.csv", index=False)

print("\n" + "=" * 60)
print("MEDIA WALK-FORWARD per loss (su tutti i fold):")
summary = res.groupby("loss")[["rmsle", "poisson_dev"]].mean().round(4)
print(summary.to_string())
best = summary["rmsle"].idxmin()
print(f"\n=> loss con miglior RMSLE medio out-of-time: {best}")
print("-> output/validation_walkforward.csv")
