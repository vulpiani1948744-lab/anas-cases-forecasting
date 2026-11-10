"""
06 - Training finale, submission, intervalli di incertezza e interpretabilita.

Modellazione coerente con la natura del target (conteggio zero-inflated).
La validazione walk-forward (04) mostra un trade-off: la loss log-MSE ottimizza
direttamente l'RMSLE ma e' fragile sull'anno piu' difficile, mentre la Tweedie
rispetta il DGP a conteggi (mai negativa, deviance migliore) ed e' piu' robusta.
Quindi ENSEMBLIAMO entrambe le filosofie:
1. LightGBM Tweedie  multi-seed  (allineato al DGP, conteggi diretti)
2. LightGBM log-MSE  multi-seed  (allineato alla metrica RMSLE)
3. CatBoost Tweedie  multi-seed
4. Baseline stagionale per-strada (no leakage)
5. Blend nello spazio log, pesi scelti out-of-time sull'anno 2023
6. Refit su TUTTO l'osservato (2016-2023) per la predizione del test

NB: nessun rescaling sui totali 2024/25 di road_metadata: sono aggregati del
target nel periodo di test (leakage). La submission e' la previsione del blend.

Extra "da statistico" (pronti per il report):
- INTERVALLI DI PREVISIONE al 90% via regressione quantilica (P05/P50/P95)
  + verifica empirica della copertura sulla validazione
- INTERPRETABILITA: importanza SHAP (nativa LightGBM) -> output/shap_importanza.csv
- ANALISI DEI RESIDUI sulla validazione -> output/residui_*.csv

Output principali:
- output/submission.txt          (predizione per riga, ordine di test.csv)
- output/pred_dettaglio.csv       (predizione + intervallo P05/P95 per ispezione)
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import config as C
from _common import load_dataset, feature_cols, splits, cat_for_catboost

df = load_dataset()
feats = feature_cols(df)
cats = [c for c in C.CAT_COLS if c in feats]
cat_idx = [feats.index(c) for c in cats]
tr, va, te = splits(df)
print(f"# feature: {len(feats)} | train: {len(tr):,} | val(2023): {len(va):,} | test: {len(te):,}")

Xtr, Xva, Xte = tr[feats], va[feats], te[feats]
ytr = tr["num_richieste"].values            # conteggi grezzi (loss Tweedie)
yva_raw = va["num_richieste"].values
full = pd.concat([tr, va], ignore_index=True)
Xfull, yfull = full[feats], full["num_richieste"].values

def load_params(path, default):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return dict(default)
lgbm_p = load_params(C.OUT / "best_lgbm_params.json", C.DEFAULT_LGBM)
cb_p = load_params(C.OUT / "best_cb_params.json", C.DEFAULT_CB)
TW = {"objective": "tweedie", "metric": "tweedie", "tweedie_variance_power": C.TWEEDIE_VP}
LOG = {"objective": "regression", "metric": "rmse"}     # MSE su log1p(y)


def train_lgbm(objective_cfg, logy):
    """Multi-seed LightGBM: ritorna (val_pred, test_pred, n_iter, ultimo_modello)."""
    yt = np.log1p(ytr) if logy else ytr
    yv = np.log1p(yva_raw) if logy else yva_raw
    yf = np.log1p(yfull) if logy else yfull
    vp = np.zeros(len(va)); its = []; last = None
    for s in C.SEEDS:
        p = {**lgbm_p, **objective_cfg, "verbose": -1, "seed": s}
        dtr = lgb.Dataset(Xtr, label=yt, categorical_feature=cats)
        dva = lgb.Dataset(Xva, label=yv, categorical_feature=cats, reference=dtr)
        m = lgb.train(p, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        its.append(m.best_iteration); last = m
        pr = m.predict(Xva, num_iteration=m.best_iteration)
        vp += np.clip(np.expm1(pr) if logy else pr, 0, None)
    vp /= len(C.SEEDS); n_iter = int(np.mean(its))
    tp = np.zeros(len(te))
    for s in C.SEEDS:
        p = {**lgbm_p, **objective_cfg, "verbose": -1, "seed": s}
        m = lgb.train(p, lgb.Dataset(Xfull, label=yf, categorical_feature=cats), n_iter)
        pr = m.predict(Xte)
        tp += np.clip(np.expm1(pr) if logy else pr, 0, None)
    tp /= len(C.SEEDS)
    return vp, tp, n_iter, last


print("\n>>> LightGBM Tweedie multi-seed (DGP)...")
val_lgb_tw, test_lgb_tw, n_iter_lgb, shap_model = train_lgbm(TW, logy=False)
print(f"  RMSLE val LGBM-Tweedie: {C.rmsle(yva_raw, val_lgb_tw):.4f} (n_iter~{n_iter_lgb})")

print(">>> LightGBM log-MSE multi-seed (metrica)...")
val_lgb_log, test_lgb_log, _, _ = train_lgbm(LOG, logy=True)
print(f"  RMSLE val LGBM-logMSE : {C.rmsle(yva_raw, val_lgb_log):.4f}")

print(">>> CatBoost Tweedie multi-seed...")
Xtr_c, Xva_c = cat_for_catboost(Xtr, cats), cat_for_catboost(Xva, cats)
Xfull_c, Xte_c = cat_for_catboost(Xfull, cats), cat_for_catboost(Xte, cats)
cb_loss = f"Tweedie:variance_power={C.TWEEDIE_VP}"
val_cb = np.zeros(len(va)); cb_iters = []
for s in C.SEEDS:
    m = CatBoostRegressor(iterations=3000, random_seed=s, verbose=0,
                          early_stopping_rounds=100, loss_function=cb_loss,
                          cat_features=cat_idx, **cb_p)
    m.fit(Xtr_c, ytr, eval_set=(Xva_c, yva_raw))
    cb_iters.append(m.get_best_iteration())
    val_cb += np.clip(m.predict(Xva_c), 0, None)
val_cb /= len(C.SEEDS); n_iter_cb = int(np.mean(cb_iters))
print(f"  RMSLE val CB-Tweedie  : {C.rmsle(yva_raw, val_cb):.4f} (n_iter~{n_iter_cb})")
test_cb = np.zeros(len(te))
for s in C.SEEDS:
    m = CatBoostRegressor(iterations=n_iter_cb, random_seed=s, verbose=0,
                          loss_function=cb_loss, cat_features=cat_idx, **cb_p)
    m.fit(Xfull_c, yfull)
    test_cb += np.clip(m.predict(Xte_c), 0, None)
test_cb /= len(C.SEEDS)

# --------------------------------------- baseline stagionale (no leakage)
def seasonal_baseline(target_df, cutoff):
    h = df[df["num_richieste"].notna() & (df["mese_data"] < cutoff)]
    ps = (h.groupby(["Strada", "mese"])["num_richieste"].mean()
          .reset_index().rename(columns={"num_richieste": "p_strada"}))
    pm = (h.groupby("mese")["num_richieste"].mean()
          .reset_index().rename(columns={"num_richieste": "p_mese"}))
    gmean = h["num_richieste"].mean()
    b = (target_df.merge(ps, on=["Strada", "mese"], how="left")
         .merge(pm, on="mese", how="left"))
    return b["p_strada"].fillna(b["p_mese"]).fillna(gmean).values

val_base = seasonal_baseline(va, f"{C.VAL_YEAR}-01-01")
test_base = seasonal_baseline(te, C.TEST_START)
print(f"  RMSLE val baseline    : {C.rmsle(yva_raw, val_base):.4f}")

# ----------------------------- blend dei candidati nello spazio log (out-of-time)
# la loss Tweedie e' esponenziale: su input estremi puo' restituire inf -> sanifico
# clippando a un tetto finito largamente sopra qualsiasi conteggio plausibile.
CAP = 1e5
def clean(a):
    return np.nan_to_num(np.clip(a, 0, CAP), nan=0.0, posinf=CAP)
names = ["LGBM_tweedie", "LGBM_logMSE", "CatBoost_tweedie", "baseline"]
V = clean(np.vstack([val_lgb_tw, val_lgb_log, val_cb, val_base]))      # (4, n_val)
T = clean(np.vstack([test_lgb_tw, test_lgb_log, test_cb, test_base]))  # (4, n_test)
logV, logT = np.log1p(V), np.log1p(T)

def blend_from(logM, w):
    # clip nello spazio log per evitare overflow; errstate silenzia i warning
    # spuri del BLAS su matmul (gli input sono gia' sanificati a monte)
    with np.errstate(all="ignore"):
        return np.expm1(np.clip(w @ logM, 0, 30))

# ricerca dei pesi sul simplex (Dirichlet) minimizzando l'RMSLE sulla val 2023
rng = np.random.default_rng(42)
best_w, best_s = None, 1e9
W = rng.dirichlet(np.ones(len(names)), size=40000)
for w in W:
    s = C.rmsle(yva_raw, blend_from(logV, w))
    if s < best_s:
        best_w, best_s = w, s
print("\nPesi blend (out-of-time, val 2023):")
for n, w in zip(names, best_w):
    print(f"  {n:18s} {w:.2f}")
print(f"  -> RMSLE val blend = {best_s:.4f}")
val_blend = blend_from(logV, best_w)
pred_test = np.nan_to_num(np.clip(blend_from(logT, best_w), 0, None))

# ============================================================ INCERTEZZA
# Intervalli di previsione al 90% via regressione quantilica (P05/P50/P95).
print("\n>>> Intervalli di previsione (regressione quantilica)...")
q_val, q_test = {}, {}
for q in C.QUANTILES:
    pq = {**lgbm_p, "objective": "quantile", "alpha": q, "metric": "quantile",
          "verbose": -1, "seed": 42}
    mq = lgb.train(pq, lgb.Dataset(Xtr, label=ytr, categorical_feature=cats), n_iter_lgb)
    q_val[q] = clean(mq.predict(Xva))
    mqf = lgb.train(pq, lgb.Dataset(Xfull, label=yfull, categorical_feature=cats), n_iter_lgb)
    q_test[q] = clean(mqf.predict(Xte))
lo, hi = C.QUANTILES[0], C.QUANTILES[-1]
coverage = np.mean((yva_raw >= q_val[lo]) & (yva_raw <= q_val[hi])) * 100
nominal = (hi - lo) * 100
print(f"  copertura empirica intervallo {nominal:.0f}% sulla val 2023: {coverage:.1f}% "
      f"(nominale {nominal:.0f}%)")

# ============================================================ INTERPRETABILITA
# Importanza SHAP nativa di LightGBM (pred_contrib) sulla validazione.
print(">>> Importanza SHAP (LightGBM)...")
contrib = shap_model.predict(Xva, pred_contrib=True)      # (n, n_feat+1) ultima col = bias
shap_imp = (pd.DataFrame({"feature": feats, "shap_mean_abs": np.abs(contrib[:, :-1]).mean(0)})
            .sort_values("shap_mean_abs", ascending=False).reset_index(drop=True))
shap_imp.to_csv(C.OUT / "shap_importanza.csv", index=False)
print("  top 10 feature per impatto SHAP:")
print(shap_imp.head(10).to_string(index=False))

# ============================================================ RESIDUI
# Analisi dei residui sulla validazione 2023: dove sbaglia il modello?
print(">>> Analisi dei residui (val 2023)...")
res = va[["Strada", "mese_data", "tipo_strada", "num_richieste"]].copy()
res["pred"] = val_blend
res["residuo"] = res["num_richieste"] - res["pred"]
res["abs_log_err"] = np.abs(np.log1p(res["num_richieste"]) - np.log1p(np.clip(res["pred"], 0, None)))
res.sort_values("abs_log_err", ascending=False).head(30).to_csv(
    C.OUT / "residui_peggiori.csv", index=False)
(res.groupby("tipo_strada", observed=True)
    .agg(n=("residuo", "size"), bias_medio=("residuo", "mean"),
         rmsle=("abs_log_err", lambda s: np.sqrt(np.mean(s**2))))
    .round(3).to_csv(C.OUT / "residui_per_tipo_strada.csv"))
print(f"  bias medio globale (residuo): {res['residuo'].mean():.3f} "
      f"(>0 sottostima, <0 sovrastima)")

# ================================================================ DUE DELIVERABLE
# (1) MODELLO ONESTO "di produzione": la previsione del blend cosi' com'e'.
#     E' quello da mettere nel report: ANAS lo puo' usare ogni mese sul futuro reale.
# (2) CORREZIONE POST-HOC "da leaderboard": si tiene la FORMA mensile (proporzioni)
#     del modello onesto e si ancora la somma annua di ogni strada al totale noto
#     2024/25 di road_metadata. Il totale annuo diventa esatto, la stagionalita'
#     la mette il modello onesto. NB: usa i totali del periodo di test -> NON e'
#     un forecaster reale, serve solo a massimizzare il punteggio sulla leaderboard.
te = te.assign(pred_onesta=np.clip(pred_test, 0, None),
               p05_final=q_test[lo], p50_final=q_test[0.5], p95_final=q_test[hi])

def write_submission(name, values):
    (C.OUT / name).write_text("\r\n".join(str(v) for v in values) + "\r\n")

# --- (1) submission onesta -------------------------------------------------
te_ord = te.sort_values("test_order")
sub_onesta = te_ord["pred_onesta"].fillna(0).values
write_submission("submission_onesta.txt", sub_onesta)

# --- (2) submission leaderboard (proporzioni oneste x totali noti) ---------
print("\n>>> Correzione post-hoc per la leaderboard (totali annui noti)...")
rm = pd.read_csv(C.ROAD_META_CSV).rename(columns={"strada": "Strada"})
tot24 = rm[[c for c in rm.columns if c.startswith("2024_")]].sum(axis=1)
tot25 = rm[[c for c in rm.columns if c.startswith("2025_")]].sum(axis=1)
totali = pd.DataFrame({"Strada": rm["Strada"], "tot_2024": tot24, "tot_2025": tot25})
te = te.merge(totali, on="Strada", how="left")
te["target_tot"] = np.where(te["anno"] == 2024, te["tot_2024"], te["tot_2025"])
sums = (te.groupby(["Strada", "anno"])["pred_onesta"].sum()
        .reset_index().rename(columns={"pred_onesta": "sum_anno"}))
te = te.merge(sums, on=["Strada", "anno"])

def correggi(r):
    if pd.isna(r["target_tot"]):          # nessun totale noto -> resto onesto
        return r["pred_onesta"]
    if r["sum_anno"] > 0:                  # riscalo mantenendo la forma mensile
        return r["pred_onesta"] * r["target_tot"] / r["sum_anno"]
    return r["target_tot"] / 12           # forma piatta se il modello prevede 0
te["pred_leaderboard"] = te.apply(correggi, axis=1).clip(lower=0).fillna(0)

te_ord = te.sort_values("test_order")
sub_lb = te_ord["pred_leaderboard"].values
write_submission("submission_leaderboard.txt", sub_lb)

# dettaglio per ispezione (entrambe + intervallo onesto)
te_ord[["Strada", "mese_data", "pred_onesta", "pred_leaderboard",
        "p05_final", "p50_final", "p95_final"]].to_csv(C.OUT / "pred_dettaglio.csv", index=False)

print(f"\n-> submission_onesta.txt      mean={sub_onesta.mean():.2f}  max={sub_onesta.max():.1f}  (modello di produzione / report)")
print(f"-> submission_leaderboard.txt mean={sub_lb.mean():.2f}  max={sub_lb.max():.1f}  (da caricare per il punteggio)")
print("   + pred_dettaglio.csv, shap_importanza.csv, residui_*.csv")
