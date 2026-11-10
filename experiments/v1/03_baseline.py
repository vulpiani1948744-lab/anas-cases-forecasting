"""
03 - Baseline e split di validazione.

Strategia:
- Split temporale: train 2016-2022, validation 2023, test 2024-2025.
- Baseline 1 (NAIVE): media storica per (Strada, mese-del-calendario) calcolata fino a 2022.
- Baseline 2 (NAIVE_SHRINK): blend tra media strada×mese e media globale per le strade nuove.
- Metrica: RMSLE su validation.

Output:
- output/submission_baseline.txt : prima submission valida (formato richiesto)
- stampa RMSLE sulla validation (numero da battere col modello principale)
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica dataset base
df = pd.read_csv(OUT_DIR / "dataset_base.csv", parse_dates=["mese_data"])
print(f"Dataset base: {df.shape}")
print(f"Split: \n{df['split'].value_counts()}")

# %% split temporale
# train: tutto il train con num_richieste osservato fino a 2022-12
# validation: tutto il train con num_richieste osservato dal 2023-01 al 2023-12
df_train_full = df[df["split"] == "train"].copy()
mask_obs = df_train_full["num_richieste"].notna()
df_obs = df_train_full[mask_obs].copy()

is_train = df_obs["mese_data"] < "2023-01-01"
is_val = (df_obs["mese_data"] >= "2023-01-01") & (df_obs["mese_data"] < "2024-01-01")

train_set = df_obs[is_train]
val_set = df_obs[is_val]
test_set = df[df["split"] == "test"].copy()

print(f"\nTrain set: {len(train_set):>6d} righe ({train_set['Strada'].nunique()} strade, {train_set['mese_data'].nunique()} mesi)")
print(f"Val set  : {len(val_set):>6d} righe ({val_set['Strada'].nunique()} strade, {val_set['mese_data'].nunique()} mesi)")
print(f"Test set : {len(test_set):>6d} righe ({test_set['Strada'].nunique()} strade)")

# %% metrica RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# %% Baseline 1: media storica per (Strada, mese-del-calendario), calcolata sul train
print("\n" + "=" * 60)
print("BASELINE 1: media (Strada, mese-del-calendario)")
print("=" * 60)

profile = train_set.groupby(["Strada", "mese"])["num_richieste"].mean().reset_index()
profile = profile.rename(columns={"num_richieste": "pred_strada_mese"})

# media globale per mese (fallback per strade mai viste)
global_month = train_set.groupby("mese")["num_richieste"].mean().reset_index()
global_month = global_month.rename(columns={"num_richieste": "pred_global_mese"})

# media globale assoluta (fallback estremo)
global_mean = train_set["num_richieste"].mean()
print(f"Media globale assoluta: {global_mean:.2f}")

def predict_baseline(df_in):
    out = df_in.merge(profile, on=["Strada", "mese"], how="left")
    out = out.merge(global_month, on="mese", how="left")
    pred = out["pred_strada_mese"].fillna(out["pred_global_mese"]).fillna(global_mean)
    return pred.values

pred_val = predict_baseline(val_set)
score_val = rmsle(val_set["num_richieste"].values, pred_val)
print(f"\nRMSLE validation 2023 (Baseline 1): {score_val:.4f}")
print(f"  → numero di predizioni dalla strada×mese : {(val_set.merge(profile, on=['Strada','mese'], how='left')['pred_strada_mese'].notna()).sum()}")
print(f"  → numero di predizioni dal mese globale  : {(val_set.merge(profile, on=['Strada','mese'], how='left')['pred_strada_mese'].isna()).sum()}")

# diagnostica veloce
val_with_pred = val_set.copy()
val_with_pred["pred"] = pred_val
print(f"\nDistribuzione predizioni baseline su validation:")
print(val_with_pred["pred"].describe().round(2))
print(f"\nVero target su validation:")
print(val_set["num_richieste"].describe().round(2))

# %% Baseline 2: stesso ma SHRINKAGE verso media globale per strade con poca storia
print("\n" + "=" * 60)
print("BASELINE 2: media strada×mese con shrinkage (poche osservazioni → tira verso media globale)")
print("=" * 60)

# count di osservazioni per (Strada, mese) nel train
count_per_strada_mese = train_set.groupby(["Strada", "mese"]).size().reset_index(name="n_obs")
profile_s = profile.merge(count_per_strada_mese, on=["Strada", "mese"])
profile_s = profile_s.merge(global_month, on="mese")

# shrinkage: w = n_obs / (n_obs + k), k = "forza" del prior
K = 3.0
profile_s["pred_shrunk"] = (
    profile_s["n_obs"] / (profile_s["n_obs"] + K) * profile_s["pred_strada_mese"]
    + K / (profile_s["n_obs"] + K) * profile_s["pred_global_mese"]
)

def predict_baseline_shrunk(df_in):
    out = df_in.merge(profile_s[["Strada", "mese", "pred_shrunk"]], on=["Strada", "mese"], how="left")
    out = out.merge(global_month, on="mese", how="left")
    pred = out["pred_shrunk"].fillna(out["pred_global_mese"]).fillna(global_mean)
    return pred.values

pred_val_s = predict_baseline_shrunk(val_set)
score_val_s = rmsle(val_set["num_richieste"].values, pred_val_s)
print(f"\nRMSLE validation 2023 (Baseline 2 shrunk, K={K}): {score_val_s:.4f}")

# %% genera submission sulla baseline migliore
best_baseline = "shrunk" if score_val_s < score_val else "naive"
print(f"\nMiglior baseline: {best_baseline}")

if best_baseline == "shrunk":
    pred_test = predict_baseline_shrunk(test_set)
else:
    pred_test = predict_baseline(test_set)

# riordina secondo test_order originale
test_set = test_set.assign(pred=pred_test).sort_values("test_order")
submission = test_set[["pred"]].rename(columns={"pred": "num_richieste"})
submission["num_richieste"] = np.clip(submission["num_richieste"], 0, None)

sub_path = OUT_DIR / "submission_baseline.txt"
submission.to_csv(sub_path, index=False)
print(f"\n→ salvato {sub_path}")

# sanity check submission
print(f"\n--- Sanity check submission ---")
print(f"Righe: {len(submission)} (devono essere 15048)")
print(f"Colonne: {list(submission.columns)}")
print(f"Min/Max/Mean: {submission['num_richieste'].min():.2f} / {submission['num_richieste'].max():.2f} / {submission['num_richieste'].mean():.2f}")
print(f"NaN: {submission['num_richieste'].isna().sum()}")

# riassunto finale
print("\n" + "=" * 60)
print("RIASSUNTO BASELINE")
print("=" * 60)
print(f"Baseline 1 (naive)  RMSLE val 2023: {score_val:.4f}")
print(f"Baseline 2 (shrunk) RMSLE val 2023: {score_val_s:.4f}")
print(f"\n→ Numero da battere col modello principale: {min(score_val, score_val_s):.4f}")
