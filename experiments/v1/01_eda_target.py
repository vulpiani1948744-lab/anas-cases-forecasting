"""
01 - EDA, deduplicazione, costruzione target mensile.

Output:
- output/target_mensile.parquet : (Strada, mese_data, num_richieste) post-dedup
- output/strade_test_status.csv  : copertura strade test vs train
- stampe a console con statistiche chiave
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent           # cartella experiments/
OUT_DIR = REPO_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# %% carica i dati
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Data Ins", "mese_data"], low_memory=False)
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["mese_data"])
road_meta = pd.read_csv(DATA_DIR / "road_metadata.csv", sep=";")

print("=" * 60)
print("SHAPES")
print("=" * 60)
print(f"train       : {train.shape}")
print(f"test        : {test.shape}")
print(f"road_meta   : {road_meta.shape}")

# %% panoramica train
print("\n" + "=" * 60)
print("TRAIN: colonne, dtype, % missing")
print("=" * 60)
info = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "pct_missing": (train.isna().mean() * 100).round(1),
    "n_unique": train.nunique(),
})
print(info)

print(f"\nIntervallo date Data Ins: {train['Data Ins'].min()} → {train['Data Ins'].max()}")
print(f"Intervallo mese_data    : {train['mese_data'].min()} → {train['mese_data'].max()}")

# %% righe senza Strada (non rilevanti per il forecasting)
n_no_strada = train["Strada"].isna().sum()
print(f"\nRighe senza Strada: {n_no_strada} ({n_no_strada/len(train)*100:.1f}%)")
print("→ verranno escluse dal target (il test richiede solo strade specifiche)")

# %% strade test vs train: chi è in test ma non in train?
strade_train = set(train["Strada"].dropna().unique())
strade_test = set(test["strada"].unique())
strade_meta = set(road_meta["strada"].unique())

print("\n" + "=" * 60)
print("COPERTURA STRADE")
print("=" * 60)
print(f"Strade uniche in train         : {len(strade_train)}")
print(f"Strade uniche in test          : {len(strade_test)}")
print(f"Strade uniche in road_metadata : {len(strade_meta)}")
print(f"Strade test PRESENTI in train  : {len(strade_test & strade_train)}")
print(f"Strade test MAI VISTE in train : {len(strade_test - strade_train)}")
print(f"Strade test in road_metadata   : {len(strade_test & strade_meta)}")

# salva status copertura
status = pd.DataFrame({
    "strada": sorted(strade_test),
    "in_train": [s in strade_train for s in sorted(strade_test)],
    "in_road_metadata": [s in strade_meta for s in sorted(strade_test)],
})
status.to_csv(OUT_DIR / "strade_test_status.csv", index=False)
print(f"\n→ salvato {OUT_DIR / 'strade_test_status.csv'}")

# %% deduplicazione: stessa segnalazione = righe identiche su tutte le colonne descrittive (ignorando Data Ins/Ora Ins)
print("\n" + "=" * 60)
print("DEDUPLICAZIONE")
print("=" * 60)

dedup_keys = [
    "Localita", "Provincia", "Canale arrivo", "Tipologia",
    "Categoria", "Gruppo", "Strada", "Tipo Persona",
]

# escludiamo righe senza Strada (non utili)
train_clean = train.dropna(subset=["Strada"]).copy()

# trasformiamo NaN nelle chiavi in stringa "__NA__" per non confondere il dedup
for c in dedup_keys:
    train_clean[c] = train_clean[c].fillna("__NA__")

# aggiungiamo mese_data alla chiave dedup: due segnalazioni identiche in mesi diversi → eventi diversi
# (la dedup serve a togliere righe ripetute per la stessa segnalazione, di solito nello stesso periodo)
dedup_keys_with_month = dedup_keys + ["mese_data"]

n_before = len(train_clean)
train_dedup = train_clean.drop_duplicates(subset=dedup_keys_with_month, keep="first")
n_after = len(train_dedup)

print(f"Righe prima  : {n_before:>8d}")
print(f"Righe dopo   : {n_after:>8d}")
print(f"Duplicate    : {n_before - n_after:>8d} ({(n_before - n_after)/n_before*100:.1f}%)")

# %% costruzione target mensile
print("\n" + "=" * 60)
print("TARGET MENSILE")
print("=" * 60)

target = (
    train_dedup
    .groupby(["Strada", "mese_data"])
    .size()
    .reset_index(name="num_richieste")
)

print(f"Righe nel target              : {len(target)}")
print(f"Strade con almeno 1 mese      : {target['Strada'].nunique()}")
print(f"Mesi unici                    : {target['mese_data'].nunique()}")
print("\nStatistiche num_richieste:")
print(target["num_richieste"].describe().round(2))

print("\nDistribuzione (quantili estesi):")
print(target["num_richieste"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(1))

# %% completiamo la griglia (strada, mese) per le strade attive
# nota: il README dice che (strada, mese) mancanti NON sono zero ma "non osservati"
# però per le strade che SONO presenti nel train, i mesi mancanti tra primo e ultimo
# evento è ragionevole trattarli come 0 (almeno per training)
print("\n" + "=" * 60)
print("GRIGLIA COMPLETA (strade attive × mesi tra primo e ultimo evento)")
print("=" * 60)

all_months = pd.date_range(target["mese_data"].min(), target["mese_data"].max(), freq="MS")
print(f"Range mesi: {all_months.min()} → {all_months.max()} ({len(all_months)} mesi)")

# per ogni strada teniamo solo i mesi tra il suo primo e ultimo evento
def expand_strada(g):
    months = pd.date_range(g["mese_data"].min(), g["mese_data"].max(), freq="MS")
    return pd.DataFrame({"mese_data": months})

grid = (
    target.groupby("Strada", group_keys=True)
    .apply(expand_strada, include_groups=False)
    .reset_index()
    .drop(columns=[c for c in ["level_1"] if c in target.columns])
)
grid = grid[["Strada", "mese_data"]]

target_full = grid.merge(target, on=["Strada", "mese_data"], how="left")
target_full["num_richieste"] = target_full["num_richieste"].fillna(0).astype(int)

print(f"Righe target completo: {len(target_full)}")
print(f"  di cui osservate   : {(target_full['num_richieste'] > 0).sum()}")
print(f"  di cui zero        : {(target_full['num_richieste'] == 0).sum()}")

# %% salva
target_full.to_csv(OUT_DIR / "target_mensile.csv", index=False)
print(f"\n→ salvato {OUT_DIR / 'target_mensile.csv'}")
print("\nPrime righe:")
print(target_full.head(10))
