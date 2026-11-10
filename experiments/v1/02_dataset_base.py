"""
02 - Costruisce il dataset base:
- griglia (Strada, mese_data) che copre TRAIN (2016-01 → 2023-12) + TEST (2024-01 → 2025-12)
- merge con road_metadata
- feature calendariali (giorni nel mese, weekend, festività italiane, vacanze scolastiche)
- tipo_strada estratto dal prefisso del nome

Output:
- output/dataset_base.csv : una riga per (Strada, mese_data), con target solo per il periodo train
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent           # cartella experiments/
OUT_DIR = REPO_DIR / "output"

# %% carica i pezzi
target = pd.read_csv(OUT_DIR / "target_mensile.csv", parse_dates=["mese_data"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["mese_data"])
road_meta = pd.read_csv(DATA_DIR / "road_metadata.csv", sep=";")

test = test.rename(columns={"strada": "Strada"})
road_meta = road_meta.rename(columns={"strada": "Strada"})

# forza conversione numerica (road_metadata ha celle vuote che pandas legge come stringa)
numeric_cols = ["n_tratte", "comuni_attraversati", "province_attraversate",
                "km_min", "km_max", "consistenza_gg_media", "leggeri_media", "pesanti_media"]
for c in numeric_cols:
    road_meta[c] = pd.to_numeric(road_meta[c], errors="coerce")

print("road_metadata: % righe con tutti i numerici NaN: "
      f"{road_meta[numeric_cols].isna().all(axis=1).mean()*100:.1f}%")

print(f"target shape   : {target.shape}")
print(f"test shape     : {test.shape}")
print(f"road_meta shape: {road_meta.shape}")

# %% costruisci griglia completa: strade in target ∪ strade in test, su tutti i mesi 2016-01 → 2025-12
all_strade = sorted(set(target["Strada"].unique()) | set(test["Strada"].unique()))
all_months = pd.date_range("2016-01-01", "2025-12-01", freq="MS")

print(f"\nStrade totali (target ∪ test): {len(all_strade)}")
print(f"Mesi totali 2016-01 → 2025-12 : {len(all_months)}")

# per il target: griglia espansa per OGNI strada × OGNI mese del periodo train
# per le strade nuove (in test ma non in train) avranno solo NaN come num_richieste durante il train
train_months = pd.date_range("2016-01-01", "2023-12-01", freq="MS")
test_months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

# griglia train: tutte le strade × tutti i mesi train
grid_train = pd.MultiIndex.from_product(
    [all_strade, train_months], names=["Strada", "mese_data"]
).to_frame(index=False)
grid_train = grid_train.merge(target, on=["Strada", "mese_data"], how="left")
# strade già viste in train: imputo 0 nei mesi mancanti tra primo e ultimo evento (logica del target_mensile)
# strade nuove: lascio NaN (non possiamo dire nulla)
seen_strade = set(target["Strada"].unique())
print(f"Strade viste in train     : {len(seen_strade)}")
print(f"Strade nuove (solo test)  : {len(set(all_strade) - seen_strade)}")

# per le strade nuove, num_richieste resta NaN su tutto il train
grid_train.loc[~grid_train["Strada"].isin(seen_strade), "num_richieste"] = np.nan
# per le strade viste, gli zeri imputati da target_mensile sono già lì; i mesi PRIMA del primo
# evento e DOPO l'ultimo evento restano NaN, e li trattiamo come "non osservato" (escludi da training)
print(f"\nRighe griglia train       : {len(grid_train)}")
print(f"  con target (osservato)  : {grid_train['num_richieste'].notna().sum()}")
print(f"  senza target (NaN)      : {grid_train['num_richieste'].isna().sum()}")

# griglia test: solo le strade del test × mesi 2024-2025 (rispettiamo l'ordine esatto di test.csv!)
grid_test = test.copy()
grid_test["num_richieste"] = np.nan
grid_test["test_order"] = np.arange(len(grid_test))  # per ricostruire l'ordine al momento submission

# concateno mantenendo un flag train/test
grid_train["split"] = "train"
grid_test["split"] = "test"
grid = pd.concat([grid_train, grid_test], ignore_index=True)
grid["test_order"] = grid["test_order"].astype("Int64")

print(f"\nRighe griglia totale (train+test): {len(grid)}")

# %% merge road_metadata
print("\nMerge road_metadata...")
grid = grid.merge(road_meta, on="Strada", how="left")
print(f"Strade senza metadata: {grid[grid['n_tratte'].isna() & grid['Strada'].isin(test['Strada'])]['Strada'].nunique()}")

# lunghezza strada
grid["lunghezza_km"] = grid["km_max"] - grid["km_min"]
# quota pesanti su totale
grid["quota_pesanti"] = grid["pesanti_media"] / grid["consistenza_gg_media"].replace(0, np.nan)

# %% tipo_strada dal prefisso del nome
def estrai_tipo(nome):
    if pd.isna(nome):
        return "ALTRO"
    nome_up = nome.upper()
    if nome_up.startswith("AU "):
        return "AUTOSTRADA_DIR" if "DIR" in nome_up else "AUTOSTRADA"
    if nome_up.startswith("SS "):
        return "STATALE"
    if nome_up.startswith("RA "):
        return "RACCORDO"
    if nome_up.startswith("NSA "):
        return "NUOVA_STATALE"
    return "ALTRO"

grid["tipo_strada"] = grid["Strada"].apply(estrai_tipo)
print("\nDistribuzione tipo_strada:")
print(grid.drop_duplicates("Strada")["tipo_strada"].value_counts())

# %% feature calendariali per mese (calcolate una volta sola per mese unico, poi join)
print("\nCalcolo feature calendariali...")
italy_holidays = holidays.country_holidays("IT", years=range(2016, 2026))

# vacanze scolastiche italiane (approssimazione: vale per la media nazionale)
# - estate: 12 giugno - 14 settembre
# - Natale: 24 dicembre - 6 gennaio
# - Pasqua: 5 giorni intorno a Pasqua (da venerdì santo a martedì dopo)
def is_vacanza_scolastica(d, easter_dates):
    # estate
    if d.month in (7, 8):
        return True
    if d.month == 6 and d.day >= 12:
        return True
    if d.month == 9 and d.day <= 14:
        return True
    # Natale
    if d.month == 12 and d.day >= 24:
        return True
    if d.month == 1 and d.day <= 6:
        return True
    # Pasqua (±2 gg)
    for e in easter_dates:
        if abs((d.date() - e).days) <= 2:
            return True
    return False

# Pasqua per anno (calcolo dalla libreria holidays - lo cerco fra i festivi)
easter_by_year = {}
for y in range(2016, 2026):
    for date, name in italy_holidays.items():
        if date.year == y and "Pasqua" in str(name):
            easter_by_year[y] = date
            break

print(f"Pasqua per anno: {easter_by_year}")

def feature_mese(mese_data):
    """Calcola tutte le feature calendariali per un mese (primo giorno)."""
    year, month = mese_data.year, mese_data.month
    # tutti i giorni del mese
    giorni = pd.date_range(mese_data, mese_data + pd.offsets.MonthEnd(0), freq="D")
    n_giorni = len(giorni)
    n_weekend = sum(d.weekday() >= 5 for d in giorni)
    n_lunedi = sum(d.weekday() == 0 for d in giorni)
    n_venerdi = sum(d.weekday() == 4 for d in giorni)
    n_festivi = sum(d in italy_holidays for d in giorni)
    # ponti: festivi infrasettimanali (martedì o giovedì o venerdì) → genera ponte tipico
    n_ponti = sum(d in italy_holidays and d.weekday() in (0, 1, 3, 4) for d in giorni)
    easter_dates = list(easter_by_year.values())
    n_vacanze = sum(is_vacanza_scolastica(d, easter_dates) for d in giorni)
    return pd.Series({
        "giorni_nel_mese": n_giorni,
        "n_weekend": n_weekend,
        "n_lunedi": n_lunedi,
        "n_venerdi": n_venerdi,
        "n_festivi": n_festivi,
        "n_ponti": n_ponti,
        "n_vacanze_scolastiche": n_vacanze,
    })

mesi_unici = pd.DataFrame({"mese_data": all_months})
mesi_unici[["giorni_nel_mese", "n_weekend", "n_lunedi", "n_venerdi",
            "n_festivi", "n_ponti", "n_vacanze_scolastiche"]] = (
    mesi_unici["mese_data"].apply(feature_mese)
)
mesi_unici["anno"] = mesi_unici["mese_data"].dt.year
mesi_unici["mese"] = mesi_unici["mese_data"].dt.month
mesi_unici["sin_mese"] = np.sin(2 * np.pi * mesi_unici["mese"] / 12)
mesi_unici["cos_mese"] = np.cos(2 * np.pi * mesi_unici["mese"] / 12)
mesi_unici["is_estate"] = mesi_unici["mese"].isin([6, 7, 8]).astype(int)
mesi_unici["is_inverno"] = mesi_unici["mese"].isin([12, 1, 2]).astype(int)
mesi_unici["is_agosto"] = (mesi_unici["mese"] == 8).astype(int)
mesi_unici["is_dicembre"] = (mesi_unici["mese"] == 12).astype(int)
mesi_unici["trend"] = (mesi_unici["mese_data"] - mesi_unici["mese_data"].min()).dt.days / 30.4

print("\nFeature calendariali (prime 3 righe):")
print(mesi_unici.head(3))

# %% merge calendario sulla griglia
grid = grid.merge(mesi_unici, on="mese_data", how="left")

# %% sanity checks
print("\n" + "=" * 60)
print("SANITY CHECK DATASET BASE")
print("=" * 60)
print(f"Righe totali: {len(grid)}")
print(f"Strade uniche: {grid['Strada'].nunique()}")
print(f"Split: \n{grid['split'].value_counts()}")
print(f"Righe TEST (devono essere 15048): {(grid['split']=='test').sum()}")
print(f"\nColonne finali: {len(grid.columns)}")
print(list(grid.columns))

# verifica ordine test preservato
test_ordered = grid[grid["split"] == "test"].sort_values("test_order")
print(f"\nPrime 3 righe test (ordine originale):")
print(test_ordered[["Strada", "mese_data", "test_order"]].head(3))
print(f"\nOrdine match con test.csv: {(test_ordered['Strada'].values == test['Strada'].values).all() and (test_ordered['mese_data'].values == test['mese_data'].values).all()}")

# %% salva
out_path = OUT_DIR / "dataset_base.csv"
grid.to_csv(out_path, index=False)
print(f"\n→ salvato {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")
