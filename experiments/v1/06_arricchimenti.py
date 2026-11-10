"""
06 - Aggiunge feature COVID dettagliate + feature incidenti al dataset finale.

Output:
- output/dataset_arricchito.csv : dataset_finale + feature aggiuntive
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import re

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica dataset finale (output di 05)
df = pd.read_csv(OUT_DIR / "dataset_finale.csv", parse_dates=["mese_data"])
print(f"Dataset finale: {df.shape}")

# %% COVID intensity index (mensile, nazionale) — approssimazione storica delle restrizioni
covid_intensity = {
    # 2020
    "2020-01": 0, "2020-02": 0,
    "2020-03": 5, "2020-04": 5, "2020-05": 4,  # lockdown nazionale
    "2020-06": 2, "2020-07": 1, "2020-08": 1, "2020-09": 2,
    "2020-10": 3, "2020-11": 4, "2020-12": 3,  # ritorno restrizioni
    # 2021
    "2021-01": 3, "2021-02": 3, "2021-03": 3, "2021-04": 3,  # zone colorate intense
    "2021-05": 2, "2021-06": 1, "2021-07": 0, "2021-08": 0, "2021-09": 1,
    "2021-10": 1, "2021-11": 1, "2021-12": 2,
    # 2022
    "2022-01": 2, "2022-02": 2, "2022-03": 1, "2022-04": 1,  # fine emergenza apr-22
    "2022-05": 0, "2022-06": 0, "2022-07": 0, "2022-08": 0,
    "2022-09": 0, "2022-10": 0, "2022-11": 0, "2022-12": 0,
}
df["covid_intensity"] = (
    df["mese_data"].dt.strftime("%Y-%m").map(covid_intensity).fillna(0).astype(float)
)
df["is_lockdown_nazionale"] = ((df["mese_data"] >= "2020-03-01") & (df["mese_data"] <= "2020-05-01")).astype(int)
df["is_zone_colorate"] = ((df["mese_data"] >= "2020-11-01") & (df["mese_data"] <= "2021-04-01")).astype(int)
df["covid_intensity_sq"] = df["covid_intensity"] ** 2  # cattura effetti non lineari

# trend di recupero post-pandemia (mesi dal 1 maggio 2022)
recovery = ((df["mese_data"] - pd.Timestamp("2022-05-01")).dt.days / 30.4).clip(lower=0)
df["mesi_recovery"] = recovery

print("\nCheck COVID intensity per anno:")
print(df.groupby(df["mese_data"].dt.year)["covid_intensity"].mean().round(2))

# %% incidenti.xlsx — strade con punti critici noti
incidenti = pd.read_excel(DATA_DIR / "incidenti.xlsx")
# normalizza colonne: rimuove \n e collassa spazi multipli
incidenti.columns = [re.sub(r"\s+", " ", c.replace("\n", " ")).strip() for c in incidenti.columns]
print("Colonne incidenti:", list(incidenti.columns))
incidenti = incidenti.rename(columns={
    "CODICE STRADA": "codice",
    "NOME STRADA": "nome_strada_inc",
})
print(f"\nIncidenti shape: {incidenti.shape}")
print("Strade con punti critici noti (codici):")
print(incidenti["codice"].tolist())

# costruisce un set di codici normalizzati per match
def normalize_code(s):
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    s = re.sub(r"[\s\.]+", "", s)  # rimuove spazi e punti
    return s

codici_critici = {normalize_code(c) for c in incidenti["codice"]}

# estrai codice strada dalla colonna Strada del dataset
def extract_code(strada):
    if pd.isna(strada):
        return None
    s = strada.upper().strip()
    # match "SS 14 ..." o "SS14 ..."
    m = re.match(r"^(AU\s*A?\d*|SS\s*\d+\w*|RA\s*\d*|NSA\s*\d*)", s)
    if m:
        return normalize_code(m.group(1))
    return None

df["codice_strada"] = df["Strada"].apply(extract_code)
df["is_strada_incidenti"] = df["codice_strada"].isin(codici_critici).astype(int)

# numero di centri abitati > 10k abitanti su strada (dalla tabella)
abitati_per_codice = incidenti.groupby(incidenti["codice"].apply(normalize_code)).agg(
    n_punti_critici=("CENTRO ABITATO > 10.000 ab.", lambda s: s.notna().sum())
).reset_index()
abitati_per_codice = abitati_per_codice.rename(columns={"codice": "codice_strada"})
df = df.merge(abitati_per_codice, on="codice_strada", how="left")
df["n_punti_critici"] = df["n_punti_critici"].fillna(0).astype(int)

print(f"\nStrade con flag incidenti: {df[df['is_strada_incidenti']==1]['Strada'].nunique()}")

# %% salva
df = df.drop(columns=["codice_strada"])
out_path = OUT_DIR / "dataset_arricchito.csv"
df.to_csv(out_path, index=False)
print(f"\n→ salvato {out_path}")
print(f"# colonne: {len(df.columns)}")
