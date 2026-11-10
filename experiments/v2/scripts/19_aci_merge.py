"""
19 - Merge degli incidenti ACI con il dataset SUS9.

Strategia:
- Codice ACI "SS01401" → ("SS", 14) → codice unificato "SS14"
- Codice ACI "AA00401" → ("AU", 4) → codice unificato "AU4"
- Strada SUS9 "SS 14 DELLA VENEZIA GIULIA" → estrai numero con regex → "SS14"
- Strada SUS9 "AU A04 ..." → "AU4"
- Aggrega ACI per (codice_unificato, mese_data)
- Merge con dataset_v5 → feature n_incidenti_mese

Output:
- v2/output/dataset_v6_aci.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"
ESTERNI = REPO_DIR / "v2" / "data_esterni"

# %% carica ACI
aci = pd.read_csv(ESTERNI / "aci_incidenti_mensile.csv", parse_dates=["mese_data"])
print(f"ACI righe: {len(aci)}, codici unici: {aci['CODICE'].nunique()}")

# %% normalizza codice ACI → codice unificato
def normalize_aci(codice):
    if pd.isna(codice):
        return None
    s = str(codice).strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", s)
    if not m:
        return None
    prefix, nums = m.group(1), m.group(2)
    # primi 3 digit dopo le lettere = numero strada (gli ultimi 2 sono tratta)
    if len(nums) >= 5:
        strada_num = int(nums[:-2])
    else:
        strada_num = int(nums)
    # AA = autostrada → "AU<n>"
    if prefix == "AA":
        return f"AU{strada_num}"
    # SS, RA, NSA: mantieni il prefix
    return f"{prefix}{strada_num}"

aci["cod_unif"] = aci["CODICE"].apply(normalize_aci)
print(f"Codici unificati ACI unici: {aci['cod_unif'].nunique()}")
print("Esempio mapping:")
print(aci[["CODICE", "cod_unif"]].drop_duplicates().head(10).to_string(index=False))

# aggrega per (cod_unif, mese_data) — somma incidenti di tutte le tratte
aci_agg = aci.dropna(subset=["cod_unif"]).groupby(["cod_unif", "mese_data"])["n_incidenti"].sum().reset_index()
aci_agg = aci_agg.rename(columns={"n_incidenti": "n_incidenti_mese"})
print(f"\nACI aggregato: {len(aci_agg)} righe")

# %% carica dataset_v5
df = pd.read_csv(V2_OUT / "dataset_v5.csv", parse_dates=["mese_data"])
print(f"\nDataset v5: {df.shape}")

# %% normalizza Strada SUS9 → cod_unif
def normalize_strada(strada):
    if pd.isna(strada):
        return None
    s = str(strada).upper()
    # cerca pattern "SS <num>" o "AU A<num>" o "RA <num>" o "NSA <num>"
    m = re.search(r"\b(SS|RA|NSA)\s*(\d+)", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    # autostrada: "AU A04 ..." o "AU A 04 ..."
    m = re.search(r"\bAU\s+A\s*(\d+)\b", s)
    if m:
        return f"AU{int(m.group(1))}"
    # autostrada con nome: "AU A CATANIA - SIRACUSA" → non c'è numero, no match
    return None

df["cod_unif"] = df["Strada"].apply(normalize_strada)
print(f"\nStrade SUS9 con cod_unif: {df['cod_unif'].notna().sum() / len(df) * 100:.1f}%")
print(f"Strade SUS9 uniche con codice : {df.dropna(subset=['cod_unif'])['Strada'].nunique()}")
print(f"Esempi mapping SUS9:")
print(df[["Strada", "cod_unif"]].dropna(subset=["cod_unif"]).drop_duplicates("Strada").head(10).to_string(index=False))

# %% merge
df = df.merge(aci_agg, on=["cod_unif", "mese_data"], how="left")
df["n_incidenti_mese"] = df["n_incidenti_mese"].fillna(0)
print(f"\nRighe con incidenti > 0: {(df['n_incidenti_mese'] > 0).sum()} / {len(df)} = {(df['n_incidenti_mese'] > 0).mean()*100:.1f}%")

# %% feature ulteriori: rolling 3 mesi e ultimo anno (no leakage: shift di 1)
df = df.sort_values(["Strada", "mese_data"])
df["n_incidenti_lag1"] = df.groupby("Strada")["n_incidenti_mese"].shift(1).fillna(0)
df["n_incidenti_roll3"] = df.groupby("Strada")["n_incidenti_mese"].shift(1).rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
df["n_incidenti_roll12"] = df.groupby("Strada")["n_incidenti_mese"].shift(1).rolling(12, min_periods=3).mean().reset_index(0, drop=True).fillna(0)

# %% salva
out = V2_OUT / "dataset_v6_aci.csv"
df = df.drop(columns=["cod_unif"])
df.to_csv(out, index=False)
print(f"\n→ salvato {out}")
print(f"# colonne: {len(df.columns)}")
print(f"# righe: {len(df)}")
