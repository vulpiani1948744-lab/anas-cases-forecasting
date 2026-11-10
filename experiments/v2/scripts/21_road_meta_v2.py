"""
21 - Integra il nuovo road_metadata con totali annuali 2024/25 per strada.

Aggiunge feature:
- tot_2024_strada : totale annuo richieste 2024 per la strada
- tot_2025_strada : totale annuo richieste 2025 per la strada
- tot_anno_corrente : tot_2024 se anno==2024 else tot_2025 (per il modello)
- mix per tipologia 2024/25 (informativi, ridotti a top 5 per evitare blow-up)

Output:
- v2/output/dataset_v7_totali.csv
- v2/output/tot_annuali_per_strada.csv (per il rescaling in fase predittiva)
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"

# %% carica nuovo road_metadata
rm = pd.read_csv(DATA_DIR / "road_metadata.csv")
print(f"road_metadata nuovo: {rm.shape}")
rm = rm.rename(columns={"strada": "Strada"})

# forza conversione numerica delle colonne statiche
static_cols = ["n_tratte", "comuni_attraversati", "province_attraversate",
               "km_min", "km_max", "consistenza_gg_media", "leggeri_media", "pesanti_media"]
for c in static_cols:
    rm[c] = pd.to_numeric(rm[c], errors="coerce")

# %% somme annuali
cols_2024 = [c for c in rm.columns if c.startswith("2024_")]
cols_2025 = [c for c in rm.columns if c.startswith("2025_")]
print(f"# cols 2024: {len(cols_2024)}, # cols 2025: {len(cols_2025)}")

rm["tot_2024_strada"] = rm[cols_2024].sum(axis=1)
rm["tot_2025_strada"] = rm[cols_2025].sum(axis=1)

# delta YoY
rm["delta_2425"] = rm["tot_2025_strada"] - rm["tot_2024_strada"]
rm["growth_2425"] = (rm["tot_2025_strada"] + 1) / (rm["tot_2024_strada"] + 1)

# mix tipologie 2024 (% del totale) per top 5 categorie
top5_2024 = ["2024_tipologia_sala_operativa_nazionale", "2024_tipologia_informazioni",
             "2024_tipologia_segnalazioni", "2024_tipologia_reclami", "2024_tipologia_richiesta"]
for c in top5_2024:
    if c in rm.columns:
        new_name = "pct_" + c.replace("2024_tipologia_", "")
        rm[new_name] = rm[c] / (rm["tot_2024_strada"] + 1)

# salva tabella dei totali per rescaling
totali = rm[["Strada", "tot_2024_strada", "tot_2025_strada"]].copy()
totali.to_csv(V2_OUT / "tot_annuali_per_strada.csv", index=False)
print(f"\nTotali annuali salvati per rescaling")

# %% carica dataset v6 (con ACI) e mergia
df = pd.read_csv(V2_OUT / "dataset_v6_aci.csv", parse_dates=["mese_data"])
print(f"\nDataset v6: {df.shape}")

# togli le vecchie colonne road_metadata che ora vengono dal nuovo file
old_rm_cols = ["n_tratte", "comuni_attraversati", "province_attraversate",
               "km_min", "km_max", "consistenza_gg_media", "leggeri_media", "pesanti_media",
               "lunghezza_km", "quota_pesanti"]
df = df.drop(columns=[c for c in old_rm_cols if c in df.columns])

# nuovo merge
new_cols = ["Strada", "n_tratte", "comuni_attraversati", "province_attraversate",
            "km_min", "km_max", "consistenza_gg_media", "leggeri_media", "pesanti_media",
            "tot_2024_strada", "tot_2025_strada", "delta_2425", "growth_2425"]
new_cols += [c for c in rm.columns if c.startswith("pct_tipologia_")]
df = df.merge(rm[new_cols], on="Strada", how="left")

# ricalcola lunghezza e quota_pesanti
df["lunghezza_km"] = df["km_max"] - df["km_min"]
df["quota_pesanti"] = df["pesanti_media"] / df["consistenza_gg_media"].replace(0, np.nan)

# feature critica: tot_anno_corrente = tot_2024 o tot_2025 a seconda dell'anno
df["tot_anno_corrente"] = np.where(
    df["mese_data"].dt.year == 2024, df["tot_2024_strada"],
    np.where(df["mese_data"].dt.year == 2025, df["tot_2025_strada"], np.nan),
)
# per le righe train (2016-2023), tot_anno_corrente = NaN (LGBM le gestisce)

print(f"\nDataset v7: {df.shape}")
print(f"tot_anno_corrente non-NaN: {df['tot_anno_corrente'].notna().sum()} (atteso ≈ 15048)")
print(f"Stats tot_anno_corrente sul test:")
print(df[df['split']=='test']['tot_anno_corrente'].describe())

# %% salva
out = V2_OUT / "dataset_v7_totali.csv"
df.to_csv(out, index=False)
print(f"\n→ salvato {out}")
print(f"# colonne: {len(df.columns)}")
