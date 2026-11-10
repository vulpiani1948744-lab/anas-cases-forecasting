"""
12 - Costruisce dataset_v2_meteo unendo dataset_arricchito (v1) + provincia + meteo.

Input:
- output/dataset_arricchito.csv (v1)
- v2/output/strada_provincia.csv
- v2/data_esterni/meteo_mensile.csv

Output:
- v2/output/dataset_v2_meteo.csv
"""

# %% imports
from pathlib import Path
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"
ESTERNI = REPO_DIR / "v2" / "data_esterni"
V1_OUT = REPO_DIR / "output"

# %% carica
df = pd.read_csv(V1_OUT / "dataset_arricchito.csv", parse_dates=["mese_data"])
mapping = pd.read_csv(V2_OUT / "strada_provincia.csv")[["Strada", "Provincia", "regione"]]
meteo = pd.read_csv(ESTERNI / "meteo_mensile.csv", parse_dates=["mese_data"])

print(f"dataset_arricchito: {df.shape}")
print(f"mapping strada-prov: {mapping.shape}")
print(f"meteo mensile     : {meteo.shape} ({meteo['Provincia'].nunique()} province)")

# %% merge
df = df.merge(mapping, on="Strada", how="left")
print(f"\nRighe con provincia: {df['Provincia'].notna().sum() / len(df) * 100:.1f}%")

df = df.merge(meteo, on=["Provincia", "mese_data"], how="left")
meteo_cov = df["temp_mean_mese"].notna().mean() * 100
print(f"Righe con meteo: {meteo_cov:.1f}%")

# %% salva
out_path = V2_OUT / "dataset_v2_meteo.csv"
df.to_csv(out_path, index=False)
print(f"\n→ salvato {out_path}")
print(f"# colonne: {len(df.columns)}")
print(f"# righe: {len(df)} (test rows: {(df['split']=='test').sum()})")
