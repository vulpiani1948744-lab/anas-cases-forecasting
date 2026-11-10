"""
11b - Costruisce i CSV master meteo dalla cache JSON di OpenMeteo.

Output:
- v2/data_esterni/meteo_giornaliero.csv : meteo giornaliero per (Provincia, data)
- v2/data_esterni/meteo_mensile.csv     : meteo aggregato per (Provincia, mese)
"""

from pathlib import Path
import pandas as pd
import json

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_DIR = REPO_DIR / "v2"
CACHE_DIR = V2_DIR / "cache_meteo"
ESTERNI = V2_DIR / "data_esterni"
ESTERNI.mkdir(exist_ok=True)

# %% leggi tutti i JSON in cache
all_dfs = []
for cache_file in sorted(CACHE_DIR.glob("*.json")):
    sigla = cache_file.stem
    data = json.loads(cache_file.read_text())
    if "daily" not in data:
        print(f"WARN {sigla}: skip (no daily)")
        continue
    d = data["daily"]
    df = pd.DataFrame({
        "data": pd.to_datetime(d["time"]),
        "temp_mean": d["temperature_2m_mean"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precip": d["precipitation_sum"],
        "snow": d["snowfall_sum"],
        "wind_max": d["windspeed_10m_max"],
    })
    df["Provincia"] = sigla
    all_dfs.append(df)

meteo = pd.concat(all_dfs, ignore_index=True)
print(f"Meteo giornaliero: {len(meteo)} righe ({meteo['Provincia'].nunique()} province × {meteo['data'].nunique()} giorni)")

# salva master giornaliero
meteo.to_csv(ESTERNI / "meteo_giornaliero.csv", index=False)
print(f"→ {ESTERNI / 'meteo_giornaliero.csv'}")

# %% aggregazione mensile
meteo["mese_data"] = meteo["data"].dt.to_period("M").dt.to_timestamp()
mens = meteo.groupby(["Provincia", "mese_data"]).agg(
    temp_mean_mese=("temp_mean", "mean"),
    temp_max_mese=("temp_max", "max"),
    temp_min_mese=("temp_min", "min"),
    precip_sum_mese=("precip", "sum"),
    n_giorni_pioggia=("precip", lambda s: (s > 1.0).sum()),
    snow_sum_mese=("snow", "sum"),
    n_giorni_neve=("snow", lambda s: (s > 0.5).sum()),
    wind_max_mese=("wind_max", "max"),
    n_giorni_vento_forte=("wind_max", lambda s: (s > 50).sum()),
).reset_index()
mens["temp_amplitude"] = mens["temp_max_mese"] - mens["temp_min_mese"]

mens.to_csv(ESTERNI / "meteo_mensile.csv", index=False)
print(f"→ {ESTERNI / 'meteo_mensile.csv'} ({len(mens)} righe)")
print(mens.head(3))
