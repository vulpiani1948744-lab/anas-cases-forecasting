"""
11 - Scarica meteo storico da OpenMeteo per ogni provincia (107 chiamate).

API: https://archive-api.open-meteo.com/v1/archive (gratis, no auth)
Periodo: 2016-01-01 → 2025-12-31
Variabili giornaliere: temp media/min/max, precipitazioni, neve, vento

Salva in v2/cache_meteo/<sigla>.json per non riscaricare.
Output combinato: v2/output/meteo_giornaliero.csv (lungo: provincia × data)
"""

# %% imports
from pathlib import Path
import pandas as pd
import urllib.request
import urllib.parse
import json
import time

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_DIR = REPO_DIR / "v2"
V2_OUT = V2_DIR / "output"
CACHE_DIR = V2_DIR / "cache_meteo"

# %% carica mapping
mapping = pd.read_csv(V2_OUT / "strada_provincia.csv")
province_uniche = mapping[["Provincia", "regione", "lat", "lon"]].dropna(subset=["Provincia"]).drop_duplicates()
print(f"Province uniche da scaricare: {len(province_uniche)}")

# %% parametri API
BASE = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]

def fetch(lat, lon):
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Rome",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())

# %% loop con cache
all_dfs = []
for i, row in enumerate(province_uniche.itertuples(index=False), 1):
    sigla = row.Provincia
    cache_file = CACHE_DIR / f"{sigla}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        # retry con backoff su 429
        data = None
        wait = 2.0
        for attempt in range(6):
            try:
                print(f"  [{i}/{len(province_uniche)}] scarico {sigla} (att.{attempt+1})...", end=" ")
                data = fetch(row.lat, row.lon)
                cache_file.write_text(json.dumps(data))
                print("ok")
                time.sleep(1.2)  # delay base più conservativo
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"429, sleep {wait:.0f}s")
                    time.sleep(wait)
                    wait *= 2
                else:
                    print(f"HTTP {e.code}")
                    break
            except Exception as e:
                print(f"ERR: {e}")
                time.sleep(wait)
                wait *= 2
        if data is None:
            continue

    if "daily" not in data:
        print(f"  WARN {sigla}: risposta senza 'daily': {list(data.keys())}")
        continue

    df = pd.DataFrame({
        "data": pd.to_datetime(data["daily"]["time"]),
        "temp_mean": data["daily"]["temperature_2m_mean"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precip": data["daily"]["precipitation_sum"],
        "snow": data["daily"]["snowfall_sum"],
        "wind_max": data["daily"]["windspeed_10m_max"],
    })
    df["Provincia"] = sigla
    all_dfs.append(df)

meteo = pd.concat(all_dfs, ignore_index=True)
print(f"\nMeteo giornaliero totale: {len(meteo)} righe ({meteo['Provincia'].nunique()} province × {meteo['data'].nunique()} giorni)")
print(meteo.head(3))

# %% salva
out_path = V2_OUT / "meteo_giornaliero.csv"
meteo.to_csv(out_path, index=False)
print(f"\n→ salvato {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")
