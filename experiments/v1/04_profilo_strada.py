"""
04 - Profilo storico per strada.

Calcolato SOLO sul train (eventi 2016-01 → 2022-12) per evitare leakage sulla validation.
Per le strade non presenti nel train (138 strade nel test), useremo la media globale come fallback.

Feature costruite per ogni Strada:
- pct_notte               : % eventi tra 22:00-06:00
- pct_weekend             : % eventi nel weekend
- pct_canale_*            : % per top-3 canali di arrivo (E-Mail, Telefono, Sito Web)
- pct_categoria_*         : % per top-5 categorie
- pct_tipologia_*         : % per top-3 tipologie
- pct_persona_cliente     : % di richieste da clienti
- media_storica_mensile   : media num_richieste/mese (post-dedup)
- std_storica_mensile     : std num_richieste/mese
- mesi_attivi             : numero di mesi con almeno 1 evento
- pct_mesi_attivi         : su totale mesi del periodo di osservazione

Output:
- output/profilo_strada.csv
"""

# %% imports
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"

# %% carica eventi train deduplicati
train_raw = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Data Ins", "mese_data"], low_memory=False)

# stessa logica di dedup di 01_eda_target.py
dedup_keys = [
    "Localita", "Provincia", "Canale arrivo", "Tipologia",
    "Categoria", "Gruppo", "Strada", "Tipo Persona",
]
train_clean = train_raw.dropna(subset=["Strada"]).copy()
for c in dedup_keys:
    train_clean[c] = train_clean[c].fillna("__NA__")
train_dedup = train_clean.drop_duplicates(subset=dedup_keys + ["mese_data"], keep="first")

# limita al periodo train per evitare leakage (val = 2023)
events = train_dedup[train_dedup["mese_data"] < "2023-01-01"].copy()
print(f"Eventi train (post-dedup, pre-2023): {len(events)}")

# %% parse ora dell'evento
def parse_ora(s):
    if pd.isna(s):
        return np.nan
    try:
        return int(str(s).split(":")[0])
    except Exception:
        return np.nan

events["ora_h"] = events["Ora Ins"].apply(parse_ora)
events["is_notte"] = ((events["ora_h"] >= 22) | (events["ora_h"] < 6)).astype(int)
events["dow"] = events["Data Ins"].dt.dayofweek
events["is_weekend"] = (events["dow"] >= 5).astype(int)
events["is_cliente"] = (events["Tipo Persona"] == "CLIENTI").astype(int)

# %% identifica top categorie/canali/tipologie globali per costruire feature consistenti
top_canali = events["Canale arrivo"].value_counts().head(3).index.tolist()
top_categorie = events["Categoria"].value_counts().head(5).index.tolist()
top_tipologie = events["Tipologia"].value_counts().head(3).index.tolist()

print(f"\nTop canali   : {top_canali}")
print(f"Top categorie: {top_categorie}")
print(f"Top tipologie: {top_tipologie}")

# %% costruzione profilo per Strada
print("\nCostruzione profilo per Strada...")
def profilo_strada(g):
    n = len(g)
    out = {
        "n_eventi_storici": n,
        "pct_notte": g["is_notte"].mean(),
        "pct_weekend": g["is_weekend"].mean(),
        "pct_persona_cliente": g["is_cliente"].mean(),
    }
    for canale in top_canali:
        slug = canale.lower().replace("-", "").replace(" ", "_")[:15]
        out[f"pct_canale_{slug}"] = (g["Canale arrivo"] == canale).mean()
    for cat in top_categorie:
        slug = cat.lower().replace(" ", "_")[:20]
        out[f"pct_cat_{slug}"] = (g["Categoria"] == cat).mean()
    for tip in top_tipologie:
        slug = tip.lower()[:12]
        out[f"pct_tip_{slug}"] = (g["Tipologia"] == tip).mean()
    return pd.Series(out)

profilo = events.groupby("Strada").apply(profilo_strada, include_groups=False).reset_index()
print(f"Strade nel profilo: {len(profilo)}")

# %% statistiche mensili storiche (su target deduplicato pre-2023)
mensile = events.groupby(["Strada", "mese_data"]).size().reset_index(name="n")
stats_mensili = mensile.groupby("Strada").agg(
    media_storica_mensile=("n", "mean"),
    std_storica_mensile=("n", "std"),
    mesi_attivi=("n", "count"),
).reset_index()

# pct_mesi_attivi: mesi attivi / mesi tra primo e ultimo evento
range_mesi = (
    mensile.groupby("Strada")["mese_data"]
    .agg(lambda s: (s.max().to_period("M") - s.min().to_period("M")).n + 1)
    .reset_index(name="mesi_range")
)
stats_mensili = stats_mensili.merge(range_mesi, on="Strada")
stats_mensili["pct_mesi_attivi"] = stats_mensili["mesi_attivi"] / stats_mensili["mesi_range"]
stats_mensili["std_storica_mensile"] = stats_mensili["std_storica_mensile"].fillna(0)

profilo = profilo.merge(stats_mensili, on="Strada", how="left")

print(f"\nColonne profilo: {len(profilo.columns)}")
print(profilo.columns.tolist())
print(profilo.head(3).round(2))

# %% salva
profilo.to_csv(OUT_DIR / "profilo_strada.csv", index=False)
print(f"\n→ salvato {OUT_DIR / 'profilo_strada.csv'}")
