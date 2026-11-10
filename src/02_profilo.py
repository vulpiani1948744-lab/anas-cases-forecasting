"""
02 - Profilo storico per strada (leakage-safe, a DUE cutoff).

Caratterizza ogni strada con le sue abitudini storiche. Per restare onesti il
profilo usato per prevedere un periodo deve basarsi SOLO sugli eventi precedenti:
  - profilo_strada_val.csv  : eventi < 2023  -> usato per train (2016-2022) e val 2023
  - profilo_strada_test.csv : eventi < 2024  -> usato per il test 2024-2025
Cosi' la validazione 2023 non vede il 2023, ma la previsione finale del 2024 puo'
sfruttare TUTTO lo storico disponibile fino al 2023 (dato passato, legittimo).

Feature (tra cui la richiesta "giorno o notte" e "tipologia di persona"):
- pct_notte            : % segnalazioni notturne (22:00-06:00)
- pct_weekend          : % segnalazioni nel weekend
- pct_persona_cliente  : % segnalazioni da clienti (tipologia persona)
- pct_canale_*         : mix dei top-3 canali di arrivo
- pct_cat_*            : mix delle top-5 categorie
- pct_tip_*            : mix delle top-3 tipologie
- media/std_storica_mensile, mesi_attivi, pct_mesi_attivi

Output: output/profilo_strada_val.csv, output/profilo_strada_test.csv
"""
import numpy as np
import pandas as pd
import config as C

print("Carico e deduplico gli eventi...")
train = pd.read_csv(C.TRAIN_CSV, parse_dates=["Data Ins", "mese_data"], low_memory=False)
train = train.dropna(subset=["Strada"]).copy()
for c in C.DEDUP_KEYS:
    train[c] = train[c].fillna("__NA__")
train = train.drop_duplicates(subset=C.DEDUP_KEYS + ["mese_data"], keep="first")

# giorno/notte, weekend, tipologia persona
def ora_h(s):
    try:
        return int(str(s).split(":")[0])
    except Exception:
        return np.nan
train["ora_h"] = train["Ora Ins"].apply(ora_h)
train["is_notte"] = ((train["ora_h"] >= 22) | (train["ora_h"] < 6)).astype(int)
train["is_weekend"] = (train["Data Ins"].dt.dayofweek >= 5).astype(int)
train["is_cliente"] = (train["Tipo Persona"] == "CLIENTI").astype(int)


def build_profile(cutoff):
    events = train[train["mese_data"] < cutoff].copy()
    top_canali = events["Canale arrivo"].value_counts().head(3).index.tolist()
    top_cat = events["Categoria"].value_counts().head(5).index.tolist()
    top_tip = events["Tipologia"].value_counts().head(3).index.tolist()

    def profilo(g):
        out = {
            "n_eventi_storici": len(g),
            "pct_notte": g["is_notte"].mean(),
            "pct_weekend": g["is_weekend"].mean(),
            "pct_persona_cliente": g["is_cliente"].mean(),
        }
        for canale in top_canali:
            slug = canale.lower().replace("-", "").replace(" ", "_")[:15]
            out[f"pct_canale_{slug}"] = (g["Canale arrivo"] == canale).mean()
        for cat in top_cat:
            out[f"pct_cat_{cat.lower().replace(' ', '_')[:20]}"] = (g["Categoria"] == cat).mean()
        for tip in top_tip:
            out[f"pct_tip_{tip.lower()[:12]}"] = (g["Tipologia"] == tip).mean()
        return pd.Series(out)

    prof = events.groupby("Strada").apply(profilo, include_groups=False).reset_index()
    mens = events.groupby(["Strada", "mese_data"]).size().reset_index(name="n")
    stats = mens.groupby("Strada").agg(
        media_storica_mensile=("n", "mean"),
        std_storica_mensile=("n", "std"),
        mesi_attivi=("n", "count"),
    ).reset_index()
    rng = (mens.groupby("Strada")["mese_data"]
           .agg(lambda s: (s.max().to_period("M") - s.min().to_period("M")).n + 1)
           .reset_index(name="mesi_range"))
    stats = stats.merge(rng, on="Strada")
    stats["pct_mesi_attivi"] = stats["mesi_attivi"] / stats["mesi_range"]
    stats["std_storica_mensile"] = stats["std_storica_mensile"].fillna(0)
    return prof.merge(stats, on="Strada", how="left")


for nome, cutoff in [("val", C.PROFILE_CUTOFF), ("test", C.TEST_START)]:
    p = build_profile(cutoff)
    out = C.OUT / f"profilo_strada_{nome}.csv"
    p.to_csv(out, index=False)
    print(f"-> {out}  (cutoff <{cutoff}: {p.shape[0]} strade, {p.shape[1]} colonne)")
