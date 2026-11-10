"""
01 - Costruzione del target mensile.

- carica train.csv (segnalazioni grezze)
- deduplica le segnalazioni identiche nello stesso mese
- conta le segnalazioni per (Strada, mese) -> num_richieste
- completa la griglia: per ogni strada, i mesi mancanti tra primo e ultimo
  evento osservato valgono 0 (mese visto ma senza segnalazioni)

Output: output/target_mensile.csv  (Strada, mese_data, num_richieste)
"""
import pandas as pd
import config as C

print("Carico train.csv...")
train = pd.read_csv(C.TRAIN_CSV, parse_dates=["Data Ins", "mese_data"], low_memory=False)
print(f"  righe grezze: {len(train):,}")

# righe senza Strada non servono (il test chiede solo strade specifiche)
train = train.dropna(subset=["Strada"]).copy()
for c in C.DEDUP_KEYS:
    train[c] = train[c].fillna("__NA__")

# due segnalazioni identiche nello stesso mese = stesso evento -> dedup
before = len(train)
train = train.drop_duplicates(subset=C.DEDUP_KEYS + ["mese_data"], keep="first")
print(f"  dopo dedup  : {len(train):,} (-{before - len(train):,})")

# conteggio mensile per strada
target = (train.groupby(["Strada", "mese_data"]).size()
          .reset_index(name="num_richieste"))
print(f"  coppie (strada, mese) osservate: {len(target):,}")

# completa la griglia per strada tra primo e ultimo evento (mesi mancanti = 0)
def expand(g):
    months = pd.date_range(g["mese_data"].min(), g["mese_data"].max(), freq="MS")
    return pd.DataFrame({"mese_data": months})

grid = (target.groupby("Strada", group_keys=True)
        .apply(expand, include_groups=False)
        .reset_index()[["Strada", "mese_data"]])
target_full = grid.merge(target, on=["Strada", "mese_data"], how="left")
target_full["num_richieste"] = target_full["num_richieste"].fillna(0).astype(int)

out = C.OUT / "target_mensile.csv"
target_full.to_csv(out, index=False)
print(f"\n-> {out}")
print(f"   righe: {len(target_full):,} | strade: {target_full['Strada'].nunique()} "
      f"| zeri: {(target_full['num_richieste'] == 0).sum():,}")
