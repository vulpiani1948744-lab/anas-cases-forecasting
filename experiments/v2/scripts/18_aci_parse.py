"""
18 - Parsa i file .ods ACI usando pyexcel (più veloce di pandas-odf).
Estrae lo sheet "TabMesi_<anno>" con incidenti MENSILI per (strada, provincia).

Output:
- v2/data_esterni/aci_incidenti_mensile.csv
"""

from pathlib import Path
import pandas as pd
import pyexcel as pe
import time

REPO_DIR = Path(__file__).resolve().parent.parent.parent
ESTERNI = REPO_DIR / "v2" / "data_esterni"
RAW = ESTERNI / "aci_raw"

files = {
    2018: "Localizzazione_Incidenti_stradali_2018.ods",
    2019: "Localizzazione_incidenti_stradali_2019.ods",
    2020: "Localizzazione_Incidenti_stradali_2020.ods",
    2021: "Localizzazione_Incidenti_stradali_2021.ods",
    2022: "Localizzazione_Incidenti_stradali_2022.ods",
    2023: "Localizzazione_incidenti_stradali-2023.ods",
    2024: "Localizzazione_incidenti_stradali-2024.ods",
}

MONTHS_IT = {
    "Gennaio": 1, "Febbraio": 2, "Marzo": 3, "Aprile": 4,
    "Maggio": 5, "Giugno": 6, "Luglio": 7, "Agosto": 8,
    "Settembre": 9, "Ottobre": 10, "Novembre": 11, "Dicembre": 12,
}

all_long = []
for year, fname in files.items():
    path = RAW / fname
    if not path.exists():
        print(f"SKIP {year}: file manca")
        continue
    sheet_name = f"TabMesi_{year}"
    t0 = time.time()
    book = pe.get_book_dict(file_name=str(path))
    sheet = book.get(sheet_name)
    if not sheet:
        print(f"WARN {year}: sheet '{sheet_name}' non trovato. Sheets: {list(book.keys())}")
        continue
    header = [str(c).strip() for c in sheet[0]]
    rows = sheet[1:]
    df = pd.DataFrame(rows, columns=header)
    print(f"{year}: {df.shape}, lettura {time.time()-t0:.1f}s")

    month_cols = {c: MONTHS_IT[c] for c in df.columns if c in MONTHS_IT}
    if not month_cols:
        print(f"  WARN: nessuna colonna mese trovata in {list(df.columns)[:15]}")
        continue

    key_cols = [c for c in ["cReg", "REGIONE", "cProv", "PROVINCIA", "CODICE", "NOME STRADA"]
                if c in df.columns]
    df = df[key_cols + list(month_cols.keys())].copy()

    long = df.melt(id_vars=key_cols, value_vars=list(month_cols.keys()),
                   var_name="mese_nome", value_name="n_incidenti")
    long["mese"] = long["mese_nome"].map(month_cols)
    long["anno"] = year
    long["mese_data"] = pd.to_datetime(dict(year=long["anno"], month=long["mese"], day=1))
    long["n_incidenti"] = pd.to_numeric(long["n_incidenti"], errors="coerce").fillna(0).astype(int)
    long = long.drop(columns=["mese_nome", "mese", "anno"])
    all_long.append(long)

if not all_long:
    print("ERRORE: nessun anno processato")
    raise SystemExit(1)

aci = pd.concat(all_long, ignore_index=True)
print(f"\nTotale righe: {len(aci)}")
print(f"Anni: {sorted(aci['mese_data'].dt.year.unique())}")
print(f"Codici unici: {aci['CODICE'].nunique()}")
print(f"\nTop 10 codici per #incidenti totali:")
print(aci.groupby("CODICE")["n_incidenti"].sum().sort_values(ascending=False).head(10))

out = ESTERNI / "aci_incidenti_mensile.csv"
aci.to_csv(out, index=False)
print(f"\n→ salvato {out} ({out.stat().st_size/1024:.0f} KB)")
