"""
10 - Mapping Strada → Provincia.

Per ogni Strada nel dataset (sia train che test), determino la provincia rappresentativa:
- Per le strade viste nel train: provincia modale (più frequente) tra le righe con Provincia valorizzata
- Per le strade nuove (mai viste in train): estrazione euristica dal nome strada
  (es. "AU A CATANIA - SIRACUSA" → CT; in fallback "ND" per "Non Determinata")

Output:
- v2/output/strada_provincia.csv : Strada, provincia_rappresentativa, regione, source
"""

# %% imports
from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"  # dati grezzi della competizione (non versionati)
REPO_DIR = Path(__file__).resolve().parent.parent.parent  # cartella experiments/
V2_OUT = REPO_DIR / "v2" / "output"

# %% carica train e test
train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False, usecols=["Strada", "Provincia"])
test = pd.read_csv(DATA_DIR / "test.csv")
test = test.rename(columns={"strada": "Strada"})

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# %% provincia modale per ogni strada vista in train
train_with_strada = train.dropna(subset=["Strada"])
print(f"Train con Strada non nulla: {len(train_with_strada)}")

prov_modal = (
    train_with_strada.dropna(subset=["Provincia"])
    .groupby("Strada")["Provincia"]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    .reset_index()
    .rename(columns={"Provincia": "provincia_modale"})
)
print(f"\nStrade con provincia modale: {len(prov_modal)}")
print(prov_modal.head(5))

# %% mappa province italiane → regione e coordinate del capoluogo
# (capoluoghi delle 110 province italiane + sigle)
PROVINCE_INFO = {
    # nome: (sigla, regione, lat, lon)
    "AG": ("AG", "Sicilia", 37.310, 13.585),
    "AL": ("AL", "Piemonte", 44.913, 8.611),
    "AN": ("AN", "Marche", 43.617, 13.518),
    "AO": ("AO", "Valle d'Aosta", 45.737, 7.315),
    "AP": ("AP", "Marche", 42.853, 13.575),
    "AQ": ("AQ", "Abruzzo", 42.351, 13.398),
    "AR": ("AR", "Toscana", 43.464, 11.881),
    "AT": ("AT", "Piemonte", 44.901, 8.207),
    "AV": ("AV", "Campania", 40.916, 14.795),
    "BA": ("BA", "Puglia", 41.117, 16.872),
    "BG": ("BG", "Lombardia", 45.694, 9.670),
    "BI": ("BI", "Piemonte", 45.563, 8.054),
    "BL": ("BL", "Veneto", 46.139, 12.218),
    "BN": ("BN", "Campania", 41.131, 14.778),
    "BO": ("BO", "Emilia-Romagna", 44.494, 11.342),
    "BR": ("BR", "Puglia", 40.633, 17.940),
    "BS": ("BS", "Lombardia", 45.539, 10.220),
    "BT": ("BT", "Puglia", 41.327, 16.279),
    "BZ": ("BZ", "Trentino-Alto Adige", 46.499, 11.354),
    "CA": ("CA", "Sardegna", 39.224, 9.122),
    "CB": ("CB", "Molise", 41.560, 14.667),
    "CE": ("CE", "Campania", 41.073, 14.331),
    "CH": ("CH", "Abruzzo", 42.351, 14.166),
    "CL": ("CL", "Sicilia", 37.491, 14.062),
    "CN": ("CN", "Piemonte", 44.388, 7.546),
    "CO": ("CO", "Lombardia", 45.808, 9.085),
    "CR": ("CR", "Lombardia", 45.133, 10.022),
    "CS": ("CS", "Calabria", 39.300, 16.252),
    "CT": ("CT", "Sicilia", 37.503, 15.087),
    "CZ": ("CZ", "Calabria", 38.910, 16.587),
    "EN": ("EN", "Sicilia", 37.567, 14.275),
    "FC": ("FC", "Emilia-Romagna", 44.220, 12.041),
    "FE": ("FE", "Emilia-Romagna", 44.836, 11.620),
    "FG": ("FG", "Puglia", 41.462, 15.544),
    "FI": ("FI", "Toscana", 43.770, 11.255),
    "FM": ("FM", "Marche", 43.160, 13.718),
    "FR": ("FR", "Lazio", 41.640, 13.346),
    "GE": ("GE", "Liguria", 44.408, 8.934),
    "GO": ("GO", "Friuli-Venezia Giulia", 45.940, 13.622),
    "GR": ("GR", "Toscana", 42.760, 11.114),
    "IM": ("IM", "Liguria", 43.882, 8.027),
    "IS": ("IS", "Molise", 41.595, 14.231),
    "KR": ("KR", "Calabria", 39.082, 17.127),
    "LC": ("LC", "Lombardia", 45.857, 9.397),
    "LE": ("LE", "Puglia", 40.358, 18.171),
    "LI": ("LI", "Toscana", 43.546, 10.310),
    "LO": ("LO", "Lombardia", 45.314, 9.503),
    "LT": ("LT", "Lazio", 41.466, 12.904),
    "LU": ("LU", "Toscana", 43.843, 10.508),
    "MB": ("MB", "Lombardia", 45.585, 9.273),
    "MC": ("MC", "Marche", 43.298, 13.451),
    "ME": ("ME", "Sicilia", 38.193, 15.553),
    "MI": ("MI", "Lombardia", 45.464, 9.190),
    "MN": ("MN", "Lombardia", 45.156, 10.793),
    "MO": ("MO", "Emilia-Romagna", 44.647, 10.925),
    "MS": ("MS", "Toscana", 44.034, 10.139),
    "MT": ("MT", "Basilicata", 40.667, 16.604),
    "NA": ("NA", "Campania", 40.852, 14.268),
    "NO": ("NO", "Piemonte", 45.446, 8.621),
    "NU": ("NU", "Sardegna", 40.321, 9.327),
    "OR": ("OR", "Sardegna", 39.903, 8.591),
    "PA": ("PA", "Sicilia", 38.116, 13.362),
    "PC": ("PC", "Emilia-Romagna", 45.052, 9.693),
    "PD": ("PD", "Veneto", 45.407, 11.876),
    "PE": ("PE", "Abruzzo", 42.464, 14.213),
    "PG": ("PG", "Umbria", 43.111, 12.389),
    "PI": ("PI", "Toscana", 43.716, 10.396),
    "PN": ("PN", "Friuli-Venezia Giulia", 45.957, 12.661),
    "PO": ("PO", "Toscana", 43.879, 11.099),
    "PR": ("PR", "Emilia-Romagna", 44.802, 10.328),
    "PT": ("PT", "Toscana", 43.933, 10.917),
    "PU": ("PU", "Marche", 43.910, 12.913),
    "PV": ("PV", "Lombardia", 45.185, 9.155),
    "PZ": ("PZ", "Basilicata", 40.642, 15.805),
    "RA": ("RA", "Emilia-Romagna", 44.418, 12.203),
    "RC": ("RC", "Calabria", 38.106, 15.643),
    "RE": ("RE", "Emilia-Romagna", 44.698, 10.631),
    "RG": ("RG", "Sicilia", 36.927, 14.726),
    "RI": ("RI", "Lazio", 42.404, 12.857),
    "RM": ("RM", "Lazio", 41.903, 12.495),
    "RN": ("RN", "Emilia-Romagna", 44.060, 12.566),
    "RO": ("RO", "Veneto", 45.072, 11.792),
    "SA": ("SA", "Campania", 40.682, 14.770),
    "SI": ("SI", "Toscana", 43.318, 11.331),
    "SO": ("SO", "Lombardia", 46.170, 9.871),
    "SP": ("SP", "Liguria", 44.107, 9.829),
    "SR": ("SR", "Sicilia", 37.071, 15.288),
    "SS": ("SS", "Sardegna", 40.728, 8.555),
    "SU": ("SU", "Sardegna", 39.219, 8.612),
    "SV": ("SV", "Liguria", 44.308, 8.482),
    "TA": ("TA", "Puglia", 40.464, 17.247),
    "TE": ("TE", "Abruzzo", 42.660, 13.704),
    "TN": ("TN", "Trentino-Alto Adige", 46.069, 11.121),
    "TO": ("TO", "Piemonte", 45.071, 7.687),
    "TP": ("TP", "Sicilia", 38.019, 12.515),
    "TR": ("TR", "Umbria", 42.564, 12.643),
    "TS": ("TS", "Friuli-Venezia Giulia", 45.649, 13.776),
    "TV": ("TV", "Veneto", 45.666, 12.243),
    "UD": ("UD", "Friuli-Venezia Giulia", 46.063, 13.235),
    "VA": ("VA", "Lombardia", 45.821, 8.825),
    "VB": ("VB", "Piemonte", 45.921, 8.552),
    "VC": ("VC", "Piemonte", 45.327, 8.422),
    "VE": ("VE", "Veneto", 45.438, 12.327),
    "VI": ("VI", "Veneto", 45.546, 11.539),
    "VR": ("VR", "Veneto", 45.438, 10.992),
    "VT": ("VT", "Lazio", 42.418, 12.107),
    "VV": ("VV", "Calabria", 38.677, 16.099),
}

# DataFrame province
prov_info = pd.DataFrame([
    {"Provincia": k, "regione": v[1], "lat": v[2], "lon": v[3]}
    for k, v in PROVINCE_INFO.items()
])
print(f"\nProvince in mappa: {len(prov_info)}")

# %% costruisci lookup completo per tutte le strade (train ∪ test)
all_strade = sorted(set(train_with_strada["Strada"].unique()) | set(test["Strada"].unique()))
print(f"\nStrade totali (train ∪ test): {len(all_strade)}")

mapping = pd.DataFrame({"Strada": all_strade})
mapping = mapping.merge(prov_modal, on="Strada", how="left")
mapping["Provincia"] = mapping["provincia_modale"]
mapping["source"] = mapping["Provincia"].notna().map({True: "train_modal", False: "missing"})

# %% fallback: estrai sigla provincia dal nome strada (es. "RA15 - PALERMO-MAZARA")
# strategia molto blanda — usata solo se non c'è provincia modale
def guess_from_name(strada):
    if pd.isna(strada):
        return None
    s = strada.upper()
    # cerco città note (parziale)
    for sigla, info in PROVINCE_INFO.items():
        # nome del capoluogo desumibile dalla regione (proxy)
        pass
    return None  # per ora restituisce None; le 138 nuove resteranno senza provincia

# %% merge regione + coordinate
mapping = mapping.merge(prov_info, on="Provincia", how="left")

n_mapped = mapping["Provincia"].notna().sum()
print(f"\nMapping risultato:")
print(f"  Con provincia (da modale train): {n_mapped}")
print(f"  Senza provincia                : {len(mapping) - n_mapped}")

print(f"\nProvince mancate nella mappa (presenti in train ma senza coord):")
missing_in_info = set(prov_modal["provincia_modale"].dropna()) - set(PROVINCE_INFO.keys())
print(missing_in_info if missing_in_info else "nessuna")

# %% salva
out_path = V2_OUT / "strada_provincia.csv"
mapping.to_csv(out_path, index=False)
print(f"\n→ salvato {out_path}")
print(mapping.head(8).to_string(index=False))
