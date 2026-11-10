"""
15 - Mappatura provincia migliorata: cerca nomi di capoluoghi/città italiane
dentro il nome strada per le 305 strade senza provincia.

Strategia:
1. Match esatto: nome capoluogo presente nel nome strada (es. "CATANIA" in "AU A CATANIA - SIRACUSA")
2. Fallback per autostrade note (A1, A4, A14, A22, ecc.) → provincia centrale
3. Se ancora ND, lascia ND

Output:
- v2/output/strada_provincia.csv (sovrascrive con versione arricchita)
"""

from pathlib import Path
import pandas as pd
import re

REPO_DIR = Path(__file__).resolve().parent.parent.parent
V2_OUT = REPO_DIR / "v2" / "output"

# %% riusa la mappa province + carica mapping iniziale
PROVINCE_INFO = {
    "AG": ("Agrigento", "Sicilia", 37.310, 13.585),
    "AL": ("Alessandria", "Piemonte", 44.913, 8.611),
    "AN": ("Ancona", "Marche", 43.617, 13.518),
    "AO": ("Aosta", "Valle d'Aosta", 45.737, 7.315),
    "AP": ("Ascoli Piceno", "Marche", 42.853, 13.575),
    "AQ": ("L'Aquila", "Abruzzo", 42.351, 13.398),
    "AR": ("Arezzo", "Toscana", 43.464, 11.881),
    "AT": ("Asti", "Piemonte", 44.901, 8.207),
    "AV": ("Avellino", "Campania", 40.916, 14.795),
    "BA": ("Bari", "Puglia", 41.117, 16.872),
    "BG": ("Bergamo", "Lombardia", 45.694, 9.670),
    "BI": ("Biella", "Piemonte", 45.563, 8.054),
    "BL": ("Belluno", "Veneto", 46.139, 12.218),
    "BN": ("Benevento", "Campania", 41.131, 14.778),
    "BO": ("Bologna", "Emilia-Romagna", 44.494, 11.342),
    "BR": ("Brindisi", "Puglia", 40.633, 17.940),
    "BS": ("Brescia", "Lombardia", 45.539, 10.220),
    "BT": ("Barletta", "Puglia", 41.327, 16.279),
    "BZ": ("Bolzano", "Trentino-Alto Adige", 46.499, 11.354),
    "CA": ("Cagliari", "Sardegna", 39.224, 9.122),
    "CB": ("Campobasso", "Molise", 41.560, 14.667),
    "CE": ("Caserta", "Campania", 41.073, 14.331),
    "CH": ("Chieti", "Abruzzo", 42.351, 14.166),
    "CL": ("Caltanissetta", "Sicilia", 37.491, 14.062),
    "CN": ("Cuneo", "Piemonte", 44.388, 7.546),
    "CO": ("Como", "Lombardia", 45.808, 9.085),
    "CR": ("Cremona", "Lombardia", 45.133, 10.022),
    "CS": ("Cosenza", "Calabria", 39.300, 16.252),
    "CT": ("Catania", "Sicilia", 37.503, 15.087),
    "CZ": ("Catanzaro", "Calabria", 38.910, 16.587),
    "EN": ("Enna", "Sicilia", 37.567, 14.275),
    "FC": ("Forlì", "Emilia-Romagna", 44.220, 12.041),
    "FE": ("Ferrara", "Emilia-Romagna", 44.836, 11.620),
    "FG": ("Foggia", "Puglia", 41.462, 15.544),
    "FI": ("Firenze", "Toscana", 43.770, 11.255),
    "FM": ("Fermo", "Marche", 43.160, 13.718),
    "FR": ("Frosinone", "Lazio", 41.640, 13.346),
    "GE": ("Genova", "Liguria", 44.408, 8.934),
    "GO": ("Gorizia", "Friuli-Venezia Giulia", 45.940, 13.622),
    "GR": ("Grosseto", "Toscana", 42.760, 11.114),
    "IM": ("Imperia", "Liguria", 43.882, 8.027),
    "IS": ("Isernia", "Molise", 41.595, 14.231),
    "KR": ("Crotone", "Calabria", 39.082, 17.127),
    "LC": ("Lecco", "Lombardia", 45.857, 9.397),
    "LE": ("Lecce", "Puglia", 40.358, 18.171),
    "LI": ("Livorno", "Toscana", 43.546, 10.310),
    "LO": ("Lodi", "Lombardia", 45.314, 9.503),
    "LT": ("Latina", "Lazio", 41.466, 12.904),
    "LU": ("Lucca", "Toscana", 43.843, 10.508),
    "MB": ("Monza", "Lombardia", 45.585, 9.273),
    "MC": ("Macerata", "Marche", 43.298, 13.451),
    "ME": ("Messina", "Sicilia", 38.193, 15.553),
    "MI": ("Milano", "Lombardia", 45.464, 9.190),
    "MN": ("Mantova", "Lombardia", 45.156, 10.793),
    "MO": ("Modena", "Emilia-Romagna", 44.647, 10.925),
    "MS": ("Massa", "Toscana", 44.034, 10.139),
    "MT": ("Matera", "Basilicata", 40.667, 16.604),
    "NA": ("Napoli", "Campania", 40.852, 14.268),
    "NO": ("Novara", "Piemonte", 45.446, 8.621),
    "NU": ("Nuoro", "Sardegna", 40.321, 9.327),
    "OR": ("Oristano", "Sardegna", 39.903, 8.591),
    "PA": ("Palermo", "Sicilia", 38.116, 13.362),
    "PC": ("Piacenza", "Emilia-Romagna", 45.052, 9.693),
    "PD": ("Padova", "Veneto", 45.407, 11.876),
    "PE": ("Pescara", "Abruzzo", 42.464, 14.213),
    "PG": ("Perugia", "Umbria", 43.111, 12.389),
    "PI": ("Pisa", "Toscana", 43.716, 10.396),
    "PN": ("Pordenone", "Friuli-Venezia Giulia", 45.957, 12.661),
    "PO": ("Prato", "Toscana", 43.879, 11.099),
    "PR": ("Parma", "Emilia-Romagna", 44.802, 10.328),
    "PT": ("Pistoia", "Toscana", 43.933, 10.917),
    "PU": ("Pesaro", "Marche", 43.910, 12.913),
    "PV": ("Pavia", "Lombardia", 45.185, 9.155),
    "PZ": ("Potenza", "Basilicata", 40.642, 15.805),
    "RA": ("Ravenna", "Emilia-Romagna", 44.418, 12.203),
    "RC": ("Reggio Calabria", "Calabria", 38.106, 15.643),
    "RE": ("Reggio Emilia", "Emilia-Romagna", 44.698, 10.631),
    "RG": ("Ragusa", "Sicilia", 36.927, 14.726),
    "RI": ("Rieti", "Lazio", 42.404, 12.857),
    "RM": ("Roma", "Lazio", 41.903, 12.495),
    "RN": ("Rimini", "Emilia-Romagna", 44.060, 12.566),
    "RO": ("Rovigo", "Veneto", 45.072, 11.792),
    "SA": ("Salerno", "Campania", 40.682, 14.770),
    "SI": ("Siena", "Toscana", 43.318, 11.331),
    "SO": ("Sondrio", "Lombardia", 46.170, 9.871),
    "SP": ("La Spezia", "Liguria", 44.107, 9.829),
    "SR": ("Siracusa", "Sicilia", 37.071, 15.288),
    "SS": ("Sassari", "Sardegna", 40.728, 8.555),
    "SU": ("Sud Sardegna", "Sardegna", 39.219, 8.612),
    "SV": ("Savona", "Liguria", 44.308, 8.482),
    "TA": ("Taranto", "Puglia", 40.464, 17.247),
    "TE": ("Teramo", "Abruzzo", 42.660, 13.704),
    "TN": ("Trento", "Trentino-Alto Adige", 46.069, 11.121),
    "TO": ("Torino", "Piemonte", 45.071, 7.687),
    "TP": ("Trapani", "Sicilia", 38.019, 12.515),
    "TR": ("Terni", "Umbria", 42.564, 12.643),
    "TS": ("Trieste", "Friuli-Venezia Giulia", 45.649, 13.776),
    "TV": ("Treviso", "Veneto", 45.666, 12.243),
    "UD": ("Udine", "Friuli-Venezia Giulia", 46.063, 13.235),
    "VA": ("Varese", "Lombardia", 45.821, 8.825),
    "VB": ("Verbania", "Piemonte", 45.921, 8.552),
    "VC": ("Vercelli", "Piemonte", 45.327, 8.422),
    "VE": ("Venezia", "Veneto", 45.438, 12.327),
    "VI": ("Vicenza", "Veneto", 45.546, 11.539),
    "VR": ("Verona", "Veneto", 45.438, 10.992),
    "VT": ("Viterbo", "Lazio", 42.418, 12.107),
    "VV": ("Vibo Valentia", "Calabria", 38.677, 16.099),
}

# città aggiuntive comuni nei nomi strada → mappate a sigle
EXTRA_CITY = {
    "FIANO ROMANO": "RM", "GUIDONIA": "RM", "MAZARA": "TP",
    "PESCARA": "PE", "ANCONA": "AN", "PORTO GARIBALDI": "FE",
    "MALPENSA": "VA", "LINATE": "MI", "CAPODICHINO": "NA",
    "ORIO AL SERIO": "BG", "FIUMICINO": "RM", "CIAMPINO": "RM",
    "TIRRENO": "GR", "ADRIATICA": "BO", "JONICA": "RC",
    "ROMEA": "VE", "FLAMINIA": "RM", "AURELIA": "RM",
    "EMILIA": "BO", "CASILINA": "RM", "APPIA": "RM",
    "PONTEBBANA": "UD", "VALSESIANA": "VC", "IGLESIENTE": "CA",
    "AMALFITANA": "SA", "VESUVIANA": "NA",
}

# mappa autostrade comuni → provincia "centrale" approssimativa
AUTOSTRADE = {
    "A1": "BO", "A2": "RC", "A4": "MI", "A5": "TO",
    "A6": "CN", "A7": "MI", "A8": "MI", "A9": "MI",
    "A10": "GE", "A11": "FI", "A12": "GE", "A13": "BO",
    "A14": "BO", "A15": "PR", "A16": "AV", "A18": "ME",
    "A19": "PA", "A20": "PA", "A21": "AL", "A22": "MO",
    "A23": "UD", "A24": "AQ", "A25": "AQ", "A26": "NO",
    "A27": "TV", "A28": "PN", "A29": "PA", "A30": "SA",
    "A31": "VI", "A32": "TO", "A33": "AT", "A34": "GO",
    "A35": "BG", "A36": "MI", "A50": "MI", "A51": "MI", "A52": "MI",
    "A53": "PV", "A55": "TO", "A56": "NA", "A57": "VE",
    "A58": "MI", "A90": "RM", "A91": "RM",
}

# %% carica mapping iniziale (esistente)
mapping_old = pd.read_csv(V2_OUT / "strada_provincia.csv")
print(f"Mapping iniziale: {len(mapping_old)} strade, {mapping_old['Provincia'].notna().sum()} con provincia")

# %% per ogni strada senza provincia, cerca match euristico
def guess_provincia(strada):
    if pd.isna(strada):
        return None
    s = strada.upper()

    # 1) cerca città nel testo (capoluoghi)
    for sigla, info in PROVINCE_INFO.items():
        nome_citta = info[0].upper()
        if nome_citta in s:
            return sigla, "city_in_name"

    # 2) cerca città extra
    for citta, sigla in EXTRA_CITY.items():
        if citta in s:
            return sigla, "extra_city"

    # 3) cerca codice autostrada (es. "A1", "A14", ...)
    m = re.search(r"\bA(\d+)\b", s)
    if m:
        key = f"A{m.group(1)}"
        if key in AUTOSTRADE:
            return AUTOSTRADE[key], "autostrada"

    return None, "no_match"

# %% applica solo dove manca
df = mapping_old.copy()
needed = df["Provincia"].isna()
print(f"Strade senza provincia da risolvere: {needed.sum()}")

guesses = df.loc[needed, "Strada"].apply(guess_provincia)
df.loc[needed, "Provincia"] = [g[0] for g in guesses]
df.loc[needed, "source"] = [g[1] for g in guesses]

# riapplica regione+lat+lon dove era NaN
prov_info_df = pd.DataFrame([
    {"Provincia": k, "regione_new": v[1], "lat_new": v[2], "lon_new": v[3]}
    for k, v in PROVINCE_INFO.items()
])
df = df.merge(prov_info_df, on="Provincia", how="left")
df["regione"] = df["regione"].fillna(df["regione_new"])
df["lat"] = df["lat"].fillna(df["lat_new"])
df["lon"] = df["lon"].fillna(df["lon_new"])
df = df.drop(columns=["regione_new", "lat_new", "lon_new"])

n_with_prov = df["Provincia"].notna().sum()
print(f"\nStrade con provincia ora: {n_with_prov} / {len(df)} ({n_with_prov/len(df)*100:.1f}%)")
print(f"\nBreakdown source:")
print(df["source"].value_counts())

df.to_csv(V2_OUT / "strada_provincia.csv", index=False)
print(f"\n→ aggiornato {V2_OUT / 'strada_provincia.csv'}")
