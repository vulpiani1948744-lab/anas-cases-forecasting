"""
03 - Costruzione della matrice di feature (train + test).

Assembla in un unico dataset robusto:
  griglia (strada x mese) 2016-2025
  + tipologie strade        (tipo_strada, categoria_strada_v2, flag, regione_macro)
  + metadati strada          (km, traffico leggeri/pesanti, n_tratte...)
  + metadati strada STATICI  (km, traffico, territorio; NO totali 2024/25 = leakage)
  + calendario               (giorni lavorativi, weekend, festivi, ponti)
  + ENCODING CICLICO sin/cos (mese/stagionalita, settimana dell'anno)
  + meteo per provincia/mese (temperatura, pioggia, neve, vento)
  + profilo storico strada   (giorno/notte, mix canali/categorie, media storica)
  + lag temporali            (stesso mese anno scorso, media rolling 12m)

Output: output/dataset_features.csv
"""
import numpy as np
import pandas as pd
import holidays
import config as C

# ---------------------------------------------------------------- griglia
print("Costruisco la griglia strada x mese...")
target = pd.read_csv(C.OUT / "target_mensile.csv", parse_dates=["mese_data"])
test = pd.read_csv(C.TEST_CSV, parse_dates=["mese_data"]).rename(columns={"strada": "Strada"})

all_strade = sorted(set(target["Strada"]) | set(test["Strada"]))
train_months = pd.date_range(C.TRAIN_START, C.TRAIN_END, freq="MS")

grid_train = (pd.MultiIndex.from_product([all_strade, train_months],
              names=["Strada", "mese_data"]).to_frame(index=False)
              .merge(target, on=["Strada", "mese_data"], how="left"))
# strade mai viste nel train -> target NaN (non osservato); altre: 0 gia' imputati
seen = set(target["Strada"])
grid_train.loc[~grid_train["Strada"].isin(seen), "num_richieste"] = np.nan
grid_train["split"] = "train"

grid_test = test.copy()
grid_test["num_richieste"] = np.nan
grid_test["split"] = "test"
grid_test["test_order"] = np.arange(len(grid_test))   # per ricostruire l'ordine submission

df = pd.concat([grid_train, grid_test], ignore_index=True)
df["test_order"] = df["test_order"].astype("Int64")
df["anno"] = df["mese_data"].dt.year
df["mese"] = df["mese_data"].dt.month
print(f"  righe totali: {len(df):,} (test: {(df['split'] == 'test').sum():,})")

# ---------------------------------------------- tipologie / categorie strada
def tipo_strada(s):
    u = str(s).upper()
    if u.startswith("AU "):
        return "AUTOSTRADA_DIR" if "DIR" in u else "AUTOSTRADA"
    if u.startswith("SS "):
        return "STATALE"
    if u.startswith("RA "):
        return "RACCORDO"
    if u.startswith("NS ") or u.startswith("NSA "):
        return "NUOVA_STATALE"
    return "ALTRO"

def categoria_v2(s):
    u = str(s).upper()
    if u.startswith("AU "):
        if "RACC" in u: return "AUTOSTRADA_RACC"
        if "DIR" in u: return "AUTOSTRADA_DIR"
        if "GRANDE RACCORDO ANULARE" in u or "GRA" in u: return "GRA"
        return "AUTOSTRADA"
    if u.startswith("RA "):
        return "TANGENZIALE_RA" if "TANGENZIALE" in u else "RACCORDO_AUTO"
    if u.startswith("SS "):
        for k, v in [("TANGENZIALE", "TANGENZIALE_SS"), ("RACC", "STATALE_RACC"),
                     ("DIR", "STATALE_DIR"), ("VAR", "STATALE_VAR"),
                     ("BIS", "STATALE_BIS"), ("TER", "STATALE_TER"),
                     ("RADD", "STATALE_RADD")]:
            if k in u: return v
        return "STATALE_PRINC"
    if u.startswith("NS "): return "NUOVA_STATALE"
    if u.startswith("XS "): return "EX_STATALE"
    return "ALTRO"

df["tipo_strada"] = df["Strada"].map(tipo_strada)
df["categoria_strada_v2"] = df["Strada"].map(categoria_v2)

# flag toponomastici robusti
U = df["Strada"].astype(str).str.upper()
mont_kw = ["VALLE", "VALLI", "MONTE", "MONTI", "COLLE", "PASSO", "ALPI", "ALPE",
           "BRENNERO", "STELVIO", "ABETONE", "GRAN SASSO", "TONALE", "DOLOMITI"]
cost_kw = ["ADRIATICA", "AURELIA", "AMALFITANA", "JONICA", "IONICA", "ROMEA",
           "COSTIERA", "TIRRENO", "GARGANICA"]
df["is_montagna"] = U.apply(lambda u: int(any(k in u for k in mont_kw)))
df["is_costiera"] = U.apply(lambda u: int(any(k in u for k in cost_kw)))
df["is_tangenziale"] = U.str.contains("TANGENZIALE").astype(int)
df["is_dir"] = U.str.contains(" DIR").astype(int)
df["is_bis"] = U.str.contains(" BIS").astype(int)
# zona geografica del tracciato: montagna / mare (costiera) / pianura-rettilineo
df["geo_zona"] = np.select(
    [df["is_montagna"] == 1, df["is_costiera"] == 1],
    ["MONTAGNA", "MARE"], default="PIANURA")

# ----------------------------------------------------- provincia / regione
prov = pd.read_csv(C.STRADA_PROV_CSV)[["Strada", "Provincia", "regione"]]
df = df.merge(prov, on="Strada", how="left")
nord = {"Lombardia", "Piemonte", "Veneto", "Friuli-Venezia Giulia", "Liguria",
        "Emilia-Romagna", "Trentino-Alto Adige", "Valle d'Aosta"}
centro = {"Toscana", "Marche", "Umbria", "Lazio", "Abruzzo", "Molise"}
sud = {"Campania", "Puglia", "Basilicata", "Calabria"}
isole = {"Sicilia", "Sardegna"}
def macro(r):
    if r in nord: return "NORD"
    if r in centro: return "CENTRO"
    if r in sud: return "SUD"
    if r in isole: return "ISOLE"
    return "__NA__"
df["regione_macro"] = df["regione"].map(macro)
df["is_isole"] = df["regione"].isin(isole).astype(int)

# ------------------------------------- metadati strada (SOLO attributi statici)
# ATTENZIONE: road_metadata contiene anche colonne 2024_* e 2025_* che NON sono
# attributi statici ma aggregati annuali del target nel periodo di test: usarle
# (o riscalare su di esse) e' LEAKAGE -> il modello "vedrebbe la risposta" e non
# generalizzerebbe al futuro reale. Le ignoriamo del tutto. Teniamo solo le
# caratteristiche fisiche/territoriali della strada, note a priori.
rm = pd.read_csv(C.ROAD_META_CSV).rename(columns={"strada": "Strada"})
static = ["n_tratte", "comuni_attraversati", "province_attraversate", "km_min",
          "km_max", "consistenza_gg_media", "leggeri_media", "pesanti_media"]
for c in static:
    rm[c] = pd.to_numeric(rm[c], errors="coerce")

df = df.merge(rm[["Strada"] + static], on="Strada", how="left")
df["lunghezza_km"] = df["km_max"] - df["km_min"]
df["quota_pesanti"] = df["pesanti_media"] / df["consistenza_gg_media"].replace(0, np.nan)
# tortuosita/complessita: piu' tratte per km -> tracciato piu' articolato (vulnerabilita)
df["tratte_per_km"] = df["n_tratte"] / df["lunghezza_km"].replace(0, np.nan)

# --------------------------------------------------- calendario per mese
print("Calcolo feature calendariali...")
ita = holidays.country_holidays("IT", years=range(2016, 2026))
mesi = pd.DataFrame({"mese_data": pd.date_range(C.TRAIN_START, C.TEST_END, freq="MS")})

def cal(md):
    giorni = pd.date_range(md, md + pd.offsets.MonthEnd(0), freq="D")
    n = len(giorni)
    lav = sum(d.weekday() < 5 and d not in ita for d in giorni)
    return pd.Series({
        "giorni_nel_mese": n,
        "n_weekend": sum(d.weekday() >= 5 for d in giorni),
        "n_sabati": sum(d.weekday() == 5 for d in giorni),
        "n_domeniche": sum(d.weekday() == 6 for d in giorni),
        "n_festivi": sum(d in ita for d in giorni),
        "n_festivi_infrasett": sum(d in ita and d.weekday() < 5 for d in giorni),
        "n_ponti": sum(d in ita and d.weekday() in (1, 3) for d in giorni),
        "n_lavorativi_effettivi": lav,
        "pct_lavorativi_mese": lav / n,
    })
mesi = pd.concat([mesi, mesi["mese_data"].apply(cal)], axis=1)

# ----------------------------- ENCODING CICLICO (stagionalita + settimana)
m = mesi["mese_data"].dt.month
# stagionalita: mese a frequenza 1x/2x/3x (annuale, semestrale, quadrimestrale)
mesi["sin_mese"] = np.sin(2 * np.pi * m / 12)
mesi["cos_mese"] = np.cos(2 * np.pi * m / 12)
mesi["sin_mese_x2"] = np.sin(4 * np.pi * m / 12)
mesi["cos_mese_x2"] = np.cos(4 * np.pi * m / 12)
mesi["sin_mese_x3"] = np.sin(6 * np.pi * m / 12)
mesi["cos_mese_x3"] = np.cos(6 * np.pi * m / 12)
# trimestre/stagione
trim = ((m - 1) // 3) + 1
mesi["sin_trim"] = np.sin(2 * np.pi * trim / 4)
mesi["cos_trim"] = np.cos(2 * np.pi * trim / 4)
# giorno-medio dell'anno (stagionalita fine)
giorno_anno = (mesi["mese_data"] + pd.offsets.Day(14)).dt.dayofyear
mesi["sin_giorno_anno"] = np.sin(2 * np.pi * giorno_anno / 365.25)
mesi["cos_giorno_anno"] = np.cos(2 * np.pi * giorno_anno / 365.25)
# parabola stagionale (alto a gen/dic, basso a giugno)
mesi["mese_centrato_sq"] = (m - 6.5) ** 2
# settimana dell'anno (ISO della meta' del mese) con encoding ciclico
sett = (mesi["mese_data"] + pd.offsets.Day(14)).dt.isocalendar().week.astype(int)
mesi["settimana_anno"] = sett
mesi["sin_sett_anno"] = np.sin(2 * np.pi * sett / 52)
mesi["cos_sett_anno"] = np.cos(2 * np.pi * sett / 52)
# flag stagionali grossolani
mesi["is_estate"] = m.isin([6, 7, 8]).astype(int)
mesi["is_inverno"] = m.isin([12, 1, 2]).astype(int)
mesi["is_agosto"] = (m == 8).astype(int)

df = df.merge(mesi, on="mese_data", how="left")

# ------------------------------------------------- meteo per provincia/mese
print("Aggiungo meteo per provincia/mese...")
meteo = pd.read_csv(C.METEO_CSV, parse_dates=["mese_data"])
meteo_cols = [c for c in meteo.columns if c not in ("Provincia", "mese_data")]
df = df.merge(meteo, on=["Provincia", "mese_data"], how="left")
print(f"  copertura meteo provinciale: {df['temp_mean_mese'].notna().mean() * 100:.1f}%")
# fallback: dove manca la provincia, uso la media mensile nazionale (copertura piena, robusto)
naz = meteo.groupby("mese_data")[meteo_cols].mean()
for c in meteo_cols:
    df[c] = df[c].fillna(df["mese_data"].map(naz[c]))
print(f"  copertura meteo con fallback nazionale: {df['temp_mean_mese'].notna().mean() * 100:.1f}%")

# ------------------------------- COVID regionale (regressore esterno, lecito)
# Severity zone-colorate 2020-2022; 0 negli altri mesi (incluso il test 2024-25).
# Spiega il crollo anomalo delle segnalazioni nel periodo pandemico ed evita che
# il modello lo scambi per stagionalita'.
print("Aggiungo COVID regionale...")
covid = pd.read_csv(C.COVID_CSV, parse_dates=["mese_data"])
df = df.merge(covid, on=["regione", "mese_data"], how="left")
df["covid_severity_reg"] = df["covid_severity_reg"].fillna(0.0)
df["is_covid"] = (df["covid_severity_reg"] > 0).astype(int)

# ------------------------------- feature DGP: esposizione / vulnerabilita / trigger
# Una segnalazione nasce da: ESPOSIZIONE (quanti veicoli passano) x VULNERABILITA
# della tratta (lunghezza, orografia) x TRIGGER (meteo, stagionalita).
# Le rendiamo esplicite: aiutano la generalizzazione e danno una narrativa SHAP.
print("Aggiungo feature DGP (esposizione/vulnerabilita/trigger)...")
df["traffico_tot"] = df["leggeri_media"].fillna(0) + df["pesanti_media"].fillna(0)
df["log_esposizione"] = np.log1p(df["traffico_tot"])          # esposizione al rischio
df["esposizione_x_km"] = df["log_esposizione"] * df["lunghezza_km"].fillna(0)
# trigger stagionali sull'esposizione (esodo estivo, traffico pesante invernale)
df["traffico_x_agosto"] = df["traffico_tot"] * df["is_agosto"]
df["pesanti_x_inverno"] = df["pesanti_media"].fillna(0) * df["is_inverno"]
# vulnerabilita orografica attivata dal meteo (neve in montagna)
df["montagna_x_neve"] = df["is_montagna"] * df["snow_sum_mese"].fillna(0)
df["montagna_x_pioggia"] = df["is_montagna"] * df["precip_sum_mese"].fillna(0)

# ----------------------------------------------------- profilo storico strada
# Profilo a doppio cutoff (leakage-safe): train+val usano lo storico <2023,
# il test usa lo storico <2024 (sfrutta legittimamente anche il 2023).
prof_val = pd.read_csv(C.OUT / "profilo_strada_val.csv")
prof_test = pd.read_csv(C.OUT / "profilo_strada_test.csv")
prof_cols = [c for c in prof_val.columns if c != "Strada"]
mask_test = df["split"] == "test"
df = df.merge(prof_val, on="Strada", how="left")          # default per train+val
# sovrascrivo le righe di test con il profilo <2024
df_test_prof = df.loc[mask_test, ["Strada"]].merge(prof_test, on="Strada", how="left")
for c in prof_cols:
    df.loc[mask_test, c] = df_test_prof[c].values

# ----------------------------------------- lag temporali (time-safe, no leak)
# stesso mese dell'anno precedente e media rolling degli ultimi 12 mesi,
# calcolati solo sui valori osservati (i mesi futuri restano NaN).
print("Calcolo lag temporali...")
df = df.sort_values(["Strada", "mese_data"]).reset_index(drop=True)
df["lag_12"] = df.groupby("Strada")["num_richieste"].shift(12)
# 2. Il dato esatto di due anni fa
df["lag_24"] = df.groupby("Strada")["num_richieste"].shift(24)

# 3. IL TREND (Variabile Killer): I casi stanno aumentando o scendendo rispetto a 2 anni fa?
# Se lag_12 > lag_24, il trend è in crescita.
df["trend_yoy"] = df["lag_12"] - df["lag_24"]
# NB: tutti i lag/rolling usano shift>=12 (solo anni precedenti): cosi' si
# comportano IDENTICAMENTE in validazione e in test (nessuna "spiata" sui mesi
# precedenti dello stesso anno, che nel test sarebbero comunque ignoti).
df["roll_mean_12"] = (df.groupby("Strada")["num_richieste"]
                      .transform(lambda s: s.shift(12).rolling(12, min_periods=1).mean()))
df["roll_std_12"] = (df.groupby("Strada")["num_richieste"]
                     .transform(lambda s: s.shift(12).rolling(12, min_periods=3).std()))

PROV_MONTAGNA = ['AO', 'TN', 'BZ', 'BL', 'SO', 'AQ', 'PZ', 'UD', 'BG', 'BS'] 
PROV_MARE = ['GE', 'SV', 'IM', 'SP', 'LI', 'GR', 'NA', 'SA', 'BA', 'LE', 'VE', 'RN', 'PA', 'CT', 'CA', 'SS'] 

def classifica_geografia(province_str):
    if pd.isna(province_str): return 'Pianura/Altro'
    prov_list = str(province_str).split(',') 
    if any(p in PROV_MONTAGNA for p in prov_list): return 'Montagna'
    if any(p in PROV_MARE for p in prov_list): return 'Mare'
    return 'Pianura/Altro'

# Applica la funzione
df['geo_zona'] = df['province_attraversate'].apply(classifica_geografia).astype('category')

# ----------------------------------------------------------------- salva
out = C.OUT / "dataset_features.csv"
df.to_csv(out, index=False)
print(f"\n-> {out}")
print(f"   righe: {len(df):,} | colonne: {df.shape[1]} | test: {(df['split'] == 'test').sum():,}")
