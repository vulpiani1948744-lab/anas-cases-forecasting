"""
16 - COVID regionale (zone colorate) — approssimazione strutturata.

Costruisce per ogni (regione, mese) tra 2020-03 e 2022-04 un punteggio
covid_severity_regionale (0=bianca, 1=gialla, 2=arancione, 3=rossa, 4=lockdown nazionale)
basato sulla storia documentata delle zone in Italia.

Periodo coperto:
- 2020-03 a 2020-05: lockdown nazionale (4) → tutte le regioni
- 2020-06 a 2020-10: ripresa graduale, restrizioni territoriali leggere (1-2)
- 2020-11 a 2021-04: zone colorate intense, varia per regione (2-3 medio)
- 2021-05 a 2021-12: ridotte (1-2)
- 2022-01 a 2022-04: omicron, varia (1-2)

Output:
- v2/data_esterni/covid_regionale.csv : (regione, mese_data, severity)
"""

from pathlib import Path
import pandas as pd
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent.parent
ESTERNI = REPO_DIR / "v2" / "data_esterni"
ESTERNI.mkdir(exist_ok=True)

REGIONI = [
    "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
    "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
    "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia",
    "Toscana", "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto",
]

# bias regionale dell'intensità delle restrizioni: regioni del Nord ed Emilia-Romagna
# colpite più duramente nella fase iniziale, Lombardia/Piemonte/Veneto/Emilia in zone arancioni-rosse
# più frequentemente
BIAS_REGIONALE = {
    "Lombardia": +0.8, "Veneto": +0.6, "Piemonte": +0.7, "Emilia-Romagna": +0.6,
    "Liguria": +0.4, "Friuli-Venezia Giulia": +0.3, "Trentino-Alto Adige": +0.3,
    "Valle d'Aosta": +0.3, "Marche": +0.2, "Toscana": +0.2, "Lazio": +0.1, "Umbria": +0.1,
    "Abruzzo": -0.1, "Molise": -0.2, "Campania": +0.3, "Basilicata": -0.2,
    "Puglia": -0.1, "Calabria": -0.2, "Sicilia": -0.1, "Sardegna": -0.2,
}

# intensità nazionale base per mese
SEVERITY_NATIONAL = {
    "2020-03": 4, "2020-04": 4, "2020-05": 3,  # lockdown
    "2020-06": 1, "2020-07": 0, "2020-08": 0, "2020-09": 1,
    "2020-10": 2, "2020-11": 3, "2020-12": 2,
    "2021-01": 2, "2021-02": 2, "2021-03": 3, "2021-04": 3,
    "2021-05": 2, "2021-06": 1, "2021-07": 0, "2021-08": 0, "2021-09": 1,
    "2021-10": 0, "2021-11": 0, "2021-12": 1,
    "2022-01": 2, "2022-02": 1, "2022-03": 1, "2022-04": 0,
}

# %% costruisci tabella (regione, mese, severity)
rows = []
all_months = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
for mese in all_months:
    ym = mese.strftime("%Y-%m")
    base = SEVERITY_NATIONAL.get(ym, 0)
    for reg in REGIONI:
        # mese fuori dal periodo COVID o pre-COVID: 0
        if ym not in SEVERITY_NATIONAL:
            sev = 0
        # lockdown nazionale: ignora bias regionale (tutti uguali)
        elif base >= 4:
            sev = base
        # base = 0 (mesi senza restrizioni nel dict): no bias regionale
        elif base == 0:
            sev = 0
        # mesi con restrizioni: applica bias regionale
        else:
            sev = max(0, min(5, base + BIAS_REGIONALE[reg]))
        rows.append({
            "regione": reg,
            "mese_data": mese,
            "covid_severity_reg": round(sev, 2),
        })

covid_reg = pd.DataFrame(rows)
print(f"Covid regionale: {len(covid_reg)} righe ({covid_reg['regione'].nunique()} regioni × {covid_reg['mese_data'].nunique()} mesi)")

covid_reg.to_csv(ESTERNI / "covid_regionale.csv", index=False)
print(f"→ {ESTERNI / 'covid_regionale.csv'}")

print("\nEsempio (Lombardia):")
print(covid_reg[covid_reg["regione"]=="Lombardia"].to_string(index=False))
