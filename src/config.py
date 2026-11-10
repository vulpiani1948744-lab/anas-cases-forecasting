"""
config.py — configurazione centrale della pipeline robusta.

Target del progetto: prevedere il numero mensile di segnalazioni/incidenti
stradali per ciascuna strada (num_richieste), per il periodo di test (2024-2025).

Filosofia di questa versione:
- una sola pipeline lineare, leggibile e riproducibile (no 40 script sparsi)
- solo le feature più robuste e generalizzabili (no overfitting su trick fragili)
- due modelli forti: LightGBM + CatBoost, ognuno multi-seed, poi blend
- validazione temporale onesta: alleniamo sul passato, validiamo sull'ultimo
  anno disponibile (mima la previsione del futuro che il prof valuterà)
- nessun leakage: i profili storici e i lag usano SOLO dati passati al cutoff
"""

from pathlib import Path

# --- percorsi -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # radice del repo
DATA = ROOT / "data"                                   # artefatti riutilizzabili

# dati grezzi della competizione (non versionati): vedi data/raw/README.md
DATA_DIR = DATA / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ROAD_META_CSV = DATA_DIR / "road_metadata.csv"

OUT = ROOT / "output"                                  # output intermedi + submission
OUT.mkdir(exist_ok=True)

# input esterni precalcolati (meteo per provincia/mese, mapping strada->provincia)
METEO_CSV = DATA / "meteo_mensile.csv"
STRADA_PROV_CSV = DATA / "strada_provincia.csv"
COVID_CSV = DATA / "covid_regionale.csv"

# --- orizzonti temporali --------------------------------------------------
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-01"     # ultimo mese con target osservato
TEST_START = "2024-01-01"
TEST_END = "2025-12-01"

# validazione: alleno su <2023, valido sul 2023 (ultimo anno pieno e onesto).
# Il profilo storico/lag viene quindi calcolato SOLO su dati < VAL_YEAR per
# non sporcare la validazione.
VAL_YEAR = 2023
PROFILE_CUTOFF = "2023-01-01"   # i profili storici usano solo eventi < questa data

# --- deduplicazione segnalazioni -----------------------------------------
DEDUP_KEYS = [
    "Localita", "Provincia", "Canale arrivo", "Tipologia",
    "Categoria", "Gruppo", "Strada", "Tipo Persona",
]

# --- colonne categoriche per i modelli ------------------------------------
CAT_COLS = ["tipo_strada", "categoria_strada_v2", "regione", "regione_macro", "Provincia","geo_zona"]

# colonne da NON dare in pasto ai modelli
DROP_COLS = [
    "Strada", "mese_data", "num_richieste", "split", "test_order",
    "anno", "trend", "n_eventi_storici",
]

# --- modellazione del conteggio (natura del target) -----------------------
# Il target e' un conteggio mensile zero-inflated e sovradisperso (var >> media):
# la loss corretta non e' l'MSE gaussiana ma la Tweedie deviance (compound
# Poisson-Gamma, 1<p<2), che modella nativamente la massa in zero e la coda
# destra e non produce mai predizioni negative. Vedi 00_eda.py per le evidenze.
TWEEDIE_VP = 1.3                  # tweedie_variance_power (1<p<2: Poisson-Gamma)

# regressione quantilica per gli intervalli di previsione (incertezza)
QUANTILES = [0.05, 0.5, 0.95]     # -> intervallo fiduciario al 90%

# validazione walk-forward out-of-time: (ultimo anno di train, anno di validazione)
WALK_FORWARD = [(2020, 2021), (2021, 2022), (2022, 2023)]

# --- ensemble -------------------------------------------------------------
SEEDS = [42, 7, 123, 2024, 999]

# parametri di default (da Optuna su questo problema). 05_tune.py li può
# rigenerare; se i file json non esistono, 06_train.py usa questi.
DEFAULT_LGBM = {
    "learning_rate": 0.0315, "num_leaves": 58, "max_depth": 4,
    "min_child_samples": 171, "feature_fraction": 0.92, "bagging_fraction": 0.917,
    "bagging_freq": 8, "lambda_l1": 0.0072, "lambda_l2": 0.0219, "min_split_gain": 0.61,
}
DEFAULT_CB = {
    "learning_rate": 0.0286, "depth": 7, "l2_leaf_reg": 1.217,
    "random_strength": 0.682, "bagging_temperature": 0.778,
    "border_count": 125, "min_data_in_leaf": 146,
}


def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error — la metrica del contest."""
    import numpy as np
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def poisson_deviance(y_true, y_pred):
    """Devianza di Poisson media — misura la bontà sul DGP a conteggi."""
    import numpy as np
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(y_pred, 1e-9, None)
    pos = yt > 0
    term = np.zeros_like(yt)
    term[pos] = yt[pos] * np.log(yt[pos] / yp[pos])
    return float(2 * np.mean(term - (yt - yp)))
