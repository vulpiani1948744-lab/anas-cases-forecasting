# Experiments

The iterations that led to the final pipeline in [`../src`](../src). They are kept as a record of what was tried: the scripts are numbered in the order they were written and each one builds on the outputs of the previous ones.

## v1 — baselines and first models

| Script | Purpose |
|---|---|
| `v1/01_eda_target.py` | Exploration, deduplication and monthly target. |
| `v1/02_dataset_base.py` | Full road × month grid for train and test, road metadata, calendar features. |
| `v1/03_baseline.py` | Temporal split (train 2016–2022, validation 2023) and seasonal-naive baselines. |
| `v1/04_profilo_strada.py` | Historical profile of each road. |
| `v1/05_train_lgbm.py` | Lag features, a simple COVID dummy and the first LightGBM model. |
| `v1/06_arricchimenti.py` | Detailed COVID features and accident features from `incidenti.xlsx`. |
| `v1/07_train_v2.py` | LightGBM on the enriched dataset. |
| `v1/08_ensemble.py` | LightGBM + CatBoost + baseline blend, with weights chosen on 2023. |
| `v1/09_fix_submissions.py` | Rewrites the submission files in the exact format of the example. |

## v2 — external data and annual totals

| Script | Purpose |
|---|---|
| `v2/scripts/10_strada_provincia.py` | Road → province mapping (modal province in train, name heuristics otherwise). |
| `v2/scripts/11_meteo_download.py` | Downloads daily weather per province from Open-Meteo, with a local cache. |
| `v2/scripts/11b_meteo_from_cache.py` | Builds daily and monthly weather tables from the cache. |
| `v2/scripts/12_meteo_aggregate.py` | Joins province and weather to the v1 dataset. |
| `v2/scripts/13_train_v3.py` | Ensemble on the dataset with weather and province. |
| `v2/scripts/14_ablations.py` | Ablations: dropping 2016, iterative imputation of missing lags. |
| `v2/scripts/15_provincia_v2.py` | Better province mapping: city names found in the road name. |
| `v2/scripts/16_covid_regionale.py` | Regional COVID restriction score per month. |
| `v2/scripts/17_train_v5.py` | Rebuilt dataset and five-seed ensemble. |
| `v2/scripts/18_aci_parse.py` | Parses the ACI accident spreadsheets into monthly counts. |
| `v2/scripts/19_aci_merge.py` | Matches ACI road codes to the competition road names. |
| `v2/scripts/20_train_v6.py` | Ensemble with ACI accident features. |
| `v2/scripts/21_road_meta_v2.py` | Adds the 2024/2025 annual totals from the updated road metadata. |
| `v2/scripts/22_train_v7.py` | Ensemble plus rescaling of the predictions to the annual totals. |

## Validation scores recorded along the way

RMSLE on 2023, as printed by the scripts:

| Model | RMSLE |
|---|---|
| Seasonal-naive baseline | 0.7606 |
| LightGBM v1 + baseline blend | 0.7392 |
| LightGBM v2 (COVID + accidents) + blend | 0.7389 |
| v1 ensemble, conservative blend | 0.7368 |
| Ensemble with weather and province | 0.7359 |

The final pipeline starts from the same seasonal baseline (0.7606) and reaches 0.7166 on 2023.

Excluding 2016 from training (script 14) worsened the leaderboard score, so 2016 was kept. Scripts 21–22 introduced the 2024/2025 annual totals as features and for rescaling. The final pipeline treats them as test-period information: it keeps them out of the forecast and applies the rescaling only to a separate leaderboard file.

## What is tracked

All scripts, and every submission file they produced (`output/`, `v2/output/`).

Not tracked, because they are large and can be regenerated: the intermediate datasets, the Open-Meteo cache, the daily weather tables and the ACI files. The monthly weather table, the road → province mapping and the regional COVID score that scripts 11b, 15 and 16 produce are the ones in [`../data`](../data).

## Running them

Scripts resolve their paths from their own location and read the competition data from `../data/raw`. Run them in numerical order from any directory, for example `python3 experiments/v1/01_eda_target.py`.

Two things to know before re-running:

- The v1 scripts were written for the first release of `road_metadata.csv`, which was semicolon-separated and had only the static columns. The file was later replaced by the comma-separated version with annual totals that scripts 21–22 and the final pipeline expect.
- Scripts 18–19 need the ACI *Localizzazione degli incidenti stradali* spreadsheets (2018–2024) in `v2/data_esterni/aci_raw/`.
