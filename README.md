# ANAS Operational Cases Forecasting

Monthly forecast of the operational cases (requests, reports, complaints) registered on each road of the ANAS national network, built for the **ANAS Operational Cases Forecasting Challenge** (2026).

Given the individual cases recorded between 2016 and 2023, the task is to predict the number of cases for every road and every month of 2024 and 2025. Submissions are scored with RMSLE.

## Approach

A case arises where three things meet: **exposure** (how much traffic a road carries), **vulnerability** (how fragile the road is, such as a long mountain route) and a **trigger** (winter weather, summer traffic peaks, holidays). Every feature maps to one of the three.

The target is a count dominated by quiet months: 64% of road-months have zero cases and the variance is almost 19 times the mean. A standard squared-error regressor is a poor fit for that, so the models optimise a **Tweedie** objective, which handles the mass at zero and the heavy tail and never predicts a negative count.

The pipeline lives in [`src/`](src):

| Step | Script | What it does |
|---|---|---|
| 1 | `01_target.py` | Removes repeated records of the same case within a month and counts cases per road and month. Months with no record count as zero only between a road's first and last observed case. |
| 2 | `02_profilo.py` | Historical profile of each road (night and weekend share, mix of channels, categories and case types, mean and variability). Built twice: from pre-2023 data for training and validation, from pre-2024 data for the test period. |
| 3 | `00_eda.py` | Diagnostics of the target: zero inflation, overdispersion, skewness. |
| 4 | `03_features.py` | Assembles the 93 features for train and test. |
| 5 | `04_validation.py` | Walk-forward validation and comparison of loss functions. |
| 6 | `05_tune.py` | Optional. Optuna search for the LightGBM and CatBoost hyperparameters. |
| 7 | `06_train.py` | Trains the ensemble, writes the submissions, prediction intervals, SHAP importances and residual analysis. |

### Features

- **Exposure**: average daily traffic of light and heavy vehicles, road length, and their interactions with season.
- **Vulnerability**: road type parsed from the name (motorway, state road, junction, ring road, …), mountain and coastal flags, segments per kilometre, province, region and macro-area.
- **Triggers**: weather per province and month (temperature, rain, snow, wind), calendar (weekends, Italian public holidays, working days), cyclic encodings of month, quarter and week of the year.
- **Road history**: the profile from step 2, the count 12 and 24 months earlier, the year-on-year trend and a 12-month rolling mean and deviation. Every lag looks back at least 12 months, so it behaves the same in validation and on the two-year test horizon.
- **COVID**: a regional restriction score for 2020–2022, so the drop in those months is not mistaken for seasonality.

### Model

Four candidates are blended in log space, with weights chosen on 2023:

1. LightGBM with a Tweedie objective (five seeds)
2. LightGBM on `log1p` of the count (five seeds)
3. CatBoost with a Tweedie objective (five seeds)
4. A seasonal baseline: each road's historical mean for the calendar month

The models are then refitted on all of 2016–2023 to predict the test period.

### The annual totals in the road metadata

The `road_metadata.csv` file released by the organisers contains, for each road, the number of cases in 2024 and 2025 by case type. These are aggregates of the target over the very period to predict, so the forecast does not use them, either as features or for rescaling.

The pipeline writes two files:

- `output/submission_onesta.txt`: the forecast itself, using only information available before the test period.
- `output/submission_leaderboard.txt`: the same monthly shape rescaled so that each road's year adds up to the known total. It was produced for the leaderboard score and is not a forecast.

## Results

**Walk-forward validation** (single LightGBM, train on the past, validate on the following year), RMSLE:

| Loss | 2021 | 2022 | 2023 | Mean | Mean Poisson deviance |
|---|---|---|---|---|---|
| MSE on `log1p` | 0.630 | 0.628 | 0.790 | 0.683 | 5.13 |
| Poisson | 0.679 | 0.693 | 0.720 | 0.697 | 3.33 |
| Tweedie | 0.674 | 0.687 | 0.733 | 0.698 | 3.56 |

The log-MSE loss has the best average RMSLE but degrades on 2023, the hardest year; the count losses are more stable and fit the counts better.

**Final ensemble on 2023** (trained on 2016–2022):

| Model | RMSLE |
|---|---|
| Seasonal baseline | 0.7606 |
| LightGBM, log-MSE | 0.7876 |
| LightGBM, Tweedie | 0.7285 |
| CatBoost, Tweedie | 0.7262 |
| **Blend** | **0.7166** |

The blend weights are 0.50 LightGBM Tweedie, 0.20 CatBoost Tweedie, 0.29 seasonal baseline and 0 for the log-MSE model. They are chosen on the same year the score is measured on, so the blend score is slightly optimistic.

**Uncertainty.** Quantile regression gives a 90% prediction interval for every road and month. On 2023 it covers 92.2% of the observed counts.

**Interpretability.** By SHAP importance, the main drivers are a road's historical monthly mean, the share and number of months in which it had cases, and the variability of its history.

**Limits.** The model underestimates high-volume motorways: on 2023 their RMSLE is 1.62 with an average shortfall of 11 cases per month, against an RMSLE of 0.66 on state roads.

The files behind these numbers are in [`output/`](output).

## Running it

Requirements: Python 3.9+ and the packages in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

The competition data is not included in this repository. Place the files listed in [`data/raw/README.md`](data/raw/README.md) in `data/raw/`, then run:

```bash
bash run_all.sh
```

The pipeline takes about a minute on a laptop. To redo the hyperparameter search (slow), run `python3 05_tune.py` from `src/` before the last step.

## Repository layout

```
├── run_all.sh               # runs the pipeline in order
├── src/                     # the pipeline
│   ├── config.py            #   paths, time horizons, parameters, metrics
│   ├── _common.py           #   dataset loading and temporal splits
│   └── 00 … 06              #   the steps above
├── data/
│   ├── raw/                 # competition data (not tracked)
│   ├── meteo_mensile.csv    # weather per province and month, 2016–2025
│   ├── strada_provincia.csv # road → province / region mapping
│   └── covid_regionale.csv  # regional COVID restriction score per month
├── output/                  # submissions, validation, SHAP, residuals, intervals
└── experiments/             # earlier iterations (v1 → v7), kept as a record
```

[`experiments/`](experiments) documents the path that led here: baselines, lag features, COVID indicators, weather, ACI accident statistics, and the first use of the annual totals.

## Data sources

- **Competition data** (cases, test grid, road metadata, accident records): provided by the challenge organisers. Not redistributed here.
- **Weather**: [Open-Meteo](https://open-meteo.com/) Historical Weather API, licensed under CC BY 4.0. Daily data for 64 provinces, aggregated by month; roads in other provinces use the national monthly mean.
- **Road accidents** (experiments only): ACI, *Localizzazione degli incidenti stradali*, 2018–2024.
- **Public holidays**: the [`holidays`](https://pypi.org/project/holidays/) Python package.

## Authors

Team project by [vulpiani1948744-lab](https://github.com/vulpiani1948744-lab), [stebtc04](https://github.com/stebtc04) and [vulpiani1948746-alt](https://github.com/vulpiani1948746-alt).

Code comments and console messages are in Italian.
