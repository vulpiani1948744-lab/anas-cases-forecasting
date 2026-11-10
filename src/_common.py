"""
_common.py — utilità condivise da 04_validation.py, 05_tune.py e 06_train.py:
caricamento dataset, selezione feature, split temporale di validazione.
"""
import numpy as np
import pandas as pd
import config as C


def load_dataset():
    df = pd.read_csv(C.OUT / "dataset_features.csv", parse_dates=["mese_data"])
    for c in C.CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in C.DROP_COLS]


def splits(df):
    """Train = osservato e < VAL_YEAR; Val = anno VAL_YEAR; Test = split test."""
    obs = df["num_richieste"].notna() & (df["split"] == "train")
    tr = df[obs & (df["anno"] < C.VAL_YEAR)].copy()
    va = df[obs & (df["anno"] == C.VAL_YEAR)].copy()
    te = df[df["split"] == "test"].copy()
    return tr, va, te


def oot_fold(df, train_end_year, val_year):
    """Un fold walk-forward out-of-time: train su <=train_end_year, val su val_year."""
    obs = df["num_richieste"].notna() & (df["split"] == "train")
    tr = df[obs & (df["anno"] <= train_end_year)].copy()
    va = df[obs & (df["anno"] == val_year)].copy()
    return tr, va


def cat_for_catboost(X, cat_cols):
    """CatBoost vuole le categoriche come stringa senza NaN."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__NA__")
    return X
