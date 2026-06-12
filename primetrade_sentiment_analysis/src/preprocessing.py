"""
preprocessing.py — Data cleaning, merging, and feature engineering.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import SENTIMENT_TO_ZONE, MERGED_DATASET


def merge_datasets(sentiment: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    fg_slim = sentiment[["date", "value", "classification"]].copy()
    merged = trades.merge(fg_slim, on="date", how="left")
    coverage = merged["classification"].notna().mean()
    print(f"[INFO] Sentiment coverage: {coverage:.1%} of trades matched")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["profitable"]      = df["Closed PnL"] > 0
    df["loss"]            = df["Closed PnL"] < 0
    df["pnl_abs"]         = df["Closed PnL"].abs()
    df["sentiment_group"] = df["classification"].map(SENTIMENT_TO_ZONE)
    df["month"]           = df["datetime"].dt.to_period("M")
    df["hour"]            = df["datetime"].dt.hour
    df["dow"]             = df["datetime"].dt.day_name()
    df["log_size_usd"]    = np.log1p(df["Size USD"].clip(lower=0))
    df["fee_pct"]         = np.where(df["Size USD"] > 0, df["Fee"] / df["Size USD"], np.nan)
    return df


def remove_outliers(df: pd.DataFrame, col: str, method: str = "iqr", k: float = 3.0) -> pd.DataFrame:
    if method == "iqr":
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = df[col].between(q1 - k * iqr, q3 + k * iqr)
    elif method == "zscore":
        z = (df[col] - df[col].mean()) / df[col].std()
        mask = z.abs() < k
    else:
        raise ValueError(f"Unknown method: {method}")
    removed = (~mask).sum()
    print(f"[INFO] Removed {removed} outliers from '{col}' ({method}, k={k})")
    return df[mask].copy()


def save_merged(df: pd.DataFrame, path=MERGED_DATASET) -> None:
    df.to_csv(path, index=False)
    print(f"[INFO] Merged dataset saved → {path}")
