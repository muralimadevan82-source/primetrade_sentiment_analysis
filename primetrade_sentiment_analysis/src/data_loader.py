"""
data_loader.py — Load, validate, and summarize raw datasets.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import RAW_SENTIMENT, RAW_TRADES


def load_sentiment(path: Path = RAW_SENTIMENT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["classification"] = pd.Categorical(
        df["classification"],
        categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
        ordered=True,
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_trades(path: Path = RAW_TRADES) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["Timestamp IST"], dayfirst=True, errors="coerce")
    df["date"]     = df["datetime"].dt.normalize()
    for col in ["Execution Price", "Size Tokens", "Size USD", "Closed PnL", "Fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Side"] = df["Side"].str.upper().str.strip()
    n_bad = df["date"].isna().sum()
    if n_bad:
        print(f"[WARN] Dropped {n_bad} rows with unparseable timestamps")
        df = df.dropna(subset=["date"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def data_quality_report(sentiment: pd.DataFrame, trades: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    for name, df in [("Sentiment", sentiment), ("Trades", trades)]:
        print(f"\n── {name} ──")
        print(f"  Shape        : {df.shape}")
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        print(f"  Null counts  : {nulls.to_dict() if len(nulls) else 'None'}")
        print(f"  Duplicates   : {df.duplicated().sum()}")
        print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    overlap = pd.merge(
        sentiment[["date"]].drop_duplicates(),
        trades[["date"]].drop_duplicates(), on="date"
    )
    print(f"\n── Overlap: {len(overlap)} matching dates ──")
    print("=" * 60)
