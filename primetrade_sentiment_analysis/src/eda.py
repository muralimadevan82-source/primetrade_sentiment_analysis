"""
eda.py — Exploratory data analysis.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import SENTIMENT_ORDER


def trader_profitability_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Account")
        .agg(
            total_pnl   =("Closed PnL", "sum"),
            trade_count =("Closed PnL", "count"),
            win_rate    =("profitable",  "mean"),
            avg_size_usd=("Size USD",    "mean"),
            avg_pnl     =("Closed PnL",  "mean"),
        )
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )


def sentiment_pnl_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("classification", observed=False)["Closed PnL"]
        .agg(["mean", "median", "std", "count"])
        .reindex(SENTIMENT_ORDER)
    )
    summary.columns = ["avg_pnl", "median_pnl", "std_pnl", "trade_count"]
    return summary


def sentiment_behavior_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("classification", observed=False)
    out = pd.DataFrame({
        "win_rate"   : grp["profitable"].mean(),
        "loss_rate"  : grp["loss"].mean(),
        "avg_size"   : grp["Size USD"].mean(),
        "trade_count": grp["Closed PnL"].count(),
        "avg_fee_pct": grp["fee_pct"].mean(),
    }).reindex(SENTIMENT_ORDER)
    buy_ratio = (
        df[df["Side"] == "BUY"]
        .groupby("classification", observed=False).size()
        / grp.size()
    )
    out["buy_ratio"] = buy_ratio
    return out


def symbol_performance(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("Coin")
        .agg(
            total_pnl   =("Closed PnL", "sum"),
            trade_count =("Closed PnL", "count"),
            win_rate    =("profitable",  "mean"),
            avg_size_usd=("Size USD",    "mean"),
        )
        .sort_values("total_pnl", ascending=False)
        .head(top_n)
        .reset_index()
    )


def symbol_by_sentiment(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    return (
        df.groupby(["sentiment_group", "Coin"])["Closed PnL"]
        .sum().reset_index()
        .sort_values(["sentiment_group", "Closed PnL"], ascending=[True, False])
        .groupby("sentiment_group").head(top_n)
        .reset_index(drop=True)
    )


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month")
        .agg(
            trades   =("Closed PnL", "count"),
            total_pnl=("Closed PnL", "sum"),
            win_rate =("profitable",  "mean"),
            avg_size =("Size USD",    "mean"),
        )
        .reset_index()
    )


def hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour")
        .agg(trades=("Closed PnL", "count"), avg_pnl=("Closed PnL", "mean"))
        .reset_index()
    )


def print_full_eda(df: pd.DataFrame) -> None:
    print("\n══════════════════════════════════════════")
    print("  EDA SUMMARY")
    print("══════════════════════════════════════════")
    print(f"\n[1] Dataset Shape: {df.shape}")
    print("\n[2] Sentiment PnL Summary:")
    print(sentiment_pnl_summary(df).to_string())
    print("\n[3] Sentiment Behavior Summary:")
    print(sentiment_behavior_summary(df).to_string())
    print("\n[4] Top 10 Accounts by PnL:")
    print(trader_profitability_summary(df).head(10).to_string(index=False))
    print("\n[5] Top 10 Symbols by Total PnL:")
    print(symbol_performance(df, top_n=10).to_string(index=False))
    print("\n[6] Monthly Trends:")
    print(monthly_trends(df).to_string(index=False))
