"""
statistical_analysis.py — Hypothesis tests, correlation, and anomaly detection.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from config import SENTIMENT_ORDER


def test_fear_vs_greed_pnl(df: pd.DataFrame) -> dict:
    """Mann-Whitney U: Are Fear vs Greed PnL distributions different?"""
    fear_pnl  = df[df["sentiment_group"] == "Fear Zone"]["Closed PnL"].dropna()
    greed_pnl = df[df["sentiment_group"] == "Greed Zone"]["Closed PnL"].dropna()
    stat, p   = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
    return {
        "test"        : "Mann-Whitney U (Fear vs Greed PnL)",
        "U_stat"      : round(stat, 2),
        "p_value"     : round(p, 6),
        "significant" : p < 0.05,
        "conclusion"  : "REJECT H0 — sentiment significantly affects PnL" if p < 0.05 else "FAIL TO REJECT H0",
    }


def test_win_rate_difference(df: pd.DataFrame) -> dict:
    """Chi-squared: Is win rate different across sentiment regimes?"""
    contingency = (
        df[df["classification"].notna()]
        .groupby(["classification", "profitable"], observed=False)
        .size().unstack(fill_value=0).reindex(SENTIMENT_ORDER)
    )
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {
        "test"       : "Chi-squared (win rate × sentiment)",
        "chi2"       : round(chi2, 4),
        "p_value"    : round(p, 6),
        "dof"        : dof,
        "significant": p < 0.05,
        "conclusion" : "Win rate significantly differs across sentiments" if p < 0.05 else "No significant difference",
    }


def test_trade_size_by_sentiment(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis: Does trade size differ across sentiment regimes?"""
    groups = [df[df["classification"] == s]["Size USD"].dropna().values for s in SENTIMENT_ORDER]
    stat, p = stats.kruskal(*groups)
    return {
        "test"       : "Kruskal-Wallis (trade size × sentiment)",
        "H_stat"     : round(stat, 4),
        "p_value"    : round(p, 6),
        "significant": p < 0.05,
        "conclusion" : "Trade size significantly differs across sentiments" if p < 0.05 else "No significant difference",
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix for key numeric variables."""
    numeric_cols = ["value", "Size USD", "Closed PnL", "Fee", "fee_pct", "log_size_usd"]
    subset = df[numeric_cols].dropna()
    corr, _ = stats.spearmanr(subset)
    return pd.DataFrame(corr, index=numeric_cols, columns=numeric_cols)


def sentiment_pnl_correlation(df: pd.DataFrame) -> dict:
    """Spearman correlation: sentiment score vs Closed PnL."""
    sub = df[["value", "Closed PnL"]].dropna()
    r, p = stats.spearmanr(sub["value"], sub["Closed PnL"])
    return {"rho": round(r, 4), "p_value": round(p, 6), "n": len(sub)}


def detect_anomalous_trades(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Isolation Forest anomaly detection on trade features."""
    features = ["Size USD", "Closed PnL", "fee_pct"]
    sub = df[features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    df = df.copy()
    df.loc[sub.index, "anomaly"] = labels == -1
    n = (df["anomaly"] == True).sum()
    print(f"[INFO] Detected {n} anomalous trades ({n/len(df):.2%})")
    return df


def print_statistical_summary(df: pd.DataFrame) -> None:
    print("\n══════════════════════════════════════════")
    print("  STATISTICAL ANALYSIS")
    print("══════════════════════════════════════════")
    for result in [test_fear_vs_greed_pnl(df), test_win_rate_difference(df), test_trade_size_by_sentiment(df)]:
        print(f"\n[{result['test']}]")
        for k, v in result.items():
            if k != "test":
                print(f"  {k}: {v}")
    print("\n[Sentiment-PnL Spearman Correlation]")
    for k, v in sentiment_pnl_correlation(df).items():
        print(f"  {k}: {v}")
