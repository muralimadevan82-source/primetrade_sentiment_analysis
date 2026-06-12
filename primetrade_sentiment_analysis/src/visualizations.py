"""
visualizations.py — All chart generation.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from config import (
    SENTIMENT_COLORS, ZONE_COLORS, SENTIMENT_ORDER,
    FIGSIZE_WIDE, FIGSIZE_SQUARE, DPI, PLOT_STYLE, OUTPUT_DIR,
)

sns.set_style(PLOT_STYLE)
SENTIMENT_PAL = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]


def _save(fig, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["classification"].value_counts().reindex(SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(counts.index, counts.values, color=SENTIMENT_PAL, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Trade Count by Bitcoin Market Sentiment (2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment Classification"); ax.set_ylabel("Number of Trades")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_sentiment_distribution")


def plot_win_rate_by_sentiment(df: pd.DataFrame) -> None:
    wr = df.groupby("classification", observed=False)["profitable"].mean().reindex(SENTIMENT_ORDER) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(wr.index, wr.values, color=SENTIMENT_PAL, edgecolor="white")
    for bar, val in zip(bars, wr.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, 55)
    ax.set_title("Win Rate by Market Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Win Rate (%)")
    ax.axvline(wr.mean(), color="gray", linestyle="--", linewidth=1, label=f"Avg {wr.mean():.1f}%")
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
    _save(fig, "02_win_rate_sentiment")


def plot_pnl_distributions(df: pd.DataFrame) -> None:
    sub = df[df["Closed PnL"].between(-5000, 5000)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.violinplot(data=sub, x="classification", y="Closed PnL", hue="classification",
                   order=SENTIMENT_ORDER, palette=SENTIMENT_PAL, inner="quartile", legend=False, ax=axes[0])
    axes[0].set_title("PnL Distribution (Violin)", fontweight="bold")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Sentiment"); axes[0].set_ylabel("Closed PnL (capped ±$5K)")
    sns.boxplot(data=sub, x="classification", y="Closed PnL", hue="classification",
                order=SENTIMENT_ORDER, palette=SENTIMENT_PAL,
                flierprops=dict(marker="o", markersize=2, alpha=0.3), legend=False, ax=axes[1])
    axes[1].set_title("PnL Distribution (Box)", fontweight="bold")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Sentiment"); axes[1].set_ylabel("Closed PnL (capped ±$5K)")
    plt.tight_layout()
    _save(fig, "03_pnl_distributions")


def plot_trade_size_by_sentiment(df: pd.DataFrame) -> None:
    avg_size = df.groupby("classification", observed=False)["Size USD"].mean().reindex(SENTIMENT_ORDER)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(avg_size.index, avg_size.values, color=SENTIMENT_PAL, edgecolor="white")
    for bar, val in zip(bars, avg_size.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"${val:,.0f}", ha="center", fontsize=9)
    ax.set_title("Average Trade Size (USD) by Sentiment — Risk Appetite Signal", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment"); ax.set_ylabel("Avg Trade Size (USD)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "04_trade_size_sentiment")


def plot_buy_sell_by_zone(df: pd.DataFrame) -> None:
    bs = (
        df.groupby(["sentiment_group", "Side"]).size()
        .unstack(fill_value=0)
        .reindex(["Fear Zone", "Neutral", "Greed Zone"])
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bs)); w = 0.35
    ax.bar(x - w/2, bs.get("BUY", 0),  width=w, label="BUY",  color="#378ADD", edgecolor="white")
    ax.bar(x + w/2, bs.get("SELL", 0), width=w, label="SELL", color="#E24B4A", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(bs.index)
    ax.set_title("BUY vs SELL Activity by Sentiment Zone", fontsize=13, fontweight="bold")
    ax.set_ylabel("Trade Count"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "05_buy_sell_zone")


def plot_monthly_pnl(df: pd.DataFrame) -> None:
    monthly = df.groupby("month").agg(
        total_pnl=("Closed PnL", "sum"),
        dominant =("sentiment_group", lambda x: x.mode()[0] if len(x) else "Neutral"),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for _, row in monthly.iterrows():
        ax.bar(row["month_str"], row["total_pnl"],
               color=ZONE_COLORS.get(row["dominant"], "#888780"), alpha=0.85, edgecolor="white")
    ax.plot(monthly["month_str"], monthly["total_pnl"],
            color="black", linewidth=1.5, marker="o", markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Monthly Total PnL (bar color = dominant sentiment zone)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Total PnL (USD)")
    plt.xticks(rotation=45)
    patches = [mpatches.Patch(color=v, label=k) for k, v in ZONE_COLORS.items()]
    ax.legend(handles=patches, fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "06_monthly_pnl")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Spearman Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_correlation_heatmap")


def plot_symbol_pnl_by_zone(df: pd.DataFrame) -> None:
    top_symbols = df.groupby("Coin")["Closed PnL"].sum().nlargest(8).index.tolist()
    sub = df[df["Coin"].isin(top_symbols)]
    grp = sub.groupby(["Coin", "sentiment_group"])["Closed PnL"].sum().reset_index()
    fig = px.bar(grp, x="Coin", y="Closed PnL", color="sentiment_group",
                 barmode="group", color_discrete_map=ZONE_COLORS,
                 title="Top 8 Symbols — PnL by Sentiment Zone",
                 labels={"Closed PnL": "Total PnL (USD)", "sentiment_group": "Zone"})
    fig.update_layout(xaxis_tickangle=-30, template="plotly_white")
    path = OUTPUT_DIR / "08_symbol_pnl_by_zone.html"
    fig.write_html(str(path))
    print(f"  [saved] {path.name}")


def plot_account_scatter(acc_summary: pd.DataFrame) -> None:
    fig = px.scatter(acc_summary, x="trade_count", y="total_pnl",
                     size="avg_size_usd", color="win_rate",
                     hover_data={"Account": True},
                     color_continuous_scale="RdYlGn",
                     title="Account Performance: Trade Frequency vs Total PnL",
                     labels={"trade_count": "Trade Count", "total_pnl": "Total PnL (USD)", "win_rate": "Win Rate"})
    fig.update_layout(template="plotly_white")
    path = OUTPUT_DIR / "09_account_scatter.html"
    fig.write_html(str(path))
    print(f"  [saved] {path.name}")


def plot_feature_importance(feature_importance: pd.Series, model_name: str = "Random Forest") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    fi = feature_importance.sort_values()
    ax.barh(fi.index, fi.values, color="#378ADD", edgecolor="white")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, "10_feature_importance")


def generate_all_charts(df: pd.DataFrame, acc_summary: pd.DataFrame, corr_matrix: pd.DataFrame) -> None:
    print("\n[Generating visualizations...]")
    plot_sentiment_distribution(df)
    plot_win_rate_by_sentiment(df)
    plot_pnl_distributions(df)
    plot_trade_size_by_sentiment(df)
    plot_buy_sell_by_zone(df)
    plot_monthly_pnl(df)
    plot_correlation_heatmap(corr_matrix)
    plot_symbol_pnl_by_zone(df)
    plot_account_scatter(acc_summary)
    print("[Done] All charts saved to outputs/figures/")
