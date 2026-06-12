"""
config.py — Project-wide constants, paths, and color mappings.
Primetrade.ai Sentiment Analysis Project.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "reports"

RAW_SENTIMENT  = DATA_DIR / "fear_greed_index.csv"
RAW_TRADES     = DATA_DIR / "historical_data.csv"
MERGED_DATASET = DATA_DIR / "merged_dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Sentiment Mappings ───────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

SENTIMENT_COLORS = {
    "Extreme Fear":  "#E24B4A",
    "Fear":          "#EF9F27",
    "Neutral":       "#888780",
    "Greed":         "#1D9E75",
    "Extreme Greed": "#3B6D11",
}

ZONE_COLORS = {
    "Fear Zone":  "#EF9F27",
    "Neutral":    "#888780",
    "Greed Zone": "#1D9E75",
}

SENTIMENT_TO_ZONE = {
    "Extreme Fear":  "Fear Zone",
    "Fear":          "Fear Zone",
    "Neutral":       "Neutral",
    "Greed":         "Greed Zone",
    "Extreme Greed": "Greed Zone",
}

# ─── Plot Style ───────────────────────────────────────────────────────────────
FIGSIZE_WIDE   = (14, 5)
FIGSIZE_SQUARE = (8, 6)
FIGSIZE_TALL   = (10, 8)
DPI            = 150
PLOT_STYLE     = "whitegrid"
