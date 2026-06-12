"""
pipeline.py — End-to-end analysis runner.
Primetrade.ai Sentiment Analysis Project.

Usage:
    python src/pipeline.py
    python src/pipeline.py --skip-ml
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader          import load_sentiment, load_trades, data_quality_report
from preprocessing        import merge_datasets, engineer_features, save_merged
from eda                  import print_full_eda, trader_profitability_summary
from statistical_analysis import print_statistical_summary, correlation_matrix, detect_anomalous_trades
from ml_models            import cluster_traders, build_sentiment_classifier
from visualizations       import generate_all_charts, plot_feature_importance


def main(skip_ml: bool = False) -> None:
    print("\n" + "═"*60)
    print("  PRIMETRADE.AI — SENTIMENT × TRADER BEHAVIOR ANALYSIS")
    print("═"*60)

    print("\n[Phase 1] Loading data...")
    sentiment = load_sentiment()
    trades    = load_trades()
    data_quality_report(sentiment, trades)

    print("\n[Phase 2] Preprocessing & merging...")
    merged = merge_datasets(sentiment, trades)
    merged = engineer_features(merged)
    save_merged(merged)

    print("\n[Phase 3] Exploratory Data Analysis...")
    print_full_eda(merged)

    print("\n[Phase 4] Statistical Analysis...")
    print_statistical_summary(merged)
    corr   = correlation_matrix(merged)
    merged = detect_anomalous_trades(merged)

    feature_importance = None
    if not skip_ml:
        print("\n[Phase 5] Machine Learning...")
        cluster_traders(merged)
        ml_result          = build_sentiment_classifier(merged)
        feature_importance = ml_result["feature_importance"]

    print("\n[Phase 6] Visualizations...")
    acc_summary = trader_profitability_summary(merged)
    generate_all_charts(merged, acc_summary, corr)
    if feature_importance is not None:
        plot_feature_importance(feature_importance)

    print("\n" + "═"*60)
    print("  PIPELINE COMPLETE")
    print("  Charts  → outputs/figures/")
    print("  Dataset → data/merged_dataset.csv")
    print("═"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML models (faster run)")
    args = parser.parse_args()
    main(skip_ml=args.skip_ml)
