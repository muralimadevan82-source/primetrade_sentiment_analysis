"""
ml_models.py — Trader clustering and sentiment impact classification.
Primetrade.ai Sentiment Analysis Project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


def cluster_traders(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """K-Means clustering of trader behavior profiles."""
    acc = df.groupby("Account").agg(
        total_pnl   =("Closed PnL", "sum"),
        trade_count =("Closed PnL", "count"),
        win_rate    =("profitable",  "mean"),
        loss_rate   =("loss",        "mean"),
        avg_size    =("Size USD",    "mean"),
        avg_fee_pct =("fee_pct",     "mean"),
    ).reset_index()

    features = ["total_pnl", "trade_count", "win_rate", "loss_rate", "avg_size", "avg_fee_pct"]
    X = acc[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    acc["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_pnl = acc.groupby("cluster")["total_pnl"].mean().rank(ascending=False).astype(int)
    acc["trader_tier"] = acc["cluster"].map({c: f"Tier {r}" for c, r in cluster_pnl.items()})

    print("\n[Trader Clusters — Mean Features per Tier]")
    print(acc.groupby("trader_tier")[features].mean().round(2).to_string())
    return acc


def build_sentiment_classifier(df: pd.DataFrame) -> dict:
    """Random Forest: predict trade profitability from sentiment + trade features."""
    sub = df[df["Closed PnL"] != 0].copy()

    le = LabelEncoder()
    sub["sentiment_enc"] = le.fit_transform(sub["classification"].astype(str))
    sub["side_enc"]      = (sub["Side"] == "BUY").astype(int)

    feature_cols = ["value", "sentiment_enc", "log_size_usd", "side_enc", "hour"]
    X = sub[feature_cols].fillna(0)
    y = sub["profitable"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    report    = classification_report(y_test, y_pred, output_dict=True)

    feature_importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("\n[Sentiment Classifier — Random Forest]")
    print(f"  CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {report['accuracy']:.4f}")
    print(f"\n  Feature Importance:\n{feature_importance.round(4).to_string()}")

    return {
        "model"              : model,
        "feature_importance" : feature_importance,
        "cv_roc_auc"         : cv_scores.mean(),
        "classification_report": report,
        "feature_cols"       : feature_cols,
    }
