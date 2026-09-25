"""
PulseGuard Data Preparation & Wrangling Engine
Covers GTU Unit 6 (Data Preparation: Slicing & Dicing, Filtering, Sorting & Shuffling,
Aggregating, Handling Outliers with IQR & Z-score, Data Normalization)
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

class DataWrangler:
    def __init__(self):
        pass

    # ==========================================
    # 1. OUTLIER DETECTION (Unit 6)
    # ==========================================

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Detect outliers using Interquartile Range (IQR) rule:
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        """
        if df.empty or column not in df.columns:
            return df, pd.DataFrame(), {}

        series = df[column].dropna()
        q1 = float(np.percentile(series, 25))
        q3 = float(np.percentile(series, 75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outliers_df = df[outliers_mask].copy()
        clean_df = df[~outliers_mask].copy()

        bounds = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": int(outliers_mask.sum())
        }
        return clean_df, outliers_df, bounds

    def detect_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect outliers using standard Z-score rule: |Z| > threshold."""
        if df.empty or column not in df.columns:
            return df, pd.DataFrame()

        series = df[column].dropna()
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            return df, pd.DataFrame()

        z_scores = np.abs((df[column] - mean) / std)
        outliers_mask = z_scores > threshold

        return df[~outliers_mask].copy(), df[outliers_mask].copy()

    # ==========================================
    # 2. DATA NORMALIZATION (Unit 6)
    # ==========================================

    def min_max_scale(self, series: pd.Series) -> pd.Series:
        """Min-Max Normalization: scales values to range [0, 1]."""
        arr = series.to_numpy(dtype=float)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val == min_val:
            return pd.Series(np.zeros_like(arr), index=series.index)
        scaled = (arr - min_val) / (max_val - min_val)
        return pd.Series(scaled, index=series.index)

    def standard_scale(self, series: pd.Series) -> pd.Series:
        """Standard Normalization (Z-score scaling): mean=0, std=1."""
        arr = series.to_numpy(dtype=float)
        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr)
        if std_val == 0:
            return pd.Series(np.zeros_like(arr), index=series.index)
        scaled = (arr - mean_val) / std_val
        return pd.Series(scaled, index=series.index)

    # ==========================================
    # 3. SLICING, DICING & AGGREGATIONS (Unit 6)
    # ==========================================

    def slice_and_aggregate_by_platform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multi-platform metrics (GTU Unit 6):
        Groups by platform and computes volume, negative ratio, toxicity count, mean sentiment.
        """
        if df.empty or "platform" not in df.columns:
            return pd.DataFrame()

        agg = df.groupby("platform").agg(
            total_comments=("id", "count"),
            negative_count=("sentiment", lambda s: (s == "NEGATIVE").sum()),
            positive_count=("sentiment", lambda s: (s == "POSITIVE").sum()),
            toxic_count=("is_toxic", "sum"),
            avg_sentiment_score=("sentiment_score", "mean"),
            avg_toxicity_score=("toxicity_score", "mean")
        ).reset_index()

        agg["negative_ratio"] = (agg["negative_count"] / agg["total_comments"]).round(4)
        agg["toxic_ratio"] = (agg["toxic_count"] / agg["total_comments"]).round(4)
        agg["avg_sentiment_score"] = agg["avg_sentiment_score"].round(4)
        agg["avg_toxicity_score"] = agg["avg_toxicity_score"].round(4)

        return agg.sort_values(by="negative_ratio", ascending=False)
