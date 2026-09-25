"""
PulseGuard Descriptive Statistics Engine
Covers GTU Unit 3 (Descriptive Statistics: Measures of Central Tendency,
Dispersion/Variation, Measures of Location, Shape & Symmetry: Skewness & Kurtosis)
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

class DescriptiveStatsEngine:
    def __init__(self):
        pass

    def compute_univariate_stats(self, series: pd.Series, name: str = "Metric") -> Dict[str, Any]:
        """
        Compute full GTU Unit 3 descriptive statistics on a numerical column:
        - Central Tendency: Mean, Median, Mode
        - Dispersion: Variance, Std Dev, Range, IQR
        - Measures of Location: Q1 (25%), Q2 (50%), Q3 (75%), 90th Percentile
        - Shape & Symmetry: Fisher-Pearson Skewness, Kurtosis
        """
        clean_s = series.dropna()
        if len(clean_s) == 0:
            return {"name": name, "count": 0}

        arr = clean_s.to_numpy()
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))

        # Mode calculation (SciPy)
        try:
            mode_res = stats.mode(arr, keepdims=True)
            mode_val = float(mode_res.mode[0])
        except Exception:
            mode_val = mean_val

        # Measures of Dispersion (Unit 3)
        var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        range_val = max_val - min_val

        # Measures of Location (Quartiles & Percentiles)
        q1 = float(np.percentile(arr, 25))
        q2 = median_val
        q3 = float(np.percentile(arr, 75))
        iqr_val = q3 - q1
        p90 = float(np.percentile(arr, 90))

        # Measures of Shape and Symmetry (Unit 3)
        skew_val = float(stats.skew(arr)) if len(arr) > 2 else 0.0
        kurt_val = float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0

        # Interpretation of shape
        if abs(skew_val) < 0.5:
            skew_label = "Fairly Symmetrical (Normal shape)"
        elif skew_val >= 0.5:
            skew_label = "Positively Skewed (Right-skewed tail: high backlash/toxicity outliers)"
        else:
            skew_label = "Negatively Skewed (Left-skewed tail)"

        if kurt_val > 1.0:
            kurt_label = "Leptokurtic (Heavy-tailed / High Outlier Risk)"
        elif kurt_val < -1.0:
            kurt_label = "Platykurtic (Flat distribution)"
        else:
            kurt_label = "Mesokurtic (Near-Normal distribution)"

        return {
            "name": name,
            "count": len(arr),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "mode": round(mode_val, 4),
            "variance": round(var_val, 4),
            "std_dev": round(std_val, 4),
            "range": round(range_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "q1": round(q1, 4),
            "q2_median": round(q2, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr_val, 4),
            "percentile_90": round(p90, 4),
            "skewness": round(skew_val, 4),
            "skewness_interpretation": skew_label,
            "kurtosis": round(kurt_val, 4),
            "kurtosis_interpretation": kurt_label
        }

    def compute_bivariate_analysis(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Bivariate analysis between two numerical features (GTU Unit 3):
        Calculates Pearson correlation coefficient and Covariance.
        """
        clean_df = df[[col1, col2]].dropna()
        if len(clean_df) < 2:
            return {"correlation": 0.0, "covariance": 0.0}

        cov_matrix = np.cov(clean_df[col1], clean_df[col2])
        covariance = float(cov_matrix[0, 1])

        corr, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])

        return {
            "feature_x": col1,
            "feature_y": col2,
            "sample_size": len(clean_df),
            "covariance": round(covariance, 4),
            "pearson_correlation": round(float(corr), 4),
            "p_value": round(float(p_val), 6)
        }
