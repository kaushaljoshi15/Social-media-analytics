"""
PulseGuard Anomaly & Crisis Velocity Detector
Combines EWMA, Rolling Window Z-Score, and Inferential Hypothesis Testing
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.config import (
    BASELINE_NEGATIVE_RATIO,
    CRISIS_Z_THRESHOLD,
    CRITICAL_Z_THRESHOLD,
    SLIDING_WINDOW_SIZE
)
from src.analytics.inferential_stats import InferentialStatsEngine

class AnomalyDetector:
    def __init__(
        self,
        baseline_ratio: float = BASELINE_NEGATIVE_RATIO,
        window_size: int = SLIDING_WINDOW_SIZE
    ):
        self.baseline_ratio = baseline_ratio
        self.window_size = window_size
        self.inferential_engine = InferentialStatsEngine()
        # Historical baseline variance (calibrated)
        self.baseline_std = 0.08

    def evaluate_window(self, recent_comments_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate rolling time window of comments for crisis backlash anomalies.
        """
        if recent_comments_df.empty:
            return {
                "alert_level": "SAFE",
                "status_color": "#10B981",
                "negative_ratio": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "sample_size": 0,
                "message": "No comments in buffer."
            }

        total_count = len(recent_comments_df)
        neg_count = (recent_comments_df["sentiment"] == "NEGATIVE").sum()
        current_negative_velocity = float(neg_count / total_count)

        # Slice window to discrete micro-batches (e.g. chunks of 5) to compute sample variance
        batch_size = max(3, total_count // 6)
        sub_ratios = []
        for i in range(0, total_count, batch_size):
            chunk = recent_comments_df.iloc[i : i + batch_size]
            if not chunk.empty:
                c_neg = (chunk["sentiment"] == "NEGATIVE").sum()
                sub_ratios.append(float(c_neg / len(chunk)))

        # Run GTU Inferential Hypothesis Testing (Unit 5)
        hyp_res = self.inferential_engine.run_one_sample_hypothesis_test(
            sample_ratios=sub_ratios,
            mu_0=self.baseline_ratio
        )

        # Compute dynamic Z-score
        effective_std = float(np.std(sub_ratios, ddof=1)) if len(sub_ratios) > 1 else self.baseline_std
        if effective_std < 0.02:
            effective_std = self.baseline_std

        z_score = float((current_negative_velocity - self.baseline_ratio) / effective_std)

        # Determine Alert Level
        if z_score >= CRITICAL_Z_THRESHOLD or (hyp_res.get("reject_null_H0") and current_negative_velocity >= 0.40):
            alert_level = "CRITICAL"
            color = "#EF4444"
            msg = f"EMERGENCY: Statistically abnormal crisis backlash detected! (Z={z_score:.2f}, p={hyp_res.get('p_value', 0):.4f})"
        elif z_score >= CRISIS_Z_THRESHOLD or current_negative_velocity >= 0.25:
            alert_level = "ELEVATED"
            color = "#F59E0B"
            msg = f"WARNING: Negative sentiment velocity rising above threshold (Z={z_score:.2f})"
        else:
            alert_level = "SAFE"
            color = "#10B981"
            msg = f"Normal brand sentiment velocity (Z={z_score:.2f})"

        return {
            "alert_level": alert_level,
            "status_color": color,
            "negative_velocity": round(current_negative_velocity, 4),
            "negative_count": int(neg_count),
            "total_comments": int(total_count),
            "z_score": round(z_score, 3),
            "p_value": hyp_res.get("p_value", 1.0),
            "reject_null_hypothesis": hyp_res.get("reject_null_H0", False),
            "confidence_interval_95": hyp_res.get("confidence_interval_95", [0.0, 0.0]),
            "message": msg
        }
