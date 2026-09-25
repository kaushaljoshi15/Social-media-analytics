"""
PulseGuard Inferential Statistics & Hypothesis Testing Engine
Covers GTU Unit 5 (Inferential Statistics):
- Point Estimation & Interval Estimation (95% & 99% Confidence Intervals)
- Central Limit Theorem (CLT) Simulation
- Hypothesis Testing: One-Sample t-test / Z-test (p-value and critical region)
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from src.config import BASELINE_NEGATIVE_RATIO, HYPOTHESIS_ALPHA

class InferentialStatsEngine:
    def __init__(self):
        self.baseline_mu0 = BASELINE_NEGATIVE_RATIO

    def run_one_sample_hypothesis_test(
        self,
        sample_ratios: List[float],
        mu_0: float = BASELINE_NEGATIVE_RATIO,
        alpha: float = HYPOTHESIS_ALPHA
    ) -> Dict[str, Any]:
        """
        One-Sample t-test (GTU Unit 5 Hypothesis Testing):
        - H0: Current negative ratio mean mu = baseline mu_0 (Normal Brand State)
        - H1: Current negative ratio mean mu > baseline mu_0 (Crisis Anomaly)
        """
        arr = np.array(sample_ratios, dtype=float)
        n = len(arr)

        if n < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_size": n,
                "p_value": 1.0,
                "is_crisis": False
            }

        sample_mean = float(np.mean(arr))
        sample_std = float(np.std(arr, ddof=1)) if n > 1 else 0.001
        standard_error = sample_std / np.sqrt(n)

        # One-Sample t-test (Right-tailed test: alternative='greater')
        t_stat, p_val_two_sided = stats.ttest_1samp(arr, mu_0)
        # Convert to one-tailed p-value for greater
        if t_stat > 0:
            p_val = p_val_two_sided / 2.0
        else:
            p_val = 1.0 - (p_val_two_sided / 2.0)

        # Critical value at alpha
        df = n - 1
        t_critical = float(stats.t.ppf(1.0 - alpha, df))

        # Hypothesis Decision
        reject_null = bool((p_val < alpha) and (t_stat > 0))

        # Interval Estimation (95% and 99% Confidence Intervals) (Unit 5)
        ci_95 = stats.t.interval(0.95, df, loc=sample_mean, scale=standard_error)
        ci_99 = stats.t.interval(0.99, df, loc=sample_mean, scale=standard_error)

        return {
            "null_hypothesis_H0": f"mu = {mu_0:.4f} (Brand is stable)",
            "alt_hypothesis_H1": f"mu > {mu_0:.4f} (Brand is under crisis backlash)",
            "sample_size_n": n,
            "sample_mean_xbar": round(sample_mean, 4),
            "sample_std_s": round(sample_std, 4),
            "standard_error_se": round(standard_error, 4),
            "degrees_of_freedom": df,
            "t_statistic": round(float(t_stat), 4),
            "t_critical": round(t_critical, 4),
            "alpha_significance": alpha,
            "p_value": round(float(p_val), 6),
            "reject_null_H0": reject_null,
            "crisis_verdict": "CRISIS DETECTED (Statistically Significant)" if reject_null else "NORMAL (Fail to reject H0)",
            "confidence_interval_95": [round(float(ci_95[0]), 4), round(float(ci_95[1]), 4)],
            "confidence_interval_99": [round(float(ci_99[0]), 4), round(float(ci_99[1]), 4)]
        }

    def simulate_central_limit_theorem(
        self,
        population_scores: List[float],
        sample_size: int = 30,
        num_simulations: int = 500
    ) -> Dict[str, Any]:
        """
        Central Limit Theorem (CLT) Demonstration (GTU Unit 5):
        Takes random samples from any raw population and plots how the sample mean
        distribution converges to a standard Normal bell curve.
        """
        if len(population_scores) < 10:
            # Generate synthetic non-normal population (e.g. exponential or bimodal)
            pop = np.concatenate([np.random.exponential(scale=2.0, size=500), np.random.uniform(0, 10, size=500)])
        else:
            pop = np.array(population_scores)

        pop_mean = float(np.mean(pop))
        pop_std = float(np.std(pop))

        # Draw random samples and compute sample means
        sample_means = []
        for _ in range(num_simulations):
            sample = np.random.choice(pop, size=sample_size, replace=True)
            sample_means.append(float(np.mean(sample)))

        clt_mean = float(np.mean(sample_means))
        clt_std = float(np.std(sample_means))
        theoretical_se = pop_std / np.sqrt(sample_size)

        return {
            "population_mean": round(pop_mean, 4),
            "population_std": round(pop_std, 4),
            "sample_size_n": sample_size,
            "simulations_count": num_simulations,
            "clt_sample_means": [round(m, 4) for m in sample_means],
            "empirical_mean_of_means": round(clt_mean, 4),
            "empirical_std_error": round(clt_std, 4),
            "theoretical_std_error": round(theoretical_se, 4),
            "clt_holds": bool(abs(clt_mean - pop_mean) < 0.2)
        }
