"""
PulseGuard Probabilistic Distributions Engine
Covers GTU Unit 5 (Probabilistic and Inferential Statistics):
- Random variables, PMF, PDF, CDF
- Discrete Distributions: Binomial & Poisson
- Continuous Distributions: Normal & Exponential
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from src.config import POISSON_LAMBDA_NORMAL, POISSON_LAMBDA_CRISIS

class ProbabilisticEngine:
    def __init__(self):
        pass

    # ==========================================
    # 1. DISCRETE DISTRIBUTIONS (Unit 5)
    # ==========================================

    def poisson_pmf_cdf(self, k_range: range, lam: float) -> Dict[str, Any]:
        """
        Poisson Distribution: Models discrete count of comments arriving per minute.
        Formula: P(X = k) = (lambda^k * e^(-lambda)) / k!
        """
        k_vals = list(k_range)
        pmf_vals = [float(stats.poisson.pmf(k, lam)) for k in k_vals]
        cdf_vals = [float(stats.poisson.cdf(k, lam)) for k in k_vals]

        return {
            "distribution": "Poisson",
            "type": "Discrete",
            "lambda_rate": lam,
            "k_values": k_vals,
            "pmf": [round(p, 5) for p in pmf_vals],
            "cdf": [round(c, 5) for c in cdf_vals],
            "expected_value": lam,
            "variance": lam
        }

    def binomial_pmf_cdf(self, n_trials: int, p_prob: float) -> Dict[str, Any]:
        """
        Binomial Distribution: Models number of negative comments k out of n total comments.
        Formula: P(X = k) = (n choose k) * p^k * (1-p)^(n-k)
        """
        k_vals = list(range(n_trials + 1))
        pmf_vals = [float(stats.binom.pmf(k, n_trials, p_prob)) for k in k_vals]
        cdf_vals = [float(stats.binom.cdf(k, n_trials, p_prob)) for k in k_vals]

        return {
            "distribution": "Binomial",
            "type": "Discrete",
            "n_trials": n_trials,
            "success_probability": p_prob,
            "k_values": k_vals,
            "pmf": [round(p, 5) for p in pmf_vals],
            "cdf": [round(c, 5) for c in cdf_vals],
            "expected_value": round(n_trials * p_prob, 4),
            "variance": round(n_trials * p_prob * (1 - p_prob), 4)
        }

    # ==========================================
    # 2. CONTINUOUS DISTRIBUTIONS (Unit 5)
    # ==========================================

    def normal_pdf_cdf(self, x_values: List[float], mu: float, sigma: float) -> Dict[str, Any]:
        """
        Continuous Normal (Gaussian) Distribution: Models standardized sentiment velocity scores.
        Formula: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
        """
        if sigma <= 0:
            sigma = 0.01

        pdf_vals = [float(stats.norm.pdf(x, loc=mu, scale=sigma)) for x in x_values]
        cdf_vals = [float(stats.norm.cdf(x, loc=mu, scale=sigma)) for x in x_values]

        return {
            "distribution": "Normal",
            "type": "Continuous",
            "mean_mu": round(mu, 4),
            "std_sigma": round(sigma, 4),
            "x_values": [round(x, 4) for x in x_values],
            "pdf": [round(p, 5) for p in pdf_vals],
            "cdf": [round(c, 5) for c in cdf_vals]
        }

    def exponential_pdf_cdf(self, x_times: List[float], scale_beta: float) -> Dict[str, Any]:
        """
        Continuous Exponential Distribution: Models inter-arrival time between consecutive toxic comments.
        Formula: f(x) = (1/beta) * exp(-x/beta)
        """
        if scale_beta <= 0:
            scale_beta = 1.0

        pdf_vals = [float(stats.expon.pdf(x, scale=scale_beta)) for x in x_times]
        cdf_vals = [float(stats.expon.cdf(x, scale=scale_beta)) for x in x_times]

        return {
            "distribution": "Exponential",
            "type": "Continuous",
            "scale_beta": round(scale_beta, 4),
            "rate_lambda": round(1.0 / scale_beta, 4),
            "x_times": [round(x, 4) for x in x_times],
            "pdf": [round(p, 5) for p in pdf_vals],
            "cdf": [round(c, 5) for c in cdf_vals]
        }
