import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from src.analytics.descriptive_stats import DescriptiveStatsEngine
from src.analytics.probabilistic_engine import ProbabilisticEngine
from src.analytics.inferential_stats import InferentialStatsEngine
from src.analytics.root_cause_miner import RootCauseMiner
from src.nlp.ml_benchmarks import MLBenchmarkPipeline

print("[Test] Testing Descriptive Stats...")
df = pd.read_csv("data/raw/sample_crisis_dataset.csv")
d = DescriptiveStatsEngine().compute_univariate_stats(df["sentiment_score"])
print(f" -> Mean: {d['mean']}, Skewness: {d['skewness']}, Kurtosis: {d['kurtosis']}")

print("[Test] Testing Probabilistic Engine...")
p = ProbabilisticEngine().poisson_pmf_cdf(range(10), 3.5)
print(f" -> Poisson PMF items: {len(p['pmf'])}, Sum of PMF: {sum(p['pmf']):.3f}")

print("[Test] Testing Inferential Stats (t-test)...")
inf = InferentialStatsEngine().run_one_sample_hypothesis_test([0.7, 0.8, 0.75, 0.9, 0.65])
print(f" -> t-stat: {inf['t_statistic']}, p-val: {inf['p_value']}, Verdict: {inf['crisis_verdict']}")

print("[Test] Testing Root Cause Miner...")
neg_texts = df[df["sentiment"] == "NEGATIVE"]["clean_text"].tolist()
rc = RootCauseMiner().extract_top_culprits(neg_texts, top_n=4)
print(f" -> Top Culprit terms: {[c['term'] for c in rc]}")

print("[Test] Testing Scikit-Learn ML Benchmarks (Unit 4)...")
ml = MLBenchmarkPipeline()
res = ml.train_and_evaluate(df)
print(f" -> Naive Bayes Acc: {res['naive_bayes']['accuracy']}, Logistic Reg Acc: {res['logistic_regression']['accuracy']}")

print("\n*** ALL GTU ENGINES PASSED SUCCESSFULLY! ***")
