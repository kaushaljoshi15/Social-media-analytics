# PulseGuard AI: Autonomous Brand Crisis & Social Media Intelligence Engine
Enterprise-Grade Real-Time Anomaly Detection, Threat Radar & NLP Platform

---

## 🌟 Executive Overview
**PulseGuard AI** is an enterprise-grade social intelligence platform engineered to monitor real-time public interactions across **YouTube, Twitter / X, Reddit, and Telegram**. 

Rather than passive counting of likes and generic positive/negative percentages, PulseGuard AI operates an **autonomous early-warning system**:
1. **Dynamic Negative Velocity:** Tracks exponential moving average (EWMA) rates of customer backlash over rolling time windows.
2. **Statistical Anomaly Detection:** Applies real-time **One-Sample Hypothesis Testing ($t$-test, $p < 0.01$)** and **Poisson arrival rate modeling** to statistically differentiate authentic brand crises from normal stochastic noise.
3. **Automated Root-Cause Attribution:** Uses **TF-IDF Bi-gram Vectorization + K-Means Clustering** to isolate the exact operational, product, or software failures driving customer frustration.
4. **Threat & Toxicity Shield:** Separates constructive consumer bug reports from severe toxicity, harassment, and bot flood attacks.

---

## 🏗️ System Architecture

```
[Ingestion Engine]                 [Dual-Stage NLP Engine]             [Statistical Anomaly Engine]            [Executive Dashboard]
 ┌────────────────┐                 ┌──────────────────────┐            ┌─────────────────────────┐             ┌─────────────────────┐
 │ YouTube API v3 │──┐              │ Preprocessor & Emoji │            │ 12-Event Sliding Window │             │ Real-Time Ops       │
 ├────────────────┤  │ Multi-       ├──────────────────────┤            ├─────────────────────────┤             ├─────────────────────┤
 │ Twitter / X    │──┼─────────────►│ Sentiment Polarity   │───────────►│ EWMA Negative Velocity  │────────────►│ Deep Target Audit   │
 ├────────────────┤  │ Channel      │ (VADER + DistilBERT) │            ├─────────────────────────┤             ├─────────────────────┤
 │ Reddit PRAW    │──┼─────────────►├──────────────────────┤            │ One-Sample t-Test       │             │ Root-Cause Clusters │
 ├────────────────┤  │ Stream       │ Multi-Label Toxicity │            │ (p < 0.01 Significance) │             │ & AI Briefs         │
 │ Telegram Bot   │──┘              │ (Threat / Profanity) │            └─────────────────────────┘             └─────────────────────┘
 └────────────────┘                 └──────────────────────┘                         │
                                                                                     ▼
                                                                        ┌─────────────────────────┐
                                                                        │ Root-Cause Attribution  │
                                                                        │ (TF-IDF + K-Means)      │
                                                                        └─────────────────────────┘
```

---

## 🚀 Quickstart & Execution

### 1. Launch the Application
```powershell
streamlit run app.py
```
Open your browser at: `http://localhost:8501`

### 2. Run Automated Verification Tests
```powershell
python test_pipeline.py
```

### 3. Model Fine-Tuning on Custom Datasets
```powershell
python src/nlp/fine_tuner.py --data data/raw/sample_crisis_dataset.csv --model logistic_regression
```

---

## 📊 Core Platform Capabilities

1. **⚡ Real-Time Operations:** Continuous backlash velocity graphs with threshold bands, live processed feed with thread-safe persistence, and active friction signals.
2. **🎯 Deep Target Inspector:** Audit any specific YouTube Video link, Twitter/X handle, Reddit post, or Telegram channel on demand. Generates an instant **AI Executive Intelligence Brief**, sentiment donut breakdown, and culprit keywords.
3. **📈 Anomaly & Predictive Intelligence:** Poisson comment arrival rate modeler ($\lambda$), One-Sample hypothesis test significance gauges ($t$-statistic, $p$-value), and sampling variance bounds.
4. **🧠 Root-Cause Attribution Matrix:** Unsupervised K-Means thematic vector cards and supervised model cross-validation benchmarks (Multinomial Naive Bayes vs. Logistic Regression).
5. **📋 Executive Intelligence Dossier:** One-click generation and export of C-level brand health audit reports.

---

## 🔬 Mathematical & Data Science Foundations

- **Poisson Process Arrival Modeling:** $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ (models message influx under normal vs. crisis states).
- **Inferential Hypothesis Testing:** $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$ testing $H_0: \mu = \mu_0$ vs $H_1: \mu > \mu_0$ with $\alpha = 0.01$.
- **IQR Outlier Detection:** Identifies viral engagement spikes and coordinated bot brigading outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
- **Aspect Extraction:** TF-IDF with bi-gram range $(1, 2)$ to extract exact culprit terms (e.g., `"battery drain"`, `"server timeout"`, `"refund delay"`).
