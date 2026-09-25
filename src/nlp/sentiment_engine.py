"""
PulseGuard Sentiment Analysis Engine
Covers GTU Unit 4 (NLP & Machine Learning)
Production Hybrid: Fine-Tuned Master ML Model (55k samples) + VADER Lexicon Blend
"""
from typing import Dict, Any, List
import re
import numpy as np
import joblib
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

class SentimentEngine:
    def __init__(self):
        self._vader_analyzer = None
        self._master_model = None
        self._master_vectorizer = None
        self._init_models()

    def _init_models(self):
        """Initialize fine-tuned master model and VADER lexicon."""
        # 1. Load fine-tuned Master ML Model (trained on 55k multi-platform samples)
        models_dir = PROCESSED_DATA_DIR / "models"
        m_path = models_dir / "master_sentiment_model.joblib"
        v_path = models_dir / "master_sentiment_vectorizer.joblib"

        if m_path.exists() and v_path.exists():
            try:
                self._master_model = joblib.load(m_path)
                self._master_vectorizer = joblib.load(v_path)
                print(f"[SentimentEngine] Successfully loaded Fine-Tuned Master ML Model from {m_path.name}")
            except Exception as e:
                print(f"[SentimentEngine] Could not load master model: {e}")

        # 2. Load VADER for compound polarity blend
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            crisis_lexicon = {
                "scam": -3.5, "fraud": -3.8, "boycott": -3.2, "overheating": -2.8,
                "lagging": -2.5, "crashing": -3.0, "worst": -3.1, "broken": -2.9,
                "garbage": -3.4, "pathetic": -3.3, "useless": -3.0, "stolen": -3.5,
                "refund": -1.5, "flawless": 3.2, "masterpiece": 3.5, "smooth": 2.5, "recommended": 2.8
            }
            self._vader_analyzer.lexicon.update(crisis_lexicon)
        except ImportError:
            self._vader_analyzer = None

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using fine-tuned model and continuous polarity blending.
        """
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5, "compound": 0.0}

        # Fine-Tuned Master Model Inference
        if self._master_model is not None and self._master_vectorizer is not None:
            try:
                vec = self._master_vectorizer.transform([text])
                label = self._master_model.predict(vec)[0]
                probs = self._master_model.predict_proba(vec)[0]
                conf = float(np.max(probs))

                # Continuous compound polarity
                if self._vader_analyzer is not None:
                    compound = self._vader_analyzer.polarity_scores(text)["compound"]
                else:
                    compound = conf if label == "POSITIVE" else (-conf if label == "NEGATIVE" else 0.0)

                # High-Precision Consensus Ensemble:
                if abs(compound) <= 0.05:
                    final_label = "NEUTRAL"
                elif compound >= 0.20 and label != "POSITIVE":
                    final_label = "POSITIVE"
                elif compound <= -0.20 and label != "NEGATIVE":
                    final_label = "NEGATIVE"
                else:
                    final_label = label

                return {
                    "label": final_label,
                    "score": round(conf, 4),
                    "compound": round(float(compound), 4)
                }
            except Exception:
                pass

        # Fallback to VADER
        if self._vader_analyzer is not None:
            scores = self._vader_analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "POSITIVE"
                score = round(scores["pos"] + 0.1, 4)
            elif compound <= -0.05:
                label = "NEGATIVE"
                score = round(scores["neg"] + 0.1, 4)
            else:
                label = "NEUTRAL"
                score = round(scores["neu"], 4)
            return {
                "label": label,
                "score": min(1.0, max(0.1, score)),
                "compound": round(compound, 4)
            }

        # Rule heuristic fallback
        return {"label": "NEUTRAL", "score": 0.50, "compound": 0.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(t) for t in texts]
