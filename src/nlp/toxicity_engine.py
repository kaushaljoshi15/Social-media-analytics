"""
PulseGuard Toxicity & Harassment Filter
Covers GTU Unit 4 (NLP Classification & Multi-Label Detection)
Powered by Fine-Tuned Master ML Model (Davidson 24k Dataset: 95.1% Accuracy) + Severe Threat Lexicon
"""
import re
from typing import Dict, Any, List
import numpy as np
import joblib
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

class ToxicityEngine:
    def __init__(self):
        self._master_model = None
        self._master_vectorizer = None
        self._init_model()

        # Severe threat keyword overrides
        self.severe_toxic_words = {
            "kill", "die", "murder", "threat", "burn", "shoot", "attack",
            "hang", "lynch", "destroy you", "suicide"
        }
        self.profane_words = {
            "fuck", "fucking", "shit", "bitch", "bastard", "asshole",
            "crap", "idiot", "moron", "dumbass", "scum", "trash"
        }

    def _init_model(self):
        """Load fine-tuned Davidson Toxicity Model (95.1% accuracy)."""
        models_dir = PROCESSED_DATA_DIR / "models"
        m_path = models_dir / "master_toxicity_model.joblib"
        v_path = models_dir / "master_toxicity_vectorizer.joblib"

        if m_path.exists() and v_path.exists():
            try:
                self._master_model = joblib.load(m_path)
                self._master_vectorizer = joblib.load(v_path)
                print(f"[ToxicityEngine] Loaded Fine-Tuned Master Toxicity Model from {m_path.name}")
            except Exception as e:
                print(f"[ToxicityEngine] Could not load toxicity model: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze comment for toxic harassment, profanity, and threats.
        """
        if not text or len(text.strip()) == 0:
            return {"is_toxic": False, "toxicity_score": 0.0, "categories": []}

        lower_text = text.lower()
        words = set(re.findall(r'\b\w+\b', lower_text))

        categories = []
        rule_score = 0.0

        if words.intersection(self.severe_toxic_words):
            categories.append("THREAT_SEVERE")
            rule_score += 0.70

        if words.intersection(self.profane_words):
            categories.append("PROFANITY")
            rule_score += 0.45

        # ML Model Inference
        ml_score = 0.0
        if self._master_model is not None and self._master_vectorizer is not None:
            try:
                vec = self._master_vectorizer.transform([text])
                prob_toxic = float(self._master_model.predict_proba(vec)[0][1])
                ml_score = prob_toxic
            except Exception:
                ml_score = 0.0

        final_score = max(rule_score, ml_score)
        final_score = min(1.0, round(final_score, 4))
        is_toxic = bool(rule_score >= 0.40 or ml_score >= 0.65)

        if is_toxic and not categories:
            categories.append("TOXIC_LANGUAGE")

        return {
            "is_toxic": is_toxic,
            "toxicity_score": final_score,
            "categories": list(set(categories))
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(t) for t in texts]
