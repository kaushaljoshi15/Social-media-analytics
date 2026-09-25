"""
PulseGuard GPU-Powered Toxicity & Threat Detection Engine
══════════════════════════════════════════════════════════
Drop-in replacement for the scikit-learn ToxicityEngine.
Uses fine-tuned RoBERTa-large (or DistilBERT) with GPU inference.
Falls back to the original rule-based + scikit-learn pipeline if no GPU model is available.
"""
import re
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import PROCESSED_DATA_DIR

GPU_MODELS_DIR = PROCESSED_DATA_DIR / "gpu_models"


class GPUToxicityEngine:
    """
    GPU-Accelerated Toxicity Detection Engine with automatic fallback.

    Priority:
    1. Fine-tuned Transformer (GPU) — if available
    2. Original ToxicityEngine (rules + scikit-learn) — fallback
    """

    def __init__(self, model_type="roberta-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        self._fallback_engine = None

        # Severe threat keyword overrides (kept even with GPU model for safety)
        self.severe_toxic_words = {
            "kill", "die", "murder", "threat", "burn", "shoot", "attack",
            "hang", "lynch", "destroy you", "suicide"
        }

        self._load_gpu_model(model_type)

    def _load_gpu_model(self, model_type):
        """Attempt to load the fine-tuned GPU toxicity model."""
        model_dir = GPU_MODELS_DIR / f"toxicity_{model_type}"

        if not model_dir.exists():
            for alt in ["roberta-large", "distilbert", "deberta-large"]:
                alt_dir = GPU_MODELS_DIR / f"toxicity_{alt}"
                if alt_dir.exists():
                    model_dir = alt_dir
                    break

        metadata_path = model_dir / "training_metadata.json"

        if model_dir.exists() and metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.model.to(self.device)
                self.model.eval()

                self.max_length = metadata.get("config", {}).get("max_length", 128)
                val_f1 = metadata.get("best_val_f1", 0)

                print(f"[GPUToxicityEngine] ✓ Loaded fine-tuned model from {model_dir.name}")
                print(f"[GPUToxicityEngine]   Val F1: {val_f1 * 100:.2f}% | Device: {self.device}")
                return

            except Exception as e:
                print(f"[GPUToxicityEngine] ✗ Failed to load GPU model: {e}")

        # Fallback to original engine
        print("[GPUToxicityEngine] No GPU model found. Falling back to rules + scikit-learn.")
        try:
            from src.nlp.toxicity_engine import ToxicityEngine
            self._fallback_engine = ToxicityEngine()
        except Exception as e:
            print(f"[GPUToxicityEngine] ✗ Fallback also failed: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single text for toxicity and threats."""
        if not text or len(text.strip()) == 0:
            return {"is_toxic": False, "toxicity_score": 0.0, "categories": []}

        lower_text = text.lower()
        words = set(re.findall(r'\b\w+\b', lower_text))

        # Always check severe threat keywords (safety override)
        categories = []
        rule_score = 0.0

        if words.intersection(self.severe_toxic_words):
            categories.append("THREAT_SEVERE")
            rule_score = 0.90

        # GPU Transformer inference
        ml_score = 0.0
        if self.model is not None and self.tokenizer is not None:
            try:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                # Binary: index 1 = toxic
                ml_score = float(probs[1]) if len(probs) > 1 else 0.0

            except Exception:
                ml_score = 0.0

        elif self._fallback_engine is not None:
            # Use fallback engine
            result = self._fallback_engine.analyze(text)
            return result

        # Combine rule-based and ML scores
        final_score = max(rule_score, ml_score)
        final_score = min(1.0, round(final_score, 4))
        is_toxic = bool(rule_score >= 0.40 or ml_score >= 0.50)

        if is_toxic and not categories:
            if ml_score >= 0.80:
                categories.append("TOXIC_HIGH_CONFIDENCE")
            else:
                categories.append("TOXIC_LANGUAGE")

        return {
            "is_toxic": is_toxic,
            "toxicity_score": final_score,
            "categories": list(set(categories)),
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 128) -> List[Dict[str, Any]]:
        """
        Batch toxicity inference with GPU.
        """
        if self.model is None or self.tokenizer is None:
            return [self.analyze(t) for t in texts]

        results = []
        self.model.eval()

        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t else "" for t in texts[i:i + batch_size]]

            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for j, (prob, text) in enumerate(zip(probs, batch_texts)):
                ml_score = float(prob[1]) if len(prob) > 1 else 0.0

                # Rule-based overrides
                lower_text = text.lower()
                words_set = set(re.findall(r'\b\w+\b', lower_text))
                categories = []
                rule_score = 0.0

                if words_set.intersection(self.severe_toxic_words):
                    categories.append("THREAT_SEVERE")
                    rule_score = 0.90

                final_score = max(rule_score, ml_score)
                final_score = min(1.0, round(final_score, 4))
                is_toxic = bool(rule_score >= 0.40 or ml_score >= 0.50)

                if is_toxic and not categories:
                    if ml_score >= 0.80:
                        categories.append("TOXIC_HIGH_CONFIDENCE")
                    else:
                        categories.append("TOXIC_LANGUAGE")

                results.append({
                    "is_toxic": is_toxic,
                    "toxicity_score": final_score,
                    "categories": list(set(categories)),
                })

        return results
