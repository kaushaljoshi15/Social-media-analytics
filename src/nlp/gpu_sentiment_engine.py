"""
PulseGuard GPU-Powered Sentiment Inference Engine
══════════════════════════════════════════════════
Drop-in replacement for the scikit-learn SentimentEngine.
Uses fine-tuned RoBERTa-large (or DistilBERT) with GPU inference.
Falls back to the original VADER + scikit-learn pipeline if no GPU model is available.
"""
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import PROCESSED_DATA_DIR

GPU_MODELS_DIR = PROCESSED_DATA_DIR / "gpu_models"


class GPUSentimentEngine:
    """
    GPU-Accelerated Sentiment Engine with automatic fallback.

    Priority:
    1. Fine-tuned Transformer (GPU) — if available
    2. Original SentimentEngine (VADER + scikit-learn) — fallback
    """

    def __init__(self, model_type="roberta-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.reverse_label_map = None
        self.max_length = 128
        self._fallback_engine = None

        self._load_gpu_model(model_type)

    def _load_gpu_model(self, model_type):
        """Attempt to load the fine-tuned GPU model."""
        model_dir = GPU_MODELS_DIR / f"sentiment_{model_type}"

        if not model_dir.exists():
            # Try alternate model types
            for alt in ["roberta-large", "distilbert", "deberta-large"]:
                alt_dir = GPU_MODELS_DIR / f"sentiment_{alt}"
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

                self.reverse_label_map = {
                    int(k): v for k, v in metadata.get("reverse_label_map", {}).items()
                }
                self.max_length = metadata.get("config", {}).get("max_length", 128)

                val_f1 = metadata.get("best_val_f1", 0)
                print(f"[GPUSentimentEngine] ✓ Loaded fine-tuned model from {model_dir.name}")
                print(f"[GPUSentimentEngine]   Val F1: {val_f1 * 100:.2f}% | Device: {self.device}")
                return

            except Exception as e:
                print(f"[GPUSentimentEngine] ✗ Failed to load GPU model: {e}")

        # Fallback to original engine
        print("[GPUSentimentEngine] No GPU model found. Falling back to VADER + scikit-learn.")
        try:
            from src.nlp.sentiment_engine import SentimentEngine
            self._fallback_engine = SentimentEngine()
        except Exception as e:
            print(f"[GPUSentimentEngine] ✗ Fallback also failed: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5, "compound": 0.0}

        # GPU Transformer inference
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
                pred_id = int(np.argmax(probs))
                confidence = float(probs[pred_id])
                label = self.reverse_label_map.get(pred_id, "NEUTRAL")

                # Compute a compound-like score (-1 to 1) from probabilities
                # If 3-class: P(POS) - P(NEG) gives a continuous polarity
                neg_prob = float(probs[self.reverse_label_map and
                                       {v: k for k, v in self.reverse_label_map.items()}.get("NEGATIVE", 0)])
                pos_prob = float(probs[self.reverse_label_map and
                                       {v: k for k, v in self.reverse_label_map.items()}.get("POSITIVE", 2)])

                compound = pos_prob - neg_prob

                return {
                    "label": label,
                    "score": round(confidence, 4),
                    "compound": round(compound, 4),
                }

            except Exception as e:
                # Fall through to fallback
                pass

        # Fallback
        if self._fallback_engine is not None:
            return self._fallback_engine.analyze(text)

        return {"label": "NEUTRAL", "score": 0.5, "compound": 0.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 128) -> List[Dict[str, Any]]:
        """
        Batch inference with GPU — much faster than one-by-one.
        Processes texts in batches for optimal GPU utilization.
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to sequential
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

            for j, prob in enumerate(probs):
                pred_id = int(np.argmax(prob))
                confidence = float(prob[pred_id])
                label = self.reverse_label_map.get(pred_id, "NEUTRAL")

                # Compound score
                label_to_id = {v: k for k, v in self.reverse_label_map.items()}
                neg_id = label_to_id.get("NEGATIVE", 0)
                pos_id = label_to_id.get("POSITIVE", 2)
                compound = float(prob[pos_id]) - float(prob[neg_id])

                results.append({
                    "label": label,
                    "score": round(confidence, 4),
                    "compound": round(compound, 4),
                })

        return results
