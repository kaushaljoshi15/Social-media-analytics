"""
PulseGuard GPU Fine-Tuning Engine (PyTorch + HuggingFace Transformers)
═══════════════════════════════════════════════════════════════════════
Leverages 64 GB VRAM NVIDIA GPU for large-scale Transformer fine-tuning.
Supports: RoBERTa-large, DistilBERT, DeBERTa-v3-large
Features: Mixed-Precision (FP16), Gradient Accumulation, Cosine LR Scheduler,
          Early Stopping, Checkpointing, Tensorboard Logging
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import PROCESSED_DATA_DIR

# ═══════════════════════════════════════════════════════════════════════
# Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════

GPU_MODELS_DIR = PROCESSED_DATA_DIR / "gpu_models"
GPU_MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = GPU_MODELS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIGS = {
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 256,
        "lr": 3e-5,
        "epochs": 3,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
    },
    "roberta-large": {
        "model_name": "roberta-large",
        "max_length": 128,
        "batch_size": 64,
        "lr": 2e-5,
        "epochs": 3,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
    },
    "deberta-large": {
        "model_name": "microsoft/deberta-v3-large",
        "max_length": 128,
        "batch_size": 48,
        "lr": 1e-5,
        "epochs": 3,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Custom PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════

class TextClassificationDataset(Dataset):
    """Memory-efficient PyTorch dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════
# GPU Fine-Tuner Class
# ═══════════════════════════════════════════════════════════════════════

class GPUFineTuner:
    """
    Enterprise GPU Fine-Tuning Engine for PulseGuard.
    Optimized for 64 GB VRAM NVIDIA GPUs (A100, A6000, L40S, etc.)
    """

    def __init__(self, model_type="roberta-large", task="sentiment", num_labels=3):
        self.model_type = model_type
        self.task = task
        self.num_labels = num_labels
        self.device = self._detect_device()
        self.config = DEFAULT_CONFIGS.get(model_type, DEFAULT_CONFIGS["roberta-large"])
        self.model = None
        self.tokenizer = None
        self.training_log = []

    def _detect_device(self):
        """Detect GPU and report VRAM."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            print(f"[GPU] Detected: {gpu_name}")
            print(f"[GPU] VRAM: {vram_gb:.1f} GB")

            # Auto-adjust batch size based on available VRAM
            if vram_gb >= 48:
                print(f"[GPU] Mode: HIGH-VRAM (≥48 GB) — Full batch sizes enabled")
            elif vram_gb >= 24:
                print(f"[GPU] Mode: MEDIUM-VRAM (24-48 GB) — Adjusted batch sizes")
                for key in DEFAULT_CONFIGS:
                    DEFAULT_CONFIGS[key]["batch_size"] = max(8, DEFAULT_CONFIGS[key]["batch_size"] // 2)
                    DEFAULT_CONFIGS[key]["gradient_accumulation_steps"] *= 2
            else:
                print(f"[GPU] Mode: LOW-VRAM (<24 GB) — Minimal batch sizes")
                for key in DEFAULT_CONFIGS:
                    DEFAULT_CONFIGS[key]["batch_size"] = max(4, DEFAULT_CONFIGS[key]["batch_size"] // 4)
                    DEFAULT_CONFIGS[key]["gradient_accumulation_steps"] *= 4

            return torch.device("cuda")
        else:
            print("[GPU] No CUDA GPU detected. Falling back to CPU (very slow).")
            return torch.device("cpu")

    def _load_model(self):
        """Load pretrained model and tokenizer."""
        model_name = self.config["model_name"]
        print(f"\n[Model] Loading {model_name} ({self.num_labels} labels)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")

        # Report VRAM after model load
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[GPU] VRAM after model load: {allocated:.2f} GB")

    def _prepare_data(self, data_path, text_col="text", label_col=None, label_map=None):
        """Load and prepare training data from Parquet or CSV."""
        print(f"\n[Data] Loading from {data_path}...")

        ext = Path(data_path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        df = df.dropna(subset=[text_col])
        df[text_col] = df[text_col].astype(str)

        # Determine label column
        if label_col is None:
            label_col = "sentiment" if self.task == "sentiment" else "is_toxic"

        # Build label mapping
        if label_map is None:
            unique_labels = sorted(df[label_col].unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}

        df["label_id"] = df[label_col].map(label_map)
        df = df.dropna(subset=["label_id"])
        df["label_id"] = df["label_id"].astype(int)

        print(f"[Data] Total samples: {len(df):,}")
        print(f"[Data] Label mapping: {label_map}")
        print(f"[Data] Distribution: {df['label_id'].value_counts().sort_index().to_dict()}")

        # Train/val split
        train_df, val_df = train_test_split(
            df, test_size=0.05, random_state=42,
            stratify=df["label_id"] if len(df["label_id"].unique()) > 1 else None
        )

        train_dataset = TextClassificationDataset(
            texts=train_df[text_col].tolist(),
            labels=train_df["label_id"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config["max_length"],
        )
        val_dataset = TextClassificationDataset(
            texts=val_df[text_col].tolist(),
            labels=val_df["label_id"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config["max_length"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(f"[Data] Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
        print(f"[Data] Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

        return train_loader, val_loader, label_map

    def _evaluate(self, val_loader):
        """Run evaluation pass and compute metrics."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                total_loss += outputs.loss.item()
                num_batches += 1

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        return {
            "val_loss": round(avg_loss, 4),
            "val_accuracy": round(accuracy, 4),
            "val_f1": round(f1, 4),
            "predictions": all_preds,
            "labels": all_labels,
        }

    def train(self, data_path, text_col="text", label_col=None, label_map=None,
              epochs=None, patience=3):
        """
        Full fine-tuning loop with:
        - Mixed Precision (FP16)
        - Gradient Accumulation
        - Cosine LR Scheduler with Warmup
        - Early Stopping
        - Periodic Checkpointing
        """
        # Load model
        self._load_model()

        # Prepare data
        train_loader, val_loader, label_map = self._prepare_data(
            data_path, text_col, label_col, label_map
        )

        if epochs is None:
            epochs = self.config["epochs"]

        grad_accum_steps = self.config["gradient_accumulation_steps"]
        effective_batch = self.config["batch_size"] * grad_accum_steps

        print(f"\n{'═' * 70}")
        print(f"[Training] Starting {self.task.upper()} fine-tuning")
        print(f"{'═' * 70}")
        print(f"  Model:              {self.config['model_name']}")
        print(f"  Epochs:             {epochs}")
        print(f"  Batch size:         {self.config['batch_size']}")
        print(f"  Grad accumulation:  {grad_accum_steps}")
        print(f"  Effective batch:    {effective_batch}")
        print(f"  Learning rate:      {self.config['lr']}")
        print(f"  Max sequence len:   {self.config['max_length']}")
        print(f"  Mixed precision:    FP16 enabled")
        print(f"  Device:             {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU:                {torch.cuda.get_device_name(0)}")
        print(f"{'═' * 70}\n")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            eps=1e-8,
        )

        # Scheduler
        total_steps = (len(train_loader) // grad_accum_steps) * epochs
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed Precision Scaler
        scaler = GradScaler()

        # Early stopping
        best_val_f1 = 0.0
        patience_counter = 0
        best_checkpoint_path = None

        # ── Training Loop ─────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum_steps

                scaler.scale(loss).backward()
                epoch_loss += outputs.loss.item()

                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Progress logging every 500 steps
                if (step + 1) % 500 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    throughput = (step + 1) * self.config["batch_size"] / elapsed
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                        print(
                            f"  Epoch {epoch}/{epochs} | Step {step + 1}/{len(train_loader)} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | "
                            f"{throughput:.0f} samples/sec | "
                            f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB"
                        )
                    else:
                        print(
                            f"  Epoch {epoch}/{epochs} | Step {step + 1}/{len(train_loader)} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | "
                            f"{throughput:.0f} samples/sec"
                        )

            # ── End of Epoch ──────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(train_loader)

            # Validation
            print(f"\n  [Evaluating epoch {epoch}...]")
            val_metrics = self._evaluate(val_loader)

            log_entry = {
                "epoch": epoch,
                "train_loss": round(avg_epoch_loss, 4),
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["val_accuracy"],
                "val_f1": val_metrics["val_f1"],
                "epoch_time_min": round(epoch_time / 60, 1),
                "lr": scheduler.get_last_lr()[0],
            }
            self.training_log.append(log_entry)

            print(f"\n  ┌─── Epoch {epoch}/{epochs} Summary ───────────────────────────────┐")
            print(f"  │ Train Loss:    {avg_epoch_loss:.4f}")
            print(f"  │ Val Loss:      {val_metrics['val_loss']:.4f}")
            print(f"  │ Val Accuracy:  {val_metrics['val_accuracy'] * 100:.2f}%")
            print(f"  │ Val F1:        {val_metrics['val_f1'] * 100:.2f}%")
            print(f"  │ Time:          {epoch_time / 60:.1f} minutes")
            print(f"  └────────────────────────────────────────────────────────────┘\n")

            # Checkpoint & Early Stopping
            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                patience_counter = 0

                # Save best checkpoint
                ckpt_name = f"{self.task}_{self.model_type}_best"
                best_checkpoint_path = CHECKPOINTS_DIR / ckpt_name
                self.model.save_pretrained(best_checkpoint_path)
                self.tokenizer.save_pretrained(best_checkpoint_path)
                print(f"  [✓] New best checkpoint saved: {ckpt_name} (F1: {best_val_f1 * 100:.2f}%)")
            else:
                patience_counter += 1
                print(f"  [~] No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"  [!] Early stopping triggered at epoch {epoch}")
                    break

        # ── Final: Load best checkpoint and save production model ─────
        print(f"\n{'═' * 70}")
        print(f"[Saving] Exporting production model...")

        if best_checkpoint_path and best_checkpoint_path.exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
            self.model.to(self.device)

        production_dir = GPU_MODELS_DIR / f"{self.task}_{self.model_type}"
        self.model.save_pretrained(production_dir)
        self.tokenizer.save_pretrained(production_dir)

        # Save label mapping and training metadata
        metadata = {
            "task": self.task,
            "model_type": self.model_type,
            "model_name": self.config["model_name"],
            "num_labels": self.num_labels,
            "label_map": label_map,
            "reverse_label_map": {v: k for k, v in label_map.items()},
            "best_val_f1": best_val_f1,
            "best_val_accuracy": max(e["val_accuracy"] for e in self.training_log),
            "total_epochs_trained": len(self.training_log),
            "training_log": self.training_log,
            "config": self.config,
            "trained_at": datetime.now().isoformat(),
        }
        with open(production_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"[✓] Production model saved to: {production_dir}")
        print(f"[✓] Best Val F1: {best_val_f1 * 100:.2f}%")
        print(f"{'═' * 70}")

        # Final detailed classification report
        if self.training_log:
            val_metrics = self._evaluate(val_loader)
            reverse_map = {v: k for k, v in label_map.items()}
            target_names = [reverse_map[i] for i in sorted(reverse_map.keys())]
            print("\n[Final Classification Report]")
            print(classification_report(
                val_metrics["labels"],
                val_metrics["predictions"],
                target_names=target_names,
                zero_division=0,
            ))

        return production_dir, metadata


# ═══════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════

def train_sentiment(data_path, model_type="roberta-large", epochs=3):
    """Fine-tune a sentiment analysis model."""
    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    tuner = GPUFineTuner(model_type=model_type, task="sentiment", num_labels=3)
    return tuner.train(
        data_path=data_path,
        text_col="text",
        label_col="sentiment",
        label_map=label_map,
        epochs=epochs,
    )


def train_toxicity(data_path, model_type="roberta-large", epochs=3):
    """Fine-tune a toxicity detection model."""
    label_map = {0: 0, 1: 1}
    tuner = GPUFineTuner(model_type=model_type, task="toxicity", num_labels=2)
    return tuner.train(
        data_path=data_path,
        text_col="text",
        label_col="is_toxic",
        label_map=label_map,
        epochs=epochs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PulseGuard GPU Fine-Tuning Engine")
    parser.add_argument("--task", choices=["sentiment", "toxicity"], required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to training data (Parquet or CSV)")
    parser.add_argument("--model", type=str, default="roberta-large",
                        choices=["distilbert", "roberta-large", "deberta-large"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.task == "sentiment":
        train_sentiment(args.data, args.model, args.epochs)
    else:
        train_toxicity(args.data, args.model, args.epochs)
