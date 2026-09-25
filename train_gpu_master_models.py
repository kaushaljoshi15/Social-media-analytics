"""
PulseGuard Master GPU Training Orchestrator
════════════════════════════════════════════
Complete 7-day training pipeline that:
  1. Downloads ~46M samples from HuggingFace + local sources
  2. Fine-tunes RoBERTa-large for Sentiment (3-class)
  3. Fine-tunes RoBERTa-large for Toxicity (binary)
  4. Saves production models for real-time PulseGuard inference

Hardware Target: 64 GB VRAM NVIDIA GPU
Expected Duration: 5-7 days continuous
Expected Accuracy: Sentiment ~95-97%, Toxicity ~97-99%

Usage:
  # Full pipeline (download + train all)
  python train_gpu_master_models.py

  # Train only (skip download if data exists)
  python train_gpu_master_models.py --skip-download

  # Quick test run with small data sample
  python train_gpu_master_models.py --test-mode

  # Choose model architecture
  python train_gpu_master_models.py --model distilbert
  python train_gpu_master_models.py --model roberta-large
  python train_gpu_master_models.py --model deberta-large
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import argparse
from pathlib import Path
from datetime import datetime

import torch

from src.config import PROCESSED_DATA_DIR

GPU_DATA_DIR = PROCESSED_DATA_DIR / "gpu_datasets"
GPU_MODELS_DIR = PROCESSED_DATA_DIR / "gpu_models"


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║   ██████╗ ██╗   ██╗██╗     ███████╗███████╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗  ║")
    print("║   ██╔══██╗██║   ██║██║     ██╔════╝██╔════╝██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗ ║")
    print("║   ██████╔╝██║   ██║██║     ███████╗█████╗  ██║  ███╗██║   ██║███████║██████╔╝██║  ██║ ║")
    print("║   ██╔═══╝ ██║   ██║██║     ╚════██║██╔══╝  ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║ ║")
    print("║   ██║     ╚██████╔╝███████╗███████║███████╗╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝ ║")
    print("║   ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ║")
    print("║                                                                      ║")
    print("║          GPU Master Training Pipeline — 64 GB VRAM Edition            ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def print_gpu_info():
    """Print detailed GPU information."""
    print("┌─── GPU Information ──────────────────────────────────────────────┐")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        cuda_version = torch.version.cuda
        print(f"│ GPU:           {gpu_name}")
        print(f"│ VRAM:          {vram_total:.1f} GB")
        print(f"│ CUDA Version:  {cuda_version}")
        print(f"│ PyTorch:       {torch.__version__}")
        print(f"│ Num GPUs:      {torch.cuda.device_count()}")

        # CUDA capability
        cap = torch.cuda.get_device_capability(0)
        print(f"│ Compute Cap:   {cap[0]}.{cap[1]}")

        # Check FP16 support
        if cap[0] >= 7:
            print(f"│ FP16/BF16:     ✓ Supported (Tensor Cores)")
        else:
            print(f"│ FP16:          ✓ Supported")
    else:
        print("│ ⚠ No CUDA GPU detected! Training will be extremely slow on CPU.")
    print("└──────────────────────────────────────────────────────────────────┘\n")


def step_download(test_mode=False):
    """Step 1: Download and merge all datasets."""
    print("\n" + "█" * 70)
    print("█  STEP 1/3: DOWNLOAD & MERGE DATASETS")
    print("█" * 70)

    from src.nlp.download_hf_datasets import download_all

    max_per = 5000 if test_mode else None
    sent_path, tox_path = download_all(max_samples_per_source=max_per)

    return sent_path, tox_path


def step_train_sentiment(data_path, model_type="roberta-large", epochs=3):
    """Step 2: Fine-tune sentiment model."""
    print("\n" + "█" * 70)
    print("█  STEP 2/3: FINE-TUNE SENTIMENT MODEL")
    print("█" * 70)

    if data_path is None or not Path(data_path).exists():
        print("[!] No sentiment training data found. Skipping.")
        return None

    from src.nlp.gpu_fine_tuner import train_sentiment
    model_dir, metadata = train_sentiment(
        data_path=str(data_path),
        model_type=model_type,
        epochs=epochs,
    )
    return model_dir


def step_train_toxicity(data_path, model_type="roberta-large", epochs=3):
    """Step 3: Fine-tune toxicity model."""
    print("\n" + "█" * 70)
    print("█  STEP 3/3: FINE-TUNE TOXICITY MODEL")
    print("█" * 70)

    if data_path is None or not Path(data_path).exists():
        print("[!] No toxicity training data found. Skipping.")
        return None

    from src.nlp.gpu_fine_tuner import train_toxicity
    model_dir, metadata = train_toxicity(
        data_path=str(data_path),
        model_type=model_type,
        epochs=epochs,
    )
    return model_dir


def run_full_pipeline(args):
    """Execute the full training pipeline."""
    print_banner()
    print_gpu_info()

    print(f"[Config] Model:          {args.model}")
    print(f"[Config] Epochs:         {args.epochs}")
    print(f"[Config] Test mode:      {args.test_mode}")
    print(f"[Config] Skip download:  {args.skip_download}")
    print(f"[Config] Started at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    pipeline_start = time.time()

    # ── Step 1: Download Datasets ─────────────────────────────────────
    sent_path = GPU_DATA_DIR / "sentiment_train_merged.parquet"
    tox_path = GPU_DATA_DIR / "toxicity_train_merged.parquet"

    if not args.skip_download:
        sent_path, tox_path = step_download(test_mode=args.test_mode)
    else:
        print("\n[~] Skipping download. Using existing datasets:")
        if sent_path.exists():
            print(f"    Sentiment: {sent_path}")
        else:
            print(f"    ⚠ Sentiment data not found at {sent_path}")
            sent_path = None
        if tox_path.exists():
            print(f"    Toxicity:  {tox_path}")
        else:
            print(f"    ⚠ Toxicity data not found at {tox_path}")
            tox_path = None

    # ── Step 2: Train Sentiment ───────────────────────────────────────
    sent_model = step_train_sentiment(
        data_path=sent_path,
        model_type=args.model,
        epochs=args.epochs,
    )

    # ── Step 3: Train Toxicity ────────────────────────────────────────
    tox_model = step_train_toxicity(
        data_path=tox_path,
        model_type=args.model,
        epochs=args.epochs,
    )

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    hours = total_time / 3600
    days = hours / 24

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    TRAINING PIPELINE COMPLETE                       ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total Time:     {hours:.1f} hours ({days:.1f} days)")
    if sent_model:
        print(f"║  Sentiment Model: {sent_model}")
    if tox_model:
        print(f"║  Toxicity Model:  {tox_model}")
    print(f"║  GPU Models Dir:  {GPU_MODELS_DIR}")
    print("║")
    print("║  Next Steps:")
    print("║  1. Models are auto-detected by GPUSentimentEngine / GPUToxicityEngine")
    print("║  2. Run `streamlit run app.py` to use the upgraded models")
    print("║  3. Or import directly:")
    print("║     from src.nlp.gpu_sentiment_engine import GPUSentimentEngine")
    print("║     from src.nlp.gpu_toxicity_engine import GPUToxicityEngine")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PulseGuard GPU Master Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download 46M samples + train all models)
  python train_gpu_master_models.py

  # Quick test run with 5K samples per source
  python train_gpu_master_models.py --test-mode

  # Skip download, train with existing data
  python train_gpu_master_models.py --skip-download

  # Use DistilBERT (faster, lower accuracy)
  python train_gpu_master_models.py --model distilbert --epochs 5

  # Use DeBERTa-v3-large (best accuracy, slower)
  python train_gpu_master_models.py --model deberta-large --epochs 3
        """
    )

    parser.add_argument(
        "--model", type=str, default="roberta-large",
        choices=["distilbert", "roberta-large", "deberta-large"],
        help="Transformer model architecture (default: roberta-large)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Quick test with 5K samples per source (for verification)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download; use existing data in data/processed/gpu_datasets/"
    )

    args = parser.parse_args()
    run_full_pipeline(args)
