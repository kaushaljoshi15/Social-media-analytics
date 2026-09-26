"""
Merge & Export Qwen 2.5 14B LoRA Weights into Standalone Production Model
════════════════════════════════════════════════════════════════════════
Usage:
  python export_qwen_model.py
  python export_qwen_model.py --lora-path data/processed/qwen_checkpoints/qwen2.5_14b_pulseguard_final --output-dir models/qwen2.5_14b_standalone
"""
import os
import sys
import argparse
from pathlib import Path
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

def export_model(lora_path, output_dir):
    print("=" * 70)
    print("🔄 Merging Qwen 2.5 14B LoRA Adapters into Standalone Base Model")
    print(f"   • LoRA Weights: {lora_path}")
    print(f"   • Export Target: {output_dir}")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("📥 Loading Base Model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("🎯 Ingesting LoRA Adapter Weights...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("⚙️ Merging layers and unloading LoRA matrices...")
    merged_model = peft_model.merge_and_unload()

    print(f"💾 Saving Standalone Model to {output_path}...")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("\n✅ Merge Complete! Your standalone 14.7B model is ready for vLLM & Streamlit deployment.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default="data/processed/qwen_checkpoints/qwen2.5_14b_pulseguard_final")
    parser.add_argument("--output-dir", type=str, default="models/qwen2.5_14b_pulseguard_standalone")
    args = parser.parse_args()

    export_model(args.lora_path, args.output_dir)
