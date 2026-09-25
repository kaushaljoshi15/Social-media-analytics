"""
PulseGuard Automated Dataset Downloader & Fine-Tuning Pipeline
Downloads real-world public brand sentiment & toxicity datasets and fine-tunes
the machine learning inference models.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import requests
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.nlp.fine_tuner import ModelFineTuner

# Target Public Benchmark Datasets
DATASETS = {
    "twitter_brand_corpus": {
        "url": "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv",
        "filename": "twitter_brand_corpus.csv",
        "description": "5,000+ real public tweets targeting Apple, Google, Microsoft, and Twitter with human labels."
    },
    "hate_speech_toxicity": {
        "url": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        "filename": "davidson_toxicity_labeled.csv",
        "description": "24,000+ real tweets classified for Hate Speech, Toxic Harassment, and Clean Language."
    }
}

def download_benchmark_data():
    """Download real-world benchmark datasets for model tuning."""
    print("=================================================================")
    print("[*] PulseGuard Automated Dataset Fetcher & Tuning Engine")
    print("=================================================================")
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_files = {}

    for key, info in DATASETS.items():
        file_path = RAW_DATA_DIR / info["filename"]
        print(f"\n[+] Fetching {info['description']}...")
        print(f" -> Source URL: {info['url']}")
        
        try:
            resp = requests.get(info["url"], timeout=15.0)
            if resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                print(f" -> Successfully saved to {file_path.name} ({len(resp.content):,} bytes)")
                downloaded_files[key] = file_path
            else:
                print(f" -> HTTP error: {resp.status_code}")
        except Exception as e:
            print(f" -> Download failed: {e}")

    return downloaded_files

def prepare_and_tune_brand_model(corpus_path: Path):
    """Normalize twitter brand corpus and train fine-tuned classifier."""
    print("\n-----------------------------------------------------------------")
    print("[*] Preprocessing & Fine-Tuning Brand Sentiment Model...")
    print("-----------------------------------------------------------------")
    
    df = pd.read_csv(corpus_path)
    # Map sentiment labels to standard PulseGuard classes
    label_map = {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
        "irrelevant": "NEUTRAL"
    }
    df["sentiment"] = df["Sentiment"].str.lower().map(label_map).fillna("NEUTRAL")
    df["text"] = df["TweetText"]
    
    clean_csv = RAW_DATA_DIR / "cleaned_brand_training.csv"
    df[["text", "sentiment"]].dropna().to_csv(clean_csv, index=False)
    print(f" -> Prepared {len(df)} labeled training samples at {clean_csv.name}")

    tuner = ModelFineTuner(output_dir=PROCESSED_DATA_DIR / "models")
    tuner.run_pipeline(
        csv_path=str(clean_csv),
        text_col="text",
        label_col="sentiment",
        model_type="logistic_regression"
    )

if __name__ == "__main__":
    downloaded = download_benchmark_data()
    if "twitter_brand_corpus" in downloaded:
        prepare_and_tune_brand_model(downloaded["twitter_brand_corpus"])
    print("\n[Done] All real datasets ingested and models successfully fine-tuned!")
