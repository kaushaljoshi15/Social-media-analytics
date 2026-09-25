"""
PulseGuard Large-Scale Dataset Downloader (HuggingFace Hub)
Downloads and unifies ~46M samples for GPU fine-tuning:
  - Sentiment: Amazon Polarity (34M), Yelp (6.7M), IMDB (50K), SST-2 (67K),
               Sentiment140 (1.6M), TweetEval (124K), + local datasets
  - Toxicity: Civil Comments (1.8M), Jigsaw (1.6M), + local Davidson (25K)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Directory for large-scale GPU training data
GPU_DATA_DIR = PROCESSED_DATA_DIR / "gpu_datasets"
GPU_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_load_hf(dataset_name, split="train", config=None, streaming=False):
    """Safely load a HuggingFace dataset with error handling."""
    try:
        from datasets import load_dataset
        kwargs = {"path": dataset_name, "split": split}
        if config:
            kwargs["name"] = config
        if streaming:
            kwargs["streaming"] = True
        ds = load_dataset(**kwargs)
        print(f"   [✓] Loaded {dataset_name} ({split})")
        return ds
    except Exception as e:
        print(f"   [✗] Failed to load {dataset_name}: {e}")
        return None


def download_sentiment_datasets(max_samples_per_source=None):
    """
    Download and unify all sentiment training data.
    Returns path to the merged CSV file.
    """
    print("\n" + "=" * 70)
    print("[1/2] Downloading Sentiment Training Datasets")
    print("=" * 70)

    all_frames = []

    # ── 1. Amazon Polarity (34M reviews) ──────────────────────────────────
    print("\n[+] Amazon Polarity (34M reviews)...")
    ds = _safe_load_hf("amazon_polarity", split="train")
    if ds is not None:
        df = ds.to_pandas() if not hasattr(ds, '__iter__') or hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        # Amazon Polarity: label 0 = negative, 1 = positive
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        df["sentiment"] = df["label"].map(label_map)
        # Combine title and content for richer text
        df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")
        df["text"] = df["text"].str.strip()
        df = df[["text", "sentiment"]].dropna()
        if max_samples_per_source:
            df = df.sample(n=min(max_samples_per_source, len(df)), random_state=42)
        all_frames.append(("Amazon Polarity", df))
        print(f"   → {len(df):,} samples")

    # ── 2. Yelp Review Full (6.7M reviews) ────────────────────────────────
    print("\n[+] Yelp Review Full (6.7M reviews)...")
    ds = _safe_load_hf("yelp_review_full", split="train")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        # Yelp: labels 0-4 (1-5 stars)
        def yelp_to_sentiment(label):
            if label <= 1:
                return "NEGATIVE"
            elif label == 2:
                return "NEUTRAL"
            else:
                return "POSITIVE"
        df["sentiment"] = df["label"].apply(yelp_to_sentiment)
        df = df.rename(columns={"text": "text"})[["text", "sentiment"]].dropna()
        if max_samples_per_source:
            df = df.sample(n=min(max_samples_per_source, len(df)), random_state=42)
        all_frames.append(("Yelp Reviews", df))
        print(f"   → {len(df):,} samples")

    # ── 3. IMDB (50K movie reviews) ───────────────────────────────────────
    print("\n[+] IMDB Movie Reviews (50K)...")
    ds = _safe_load_hf("imdb", split="train")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        df["sentiment"] = df["label"].map(label_map)
        df = df[["text", "sentiment"]].dropna()
        all_frames.append(("IMDB", df))
        print(f"   → {len(df):,} samples")

    # ── 4. SST-2 (Stanford Sentiment Treebank) ────────────────────────────
    print("\n[+] SST-2 Stanford Sentiment (67K)...")
    ds = _safe_load_hf("glue", split="train", config="sst2")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        df["sentiment"] = df["label"].map(label_map)
        df = df.rename(columns={"sentence": "text"})[["text", "sentiment"]].dropna()
        all_frames.append(("SST-2", df))
        print(f"   → {len(df):,} samples")

    # ── 5. TweetEval Sentiment (124K tweets) ──────────────────────────────
    print("\n[+] TweetEval Sentiment (124K tweets)...")
    ds = _safe_load_hf("tweet_eval", split="train", config="sentiment")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        df["sentiment"] = df["label"].map(label_map)
        df = df[["text", "sentiment"]].dropna()
        all_frames.append(("TweetEval", df))
        print(f"   → {len(df):,} samples")

    # ── 6. Local Sentiment140 (1.6M tweets, already downloaded) ───────────
    print("\n[+] Local Sentiment140 (1.6M tweets)...")
    p_sent140_full = RAW_DATA_DIR / "training.1600000.processed.noemoticon.csv"
    if p_sent140_full.exists():
        try:
            df = pd.read_csv(
                p_sent140_full,
                encoding="latin-1",
                header=None,
                names=["target", "id", "date", "flag", "user", "text"]
            )
            label_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
            df["sentiment"] = df["target"].map(label_map)
            df = df[["text", "sentiment"]].dropna()
            if max_samples_per_source:
                df = df.sample(n=min(max_samples_per_source, len(df)), random_state=42)
            all_frames.append(("Sentiment140 Full", df))
            print(f"   → {len(df):,} samples")
        except Exception as e:
            print(f"   [✗] Failed to load Sentiment140: {e}")

    # ── 7. Local Brand Corpus ─────────────────────────────────────────────
    print("\n[+] Local Twitter Brand Corpus...")
    p_brand = RAW_DATA_DIR / "cleaned_brand_training.csv"
    if p_brand.exists():
        try:
            df = pd.read_csv(p_brand)
            df = df[["text", "sentiment"]].dropna()
            all_frames.append(("Brand Corpus", df))
            print(f"   → {len(df):,} samples")
        except Exception as e:
            print(f"   [✗] Failed: {e}")

    # ── 8. Local GoEmotions (Reddit) ──────────────────────────────────────
    print("\n[+] Local Reddit GoEmotions...")
    p_reddit = RAW_DATA_DIR / "real_reddit_discussions.tsv"
    if p_reddit.exists():
        try:
            df = pd.read_csv(p_reddit, sep="\t", header=None, names=["text", "label_id", "comment_id"])
            # Simplified: score > 15 positive, < 5 negative, else neutral
            def reddit_score_to_sent(score):
                try:
                    s = int(score)
                    if s >= 15:
                        return "POSITIVE"
                    elif s <= 2:
                        return "NEGATIVE"
                    else:
                        return "NEUTRAL"
                except (ValueError, TypeError):
                    return "NEUTRAL"
            df["sentiment"] = df["label_id"].apply(reddit_score_to_sent)
            df = df[["text", "sentiment"]].dropna()
            all_frames.append(("Reddit GoEmotions", df))
            print(f"   → {len(df):,} samples")
        except Exception as e:
            print(f"   [✗] Failed: {e}")

    # ── Merge all sentiment data ──────────────────────────────────────────
    if not all_frames:
        print("\n[!] No sentiment datasets available!")
        return None

    print(f"\n{'─' * 70}")
    print("[+] Merging all sentiment datasets...")
    merged = pd.concat([df for _, df in all_frames], ignore_index=True)
    merged = merged[merged["sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])]
    merged = merged[merged["text"].str.strip().str.len() > 3]
    merged = merged.drop_duplicates(subset=["text"])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = GPU_DATA_DIR / "sentiment_train_merged.parquet"
    merged.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"\n[✓] Unified Sentiment Dataset: {len(merged):,} samples")
    print(f"    Distribution: {merged['sentiment'].value_counts().to_dict()}")
    print(f"    Saved to: {out_path}")

    for name, df in all_frames:
        print(f"    • {name}: {len(df):,}")

    return out_path


def download_toxicity_datasets(max_samples_per_source=None):
    """
    Download and unify all toxicity / harassment training data.
    Returns path to the merged CSV file.
    """
    print("\n" + "=" * 70)
    print("[2/2] Downloading Toxicity & Harassment Datasets")
    print("=" * 70)

    all_frames = []

    # ── 1. Civil Comments (1.8M comments) ─────────────────────────────────
    print("\n[+] Civil Comments (1.8M labeled comments)...")
    ds = _safe_load_hf("civil_comments", split="train")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        # Toxicity score >= 0.5 means toxic
        df["is_toxic"] = (df["toxicity"] >= 0.5).astype(int)
        df = df.rename(columns={"text": "text"})[["text", "is_toxic"]].dropna()
        if max_samples_per_source:
            df = df.sample(n=min(max_samples_per_source, len(df)), random_state=42)
        all_frames.append(("Civil Comments", df))
        print(f"   → {len(df):,} samples")

    # ── 2. Jigsaw Toxicity (via HuggingFace) ──────────────────────────────
    print("\n[+] Jigsaw Toxic Comment Classification...")
    ds = _safe_load_hf("jigsaw_toxicity_pred", split="train")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        if "toxic" in df.columns:
            df["is_toxic"] = (df["toxic"] >= 0.5).astype(int)
            text_col = "comment_text" if "comment_text" in df.columns else "text"
            df = df.rename(columns={text_col: "text"})[["text", "is_toxic"]].dropna()
            if max_samples_per_source:
                df = df.sample(n=min(max_samples_per_source, len(df)), random_state=42)
            all_frames.append(("Jigsaw Toxicity", df))
            print(f"   → {len(df):,} samples")
    else:
        # Fallback: Try alternative jigsaw dataset
        print("   [~] Trying alternative toxicity dataset...")
        ds = _safe_load_hf("SetFit/toxic_conversations", split="train")
        if ds is not None:
            df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
            if "label" in df.columns:
                df["is_toxic"] = df["label"].astype(int)
                df = df[["text", "is_toxic"]].dropna()
                all_frames.append(("Toxic Conversations", df))
                print(f"   → {len(df):,} samples")

    # ── 3. TweetEval Hate Speech ──────────────────────────────────────────
    print("\n[+] TweetEval Hate Speech Detection...")
    ds = _safe_load_hf("tweet_eval", split="train", config="hate")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        df["is_toxic"] = df["label"].astype(int)
        df = df[["text", "is_toxic"]].dropna()
        all_frames.append(("TweetEval Hate", df))
        print(f"   → {len(df):,} samples")

    # ── 4. TweetEval Offensive Language ───────────────────────────────────
    print("\n[+] TweetEval Offensive Language...")
    ds = _safe_load_hf("tweet_eval", split="train", config="offensive")
    if ds is not None:
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else pd.DataFrame(ds)
        df["is_toxic"] = df["label"].astype(int)
        df = df[["text", "is_toxic"]].dropna()
        all_frames.append(("TweetEval Offensive", df))
        print(f"   → {len(df):,} samples")

    # ── 5. Local Davidson Toxicity (25K tweets) ───────────────────────────
    print("\n[+] Local Davidson Toxicity Corpus (25K)...")
    p_tox = RAW_DATA_DIR / "real_toxicity_harassment.csv"
    if not p_tox.exists():
        p_tox = RAW_DATA_DIR / "davidson_toxicity_labeled.csv"
    if p_tox.exists():
        try:
            df = pd.read_csv(p_tox)
            df["is_toxic"] = (df["class"] != 2).astype(int)
            df = df.rename(columns={"tweet": "text"})[["text", "is_toxic"]].dropna()
            all_frames.append(("Davidson Toxicity", df))
            print(f"   → {len(df):,} samples")
        except Exception as e:
            print(f"   [✗] Failed: {e}")

    # ── Merge all toxicity data ───────────────────────────────────────────
    if not all_frames:
        print("\n[!] No toxicity datasets available!")
        return None

    print(f"\n{'─' * 70}")
    print("[+] Merging all toxicity datasets...")
    merged = pd.concat([df for _, df in all_frames], ignore_index=True)
    merged = merged[merged["text"].str.strip().str.len() > 3]
    merged = merged.drop_duplicates(subset=["text"])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = GPU_DATA_DIR / "toxicity_train_merged.parquet"
    merged.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"\n[✓] Unified Toxicity Dataset: {len(merged):,} samples")
    print(f"    Distribution: {merged['is_toxic'].value_counts().to_dict()}")
    print(f"    Saved to: {out_path}")

    for name, df in all_frames:
        print(f"    • {name}: {len(df):,}")

    return out_path


def download_all(max_samples_per_source=None):
    """Download and merge all datasets for both tasks."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   PulseGuard GPU Dataset Downloader — Large-Scale Training Data     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    sent_path = download_sentiment_datasets(max_samples_per_source)
    tox_path = download_toxicity_datasets(max_samples_per_source)

    print("\n" + "=" * 70)
    print("[Done] All datasets downloaded and merged!")
    if sent_path:
        print(f"  Sentiment: {sent_path}")
    if tox_path:
        print(f"  Toxicity:  {tox_path}")
    print("=" * 70)

    return sent_path, tox_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download large-scale datasets for GPU fine-tuning")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Max samples per data source (for testing). None = full download.")
    args = parser.parse_args()
    download_all(max_samples_per_source=args.max_per_source)
