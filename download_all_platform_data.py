"""
PulseGuard Multi-Platform Benchmark Dataset Fetcher
Downloads free real-world datasets across YouTube, Reddit, Twitter, and Telegram
from HuggingFace, Google Research, UCI, and open GitHub academic repositories.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import requests
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR

DATASETS = {
    "youtube": {
        "name": "YouTube Video Comments Collection",
        "url": "https://raw.githubusercontent.com/albertyw/sentiment-analysis/master/test/data/youtube.csv",
        "fallback_url": "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv",
        "target_file": "real_youtube_comments.csv",
        "source": "UCI / Open Benchmark"
    },
    "reddit": {
        "name": "Reddit Google Research GoEmotions",
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
        "target_file": "real_reddit_discussions.tsv",
        "source": "Google Research / Reddit"
    },
    "twitter": {
        "name": "Twitter Brand Backlash Corpus",
        "url": "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv",
        "target_file": "real_twitter_brand_tweets.csv",
        "source": "Twitter Research Benchmark"
    },
    "toxicity": {
        "name": "Davidson Social Media Toxicity & Hate Speech",
        "url": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        "target_file": "real_toxicity_harassment.csv",
        "source": "Cornell University NLP"
    }
}

def fetch_all_platform_datasets():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=======================================================================")
    print("[*] Downloading Official Open Datasets for YouTube, Reddit, Twitter & Telegram")
    print("=======================================================================")

    downloaded = {}
    for platform, info in DATASETS.items():
        dest = RAW_DATA_DIR / info["target_file"]
        print(f"\n[+] Fetching {info['name']} ({info['source']})...")
        print(f" -> URL: {info['url']}")
        try:
            resp = requests.get(info["url"], timeout=20.0)
            if resp.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(resp.content)
                print(f" -> Saved to {dest.name} ({len(resp.content):,} bytes)")
                downloaded[platform] = dest
            else:
                print(f" -> HTTP {resp.status_code}")
        except Exception as e:
            print(f" -> Error downloading: {e}")

    # For Telegram: Extract live public channel dataset from @durov, @telegram, and @techcrunch
    print("\n[+] Generating Real Telegram Public Channel Dataset via live extractor...")
    try:
        from src.ingestion.telegram_client import TelegramClient
        tc = TelegramClient()
        tele_records = []
        for ch in ["durov", "telegram", "techcrunch"]:
            msgs = tc.inspect_channel(ch, max_messages=25)
            tele_records.extend(msgs)
        if tele_records:
            t_df = pd.DataFrame(tele_records)
            t_path = RAW_DATA_DIR / "real_telegram_messages.csv"
            t_df.to_csv(t_path, index=False)
            print(f" -> Saved {len(t_df)} real Telegram channel messages to {t_path.name}")
            downloaded["telegram"] = t_path
    except Exception as e:
        print(f" -> Telegram extraction note: {e}")

    print("\n=======================================================================")
    print(f"[Done] Ingested {len(downloaded)} platform datasets into {RAW_DATA_DIR}")
    print("=======================================================================")
    return downloaded

if __name__ == "__main__":
    fetch_all_platform_datasets()
