import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import random
import datetime
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, PLATFORMS
from src.nlp.preprocessor import TextPreprocessor
from src.nlp.sentiment_engine import SentimentEngine
from src.nlp.toxicity_engine import ToxicityEngine
from src.database.db_manager import DatabaseManager

def generate_multiplatform_dataset(count: int = 500) -> pd.DataFrame:
    """Generate rich, realistic multi-platform social media comments."""
    preprocessor = TextPreprocessor()
    sentiment_engine = SentimentEngine()
    toxicity_engine = ToxicityEngine()
    db = DatabaseManager()

    scenarios = [
        # Normal positive comments
        ("YouTube", "The battery life on this laptop is incredible, lasts 12 hours easily! 😍", "TechReviewer", "Normal"),
        ("Twitter", "Loving the speed of the new update @Brand, smooth animation and no lag. 🚀", "Alex_K", "Normal"),
        ("Reddit", "Thermals stay under 70C even during heavy Python data science compilation.", "DevOps_Dan", "Normal"),
        ("Telegram", "Customer support answered my ticket in less than 5 minutes. Very helpful.", "Sara_G", "Normal"),
        ("YouTube", "The display resolution and viewing angles are top-tier for video editing.", "CreativeEye", "Normal"),
        ("Twitter", "Upgraded from the previous generation and the improvement is massive. #Tech", "GadgetFan", "Normal"),
        ("Reddit", "Linux kernel drivers work out of the box with zero configuration needed.", "TuxLover", "Normal"),
        ("Telegram", "Delivery arrived 2 days earlier than scheduled. Excellent packaging.", "Kavita_M", "Normal"),
        
        # Product defect crisis comments
        ("YouTube", "Severe overheating issue! Reaching 96C just watching a 4K video. Defective batch! 🤮", "AngryGeek", "Defect"),
        ("Twitter", "Screen flickering constantly and unit won't wake from sleep! Demanding a refund @Brand! 😡", "Frustrated_99", "Defect"),
        ("Reddit", "Motherboard completely died after 48 hours. Service center refuses replacement warranty.", "HardwareVictim", "Defect"),
        ("Telegram", "Battery draining from 100% to 0% in under 40 minutes! Terrible product.", "MobileUser", "Defect"),
        ("YouTube", "Do not buy this model! Huge FPS drop and thermal throttling everywhere.", "GamerX", "Defect"),
        ("Twitter", "Total scam! Selling broken junk and ignoring customer emails. Worst purchase ever! 💩", "ScamAlert", "Defect"),
        ("Reddit", "Trackpad clicking is broken and keyboard keys are getting stuck. Cheap build quality.", "Disappointed", "Defect"),
        ("Telegram", "Customer care helpline is disconnect after 30 minutes on hold. Pathetic service!", "StuckBuyer", "Defect"),
        
        # Server outage / Data crisis comments
        ("Twitter", "🚨 Is the entire platform down? Server timeout Error 504 on login! @BrandDown", "SysAdminSam", "Outage"),
        ("Telegram", "App crashed and deleted my database project! Completely lost 6 hours of work! 🤬", "StudentCoder", "Outage"),
        ("Reddit", "API authentication failing across all endpoints. Zero communication from the engineering team.", "LeadDev", "Outage"),
        ("YouTube", "Cloud sync failure corrupted my files. This is unacceptable for enterprise users.", "EnterpriseGuy", "Outage"),
        ("Twitter", "Boycott this service! Security breach leaked passwords and they hid it for weeks!", "PrivacyAdvocate", "Outage")
    ]

    records = []
    base_time = datetime.datetime.now() - datetime.timedelta(hours=4)

    for i in range(count):
        # 75% normal, 25% crisis baseline
        chosen = random.choice(scenarios[:8] if random.random() < 0.72 else scenarios[8:])
        platform, raw_text, author_prefix, scenario_type = chosen

        timestamp = (base_time + datetime.timedelta(seconds=i * 25)).isoformat()
        author = f"{author_prefix}_{random.randint(10, 999)}"
        source_target = f"{platform} Stream"

        # Preprocess
        clean_text = preprocessor.clean_text(raw_text)

        # NLP Inference
        sent_res = sentiment_engine.analyze(clean_text)
        tox_res = toxicity_engine.analyze(clean_text)

        record = {
            "platform": platform,
            "source_target": source_target,
            "author": author,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "sentiment": sent_res["label"],
            "sentiment_score": sent_res["score"],
            "is_toxic": tox_res["is_toxic"],
            "toxicity_score": tox_res["toxicity_score"],
            "word_count": len(clean_text.split()),
            "char_length": len(raw_text),
            "created_at": timestamp
        }
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = RAW_DATA_DIR / "sample_crisis_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Generated {len(df)} records and saved to {csv_path}")

    # Populate SQLite database
    db.clear_all()
    inserted = db.insert_comments_batch(records)
    print(f"[Database] Inserted {inserted} records into SQLite database.")

    return df

if __name__ == "__main__":
    generate_multiplatform_dataset(400)
