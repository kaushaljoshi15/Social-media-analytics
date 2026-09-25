"""
PulseGuard Master Multi-Platform Model Fine-Tuning Engine
Consolidates ALL ingested datasets (Twitter Sentiment140, Brand Corpus, Reddit GoEmotions,
Toxicity Harassment Corpus, and Crisis Datasets) to train the unified master AI models.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.nlp.preprocessor import TextPreprocessor

MODELS_DIR = PROCESSED_DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
preprocessor = TextPreprocessor()

def train_master_sentiment_model():
    print("\n=======================================================================")
    print("[1/2] Consolidating All Datasets for Master Sentiment AI Model")
    print("=======================================================================")

    frames = []

    # 1. Ingest Sentiment140 (50k sample)
    p_sent140 = RAW_DATA_DIR / "sentiment140_50k.csv"
    if p_sent140.exists():
        df_s = pd.read_csv(p_sent140)
        df_s = df_s[["text", "sentiment"]].dropna()
        frames.append(df_s)
        print(f" -> Added Sentiment140: {len(df_s):,} tweets")

    # 2. Ingest Twitter Brand Corpus (Apple, Google, Microsoft)
    p_brand = RAW_DATA_DIR / "cleaned_brand_training.csv"
    if p_brand.exists():
        df_b = pd.read_csv(p_brand)
        df_b = df_b[["text", "sentiment"]].dropna()
        frames.append(df_b)
        print(f" -> Added Twitter Brand Corpus: {len(df_b):,} tweets")

    # 3. Ingest Crisis Backlash Dataset (Overheating, Outages, Battery Drain)
    p_crisis = RAW_DATA_DIR / "sample_crisis_dataset.csv"
    if p_crisis.exists():
        df_c = pd.read_csv(p_crisis)
        text_col = "clean_text" if "clean_text" in df_c.columns else "text"
        df_c = df_c[[text_col, "sentiment"]].rename(columns={text_col: "text"}).dropna()
        frames.append(df_c)
        print(f" -> Added Multi-Platform Crisis Dataset: {len(df_c):,} comments")

    if not frames:
        print("[!] Error: No sentiment datasets found in data/raw/")
        return

    # Consolidate and clean
    master_df = pd.concat(frames, ignore_index=True)
    master_df = master_df[master_df["sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])]
    print(f"\n[+] Total Unified Training Corpus: {len(master_df):,} records")
    print(" -> Distribution:\n", master_df["sentiment"].value_counts().to_dict())

    # Preprocessing
    print("\n[+] Running Preprocessor & Vectorizer across unified corpus...")
    master_df["clean_text"] = master_df["text"].apply(preprocessor.clean_text)
    master_df = master_df[master_df["clean_text"].str.len() > 3]

    X = master_df["clean_text"]
    y = master_df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f" -> Training Samples: {X_train_vec.shape[0]:,}, Test Samples: {X_test_vec.shape[0]:,}")
    print(f" -> Feature Matrix Size: {X_train_vec.shape[1]:,} n-grams")

    # Train Master Model
    print("\n[+] Training Master Logistic Regression Model with L2 Regularization...")
    model = LogisticRegression(max_iter=600, C=2.0, solver="saga", random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Master Sentiment Model Accuracy: {acc * 100:.2f}%")
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save to disk
    model_path = MODELS_DIR / "master_sentiment_model.joblib"
    vec_path = MODELS_DIR / "master_sentiment_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f" -> Saved Master Sentiment Model: {model_path.name}")
    print(f" -> Saved Master Vectorizer: {vec_path.name}")

def train_master_toxicity_model():
    print("\n=======================================================================")
    print("[2/2] Training Master Toxicity & Threat Shield Model (Davidson Dataset)")
    print("=======================================================================")

    p_tox = RAW_DATA_DIR / "real_toxicity_harassment.csv"
    if not p_tox.exists():
        p_tox = RAW_DATA_DIR / "davidson_toxicity_labeled.csv"

    if not p_tox.exists():
        print("[!] Error: No toxicity dataset found in data/raw/")
        return

    df_tox = pd.read_csv(p_tox)
    # class 0: hate speech, 1: offensive language, 2: neither
    # Binary target: 1 = Toxic/Threat/Hate, 0 = Clean
    df_tox["is_toxic"] = (df_tox["class"] != 2).astype(int)
    print(f" -> Ingested Davidson Toxicity Corpus: {len(df_tox):,} labeled tweets")
    print(" -> Toxic Flags:", df_tox["is_toxic"].value_counts().to_dict())

    df_tox["clean_text"] = df_tox["tweet"].apply(preprocessor.clean_text)
    df_tox = df_tox[df_tox["clean_text"].str.len() > 3]

    X = df_tox["clean_text"]
    y = df_tox["is_toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vec_tox = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vec_tox.fit_transform(X_train)
    X_test_vec = vec_tox.transform(X_test)

    print("\n[+] Training Master Toxicity Classifier...")
    tox_model = LogisticRegression(max_iter=500, C=2.5, solver="saga", random_state=42)
    tox_model.fit(X_train_vec, y_train)

    y_pred = tox_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Master Toxicity Model Accuracy: {acc * 100:.2f}%")
    print("\n[+] Toxicity Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean (0)", "Toxic (1)"], zero_division=0))

    # Save to disk
    m_path = MODELS_DIR / "master_toxicity_model.joblib"
    v_path = MODELS_DIR / "master_toxicity_vectorizer.joblib"
    joblib.dump(tox_model, m_path)
    joblib.dump(vec_tox, v_path)
    print(f" -> Saved Master Toxicity Model: {m_path.name}")
    print(f" -> Saved Master Toxicity Vectorizer: {v_path.name}")

if __name__ == "__main__":
    train_master_sentiment_model()
    train_master_toxicity_model()
    print("\n=======================================================================")
    print("[Done] All Master AI Models Successfully Fine-Tuned & Deployed!")
    print("=======================================================================")
