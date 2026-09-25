import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import BASE_DIR, PROCESSED_DATA_DIR
from src.nlp.preprocessor import TextPreprocessor

class ModelFineTuner:
    def __init__(self, output_dir: Path = PROCESSED_DATA_DIR / "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )

    def run_pipeline(
        self,
        csv_path: str,
        text_col: str = "text",
        label_col: str = "sentiment",
        model_type: str = "logistic_regression"
    ):
        """
        Executes the complete 6-step KDD lifecycle on any input dataset.
        """
        print(f"\n=======================================================")
        print(f"[*] PulseGuard Fine-Tuning Pipeline: {Path(csv_path).name}")
        print(f"=======================================================")

        # 1. Data Integration & Selection
        print(f"[Stage 1/6] Loading and selecting data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f" -> Raw dataset shape: {df.shape}")

        if text_col not in df.columns:
            # Fallback column search
            potential = [c for c in df.columns if any(k in c.lower() for k in ["text", "comment", "tweet", "content"])]
            if potential:
                text_col = potential[0]
                print(f" -> Auto-selected text column: '{text_col}'")
            else:
                raise ValueError(f"Could not find text column in {df.columns}")

        # 2. Cleaning & Preprocessing (Unstructured -> Structured Clean)
        print(f"[Stage 2/6] Cleaning text and removing duplicates (GTU Unit 6)...")
        df_clean = self.preprocessor.clean_dataframe(df, text_column=text_col)
        print(f" -> Cleaned dataset size: {len(df_clean)} records")

        # 3. Transformation & Train/Test Split
        print(f"[Stage 3/6] Transforming text to TF-IDF feature space (GTU Unit 4)...")
        X = df_clean["clean_text"]
        y = df_clean[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f" -> Vocabulary size: {len(self.vectorizer.vocabulary_)} features")
        print(f" -> Training samples: {X_train_vec.shape[0]}, Test samples: {X_test_vec.shape[0]}")

        # 4. Data Mining & Model Training
        print(f"[Stage 4/6] Training {model_type.upper()} model...")
        if model_type == "naive_bayes":
            model = MultinomialNB(alpha=0.1)
        else:
            model = LogisticRegression(max_iter=500, C=1.5, random_state=42)

        model.fit(X_train_vec, y_train)

        # 5. Pattern Evaluation (GTU Unit 4 & 5)
        print(f"[Stage 5/6] Evaluating model performance...")
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n[+] Overall Accuracy: {acc * 100:.2f}%")
        print("\n[+] Detailed Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("[+] Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # 6. Deployment (Save Artifacts)
        print(f"\n[Stage 6/6] Saving trained model and vectorizer for deployment...")
        model_path = self.output_dir / f"{model_type}_pulseguard.joblib"
        vec_path = self.output_dir / "tfidf_vectorizer.joblib"
        joblib.dump(model, model_path)
        joblib.dump(self.vectorizer, vec_path)
        print(f" -> Saved model to: {model_path}")
        print(f" -> Saved vectorizer to: {vec_path}")
        print("\n[Done] Fine-Tuning Pipeline Complete! Ready for Real-Time Inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseGuard Model Fine-Tuning Pipeline")
    parser.add_argument("--data", type=str, default="data/raw/sample_crisis_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--text_col", type=str, default="clean_text", help="Text column name")
    parser.add_argument("--label_col", type=str, default="sentiment", help="Label column name")
    parser.add_argument("--model", type=str, choices=["logistic_regression", "naive_bayes"], default="logistic_regression")

    args = parser.parse_args()
    tuner = ModelFineTuner()
    tuner.run_pipeline(
        csv_path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
        model_type=args.model
    )
