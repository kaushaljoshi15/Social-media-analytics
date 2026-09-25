"""
PulseGuard Scikit-Learn Machine Learning Benchmarks
Covers GTU Unit 4 (Basic Functionalities of Machine Learning using Scikit-Learn)
Implements Train/Test split, TF-IDF Vectorization, Multinomial Naive Bayes,
Logistic Regression, Confusion Matrix, and Precision/Recall/F1 evaluation.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class MLBenchmarkPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        self.nb_model = MultinomialNB()
        self.lr_model = LogisticRegression(max_iter=200, random_state=42)
        self.is_trained = False
        self.metrics = {}

    def train_and_evaluate(self, df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "sentiment") -> Dict[str, Any]:
        """
        Train both Naive Bayes and Logistic Regression on dataset and compute GTU metrics (Unit 4).
        """
        if df.empty or len(df) < 20:
            return {"status": "Insufficient data for ML training"}

        # Filter valid classes
        valid_df = df[df[label_col].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])].copy()
        if len(valid_df[label_col].unique()) < 2:
            return {"status": "Requires at least 2 distinct classes"}

        X = valid_df[text_col]
        y = valid_df[label_col]

        # Train-Test Split (80/20) (GTU Unit 4)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # TF-IDF Feature Extraction (NumPy array transformation)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # 1. Train Multinomial Naive Bayes
        self.nb_model.fit(X_train_vec, y_train)
        nb_preds = self.nb_model.predict(X_test_vec)
        nb_acc = accuracy_score(y_test, nb_preds)
        nb_prec, nb_rec, nb_f1, _ = precision_recall_fscore_support(y_test, nb_preds, average='weighted', zero_division=0)
        nb_cm = confusion_matrix(y_test, nb_preds)

        # 2. Train Logistic Regression
        self.lr_model.fit(X_train_vec, y_train)
        lr_preds = self.lr_model.predict(X_test_vec)
        lr_acc = accuracy_score(y_test, lr_preds)
        lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average='weighted', zero_division=0)
        lr_cm = confusion_matrix(y_test, lr_preds)

        self.is_trained = True
        self.metrics = {
            "classes": list(self.nb_model.classes_),
            "sample_size": len(valid_df),
            "train_count": len(X_train),
            "test_count": len(X_test),
            "naive_bayes": {
                "accuracy": round(float(nb_acc), 4),
                "precision": round(float(nb_prec), 4),
                "recall": round(float(nb_rec), 4),
                "f1_score": round(float(nb_f1), 4),
                "confusion_matrix": nb_cm.tolist()
            },
            "logistic_regression": {
                "accuracy": round(float(lr_acc), 4),
                "precision": round(float(lr_prec), 4),
                "recall": round(float(lr_rec), 4),
                "f1_score": round(float(lr_f1), 4),
                "confusion_matrix": lr_cm.tolist()
            }
        }
        return self.metrics

    def predict_sentiment(self, text: str, model_type: str = "logistic_regression") -> Tuple[str, float]:
        """Classify single string using trained Scikit-learn model."""
        if not self.is_trained:
            return "NEUTRAL", 0.5
        vec = self.vectorizer.transform([text])
        model = self.lr_model if model_type == "logistic_regression" else self.nb_model
        pred = model.predict(vec)[0]
        prob = float(np.max(model.predict_proba(vec)))
        return pred, round(prob, 4)
