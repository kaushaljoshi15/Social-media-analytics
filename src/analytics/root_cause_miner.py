"""
PulseGuard Root-Cause Aspect Miner
Covers GTU Unit 4 (Scikit-Learn Machine Learning: TF-IDF Vectorizer + K-Means Clustering)
Extracts culprit bigrams and groups negative customer comments into thematic clusters.
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, KMEANS_CLUSTERS

class RootCauseMiner:
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES, n_clusters: int = KMEANS_CLUSTERS):
        self.max_features = max_features
        self.n_clusters = n_clusters
        # Domain stop words to remove generic social media filler
        self.stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
            "video", "post", "comment", "channel", "people", "really", "just", "like", "get", "make"
        ]

    def extract_top_culprits(self, texts: List[str], top_n: int = 8) -> List[Dict[str, Any]]:
        """Extract highest-weighted TF-IDF keywords and bi-grams from negative comments."""
        if not texts or len(texts) < 2:
            return []

        try:
            vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=TFIDF_NGRAM_RANGE,
                stop_words=self.stop_words
            )
            tfidf_matrix = vec.fit_transform(texts)
            feature_names = np.array(vec.get_feature_names_out())
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

            top_indices = np.argsort(scores)[::-1][:top_n]
            results = []
            for idx in top_indices:
                if scores[idx] > 0.01:
                    results.append({
                        "term": str(feature_names[idx]),
                        "tfidf_score": round(float(scores[idx]), 3),
                        "is_bigram": " " in str(feature_names[idx])
                    })
            return results
        except Exception as e:
            print(f"[RootCauseMiner] Error in extract_top_culprits: {e}")
            return []

    def cluster_root_causes(self, texts: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        Unsupervised K-Means clustering (GTU Unit 4) to group complaints into root-cause themes.
        """
        if not texts or len(texts) < k:
            return []

        try:
            k = min(k, len(texts))
            vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=TFIDF_NGRAM_RANGE,
                stop_words=self.stop_words
            )
            X = vec.fit_transform(texts)
            feature_names = np.array(vec.get_feature_names_out())

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)

            clusters = []
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

            labels = kmeans.labels_
            for i in range(k):
                cluster_comment_count = int(np.sum(labels == i))
                top_terms = [str(feature_names[ind]) for ind in order_centroids[i, :4]]
                theme_name = " / ".join(top_terms[:2]).title()
                clusters.append({
                    "cluster_id": i + 1,
                    "theme": theme_name,
                    "comment_count": cluster_comment_count,
                    "top_keywords": top_terms
                })

            return sorted(clusters, key=lambda c: c["comment_count"], reverse=True)
        except Exception as e:
            print(f"[RootCauseMiner] Error in cluster_root_causes: {e}")
            return []
