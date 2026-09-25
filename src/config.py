"""
PulseGuard Configuration & Global Constants
Covers GTU Unit 1 (Variables, Data Types, Dictionaries, Modules)
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DATA_DIR / "pulseguard.db"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Supported Platforms
PLATFORMS = ["YouTube", "Twitter", "Reddit", "Telegram"]

# Statistical Anomaly Detection Thresholds (Unit 3 & 5)
BASELINE_NEGATIVE_RATIO = 0.12     # Normal baseline: ~12% negative comments
CRISIS_Z_THRESHOLD = 2.0           # Z-Score >= 2.0 -> Elevated Warning
CRITICAL_Z_THRESHOLD = 2.8         # Z-Score >= 2.8 -> Red Emergency Crisis
HYPOTHESIS_ALPHA = 0.01            # 99% confidence level for t-test (p < 0.01)

# Probabilistic Parameters (Unit 5: Poisson & Exponential)
POISSON_LAMBDA_NORMAL = 3.5        # Avg comments/min during normal state
POISSON_LAMBDA_CRISIS = 32.0       # Avg comments/min during crisis spike
EXP_SCALE_NORMAL = 17.0            # Avg seconds between negative comments (normal)
EXP_SCALE_CRISIS = 1.8             # Avg seconds between negative comments (crisis)

# Sliding Window Settings (Unit 6: Wrangling & Slicing)
SLIDING_WINDOW_SIZE = 50           # Number of recent comments for rolling stats
MIN_BATCH_FOR_TESTING = 15         # Minimum observations required for t-test

# TF-IDF & Root-Cause Parameters (Unit 4: Scikit-learn)
TFIDF_MAX_FEATURES = 150
TFIDF_NGRAM_RANGE = (1, 2)         # Unigrams and Bigrams (e.g. "battery drain")
KMEANS_CLUSTERS = 3                # Default root-cause clusters

# UI Color Palette (Executive Dark Theme)
UI_THEME = {
    "background": "#0B0F19",
    "card_bg": "#151C2C",
    "safe_green": "#10B981",
    "warning_amber": "#F59E0B",
    "danger_red": "#EF4444",
    "accent_blue": "#3B82F6",
    "text_primary": "#F9FAFB",
    "text_secondary": "#9CA3AF"
}
