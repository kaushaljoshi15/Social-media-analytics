"""
PulseGuard Text Preprocessor & Data Cleaner
Covers GTU Unit 6 (Data Loading, Data Cleaning, Dealing with Missing Data,
Removing Duplicates, Filtering, String Manipulation)
"""
import re
import html
from typing import List, Dict, Any, Tuple
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        # Common social media emoji sentiment dictionary
        self.emoji_map = {
            "😡": " angry ",
            "🤬": " furious ",
            "🤮": " disgusting ",
            "👎": " bad ",
            "💔": " broken ",
            "🔥": " fire ",
            "❤️": " love ",
            "😍": " amazing ",
            "👍": " good ",
            "🎉": " celebration ",
            "💀": " dead ",
            "🤡": " clown ",
            "💩": " trash "
        }

    def clean_text(self, text: str) -> str:
        """
        Clean raw social media text:
        - Handle missing/NaN values
        - Decode HTML entities
        - Normalize emojis to sentiment words
        - Remove URLs and user mentions (@handle)
        - Remove special symbols while preserving punctuation context
        - Normalize excessive whitespaces
        """
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        # Decode HTML entities (e.g. &amp; -> &)
        cleaned = html.unescape(text)

        # Map emojis to word equivalents
        for emoji, word in self.emoji_map.items():
            cleaned = cleaned.replace(emoji, word)

        # Remove URLs
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)

        # Remove social media user handles (e.g. @elonmusk)
        cleaned = re.sub(r'@\w+', '', cleaned)

        # Remove standalone hashtags symbol but keep the word (#boycott -> boycott)
        cleaned = re.sub(r'#(\w+)', r'\1', cleaned)

        # Remove non-ASCII weird artifacts while keeping standard alphanumeric and punctuation
        cleaned = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', cleaned)

        # Normalize repeated punctuation (e.g. '?????' -> '?')
        cleaned = re.sub(r'([!?.]){2,}', r'\1', cleaned)

        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def clean_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Comprehensive data wrangling pipeline on DataFrame (GTU Unit 6):
        - Handles missing values (dropna / fillna)
        - Removes duplicates (bot spam)
        - Appends cleaned text and text length features
        """
        if df.empty:
            return df

        # Handle missing data (Unit 6)
        df_clean = df.dropna(subset=[text_column]).copy()

        # Remove empty string or whitespace-only records
        df_clean = df_clean[df_clean[text_column].str.strip().str.len() > 0]

        # Duplicate handling (deduplication of repetitive bot spam)
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=[text_column])
        duplicate_count = initial_count - len(df_clean)

        # Feature creation: clean_text, char_length, word_count
        df_clean["clean_text"] = df_clean[text_column].apply(self.clean_text)
        df_clean["char_length"] = df_clean[text_column].str.len()
        df_clean["word_count"] = df_clean["clean_text"].apply(lambda s: len(s.split()))

        # Filter out rows that became empty after cleaning
        df_clean = df_clean[df_clean["clean_text"].str.len() > 0].reset_index(drop=True)

        return df_clean
