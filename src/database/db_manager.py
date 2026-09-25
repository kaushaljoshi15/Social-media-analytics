"""
PulseGuard Database Manager
Covers GTU Unit 2 (File Handling, Exception Handling, SQLite/Database Operations)
"""
import sqlite3
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import DB_PATH

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row factory."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            print(f"[DB Error] Connection failed: {e}")
            raise

    def init_db(self) -> None:
        """Initialize SQLite schema for real-time comment and crisis telemetry."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Comments telemetry table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS comments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        platform TEXT NOT NULL,
                        source_target TEXT NOT NULL,
                        author TEXT,
                        raw_text TEXT NOT NULL,
                        clean_text TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        sentiment_score REAL NOT NULL,
                        is_toxic INTEGER NOT NULL,
                        toxicity_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Crisis alerts history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS crisis_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        platform TEXT NOT NULL,
                        alert_level TEXT NOT NULL,
                        z_score REAL NOT NULL,
                        p_value REAL NOT NULL,
                        negative_ratio REAL NOT NULL,
                        sample_size INTEGER NOT NULL,
                        root_causes TEXT,
                        triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            print(f"[DB Error] Schema initialization failed: {e}")

    def insert_comment(self, record: Dict[str, Any]) -> int:
        """Insert a single processed comment into the database."""
        sql = """
            INSERT INTO comments (
                platform, source_target, author, raw_text, clean_text,
                sentiment, sentiment_score, is_toxic, toxicity_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = record.get("created_at") or datetime.datetime.now().isoformat()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    record.get("platform", "Unknown"),
                    record.get("source_target", "Stream"),
                    record.get("author", "User"),
                    record.get("raw_text", ""),
                    record.get("clean_text", ""),
                    record.get("sentiment", "NEUTRAL"),
                    float(record.get("sentiment_score", 0.0)),
                    1 if record.get("is_toxic", False) else 0,
                    float(record.get("toxicity_score", 0.0)),
                    now
                ))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[DB Error] insert_comment failed: {e}")
            return -1

    def insert_comments_batch(self, records: List[Dict[str, Any]]) -> int:
        """Batch insert for high-throughput streaming (Unit 2)."""
        if not records:
            return 0
        sql = """
            INSERT INTO comments (
                platform, source_target, author, raw_text, clean_text,
                sentiment, sentiment_score, is_toxic, toxicity_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                r.get("platform", "Unknown"),
                r.get("source_target", "Stream"),
                r.get("author", "User"),
                r.get("raw_text", ""),
                r.get("clean_text", ""),
                r.get("sentiment", "NEUTRAL"),
                float(r.get("sentiment_score", 0.0)),
                1 if r.get("is_toxic", False) else 0,
                float(r.get("toxicity_score", 0.0)),
                r.get("created_at") or datetime.datetime.now().isoformat()
            )
            for r in records
        ]
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(sql, data)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            print(f"[DB Error] Batch insert failed: {e}")
            return 0

    def insert_alert(self, alert: Dict[str, Any]) -> int:
        """Record a crisis alert event in history."""
        sql = """
            INSERT INTO crisis_alerts (
                platform, alert_level, z_score, p_value,
                negative_ratio, sample_size, root_causes, triggered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    alert.get("platform", "All"),
                    alert.get("alert_level", "CRITICAL"),
                    float(alert.get("z_score", 0.0)),
                    float(alert.get("p_value", 1.0)),
                    float(alert.get("negative_ratio", 0.0)),
                    int(alert.get("sample_size", 0)),
                    alert.get("root_causes", ""),
                    alert.get("triggered_at") or datetime.datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[DB Error] insert_alert failed: {e}")
            return -1

    def get_recent_comments(self, limit: int = 100, platform: Optional[str] = None) -> pd.DataFrame:
        """Fetch the most recent comments as a Pandas DataFrame (Unit 4)."""
        try:
            with self.get_connection() as conn:
                if platform and platform != "All":
                    query = "SELECT * FROM comments WHERE platform = ? ORDER BY id DESC LIMIT ?"
                    df = pd.read_sql_query(query, conn, params=(platform, limit))
                else:
                    query = "SELECT * FROM comments ORDER BY id DESC LIMIT ?"
                    df = pd.read_sql_query(query, conn, params=(limit,))
                return df
        except Exception as e:
            print(f"[DB Error] get_recent_comments failed: {e}")
            return pd.DataFrame()

    def get_recent_alerts(self, limit: int = 10) -> pd.DataFrame:
        """Fetch recent crisis alert triggers."""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM crisis_alerts ORDER BY id DESC LIMIT ?"
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            print(f"[DB Error] get_recent_alerts failed: {e}")
            return pd.DataFrame()

    def clear_all(self) -> None:
        """Reset database tables for clean testing."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM comments")
            conn.execute("DELETE FROM crisis_alerts")
            conn.commit()
