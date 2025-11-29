# backend/datastore.py
import sqlite3
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

DB_PATH = "backend/soilsense.db"

@dataclass
class HistoryRecord:
    id: Optional[int]
    user_id: Optional[int]
    input_json: str
    recommended   : str
    timestamp: str

class DataStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._setup()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _setup(self):
        conn = self._conn()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                input_json TEXT,
                recommended TEXT,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_history(self, user_id, input_json, recommended):
        conn = self._conn()
        cur = conn.cursor()
        ts = datetime.utcnow().isoformat()

        cur.execute("""
            INSERT INTO history (user_id, input_json, recommended, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_id, input_json, recommended, ts))

        conn.commit()
        conn.close()

    def get_history(self, user_id: int) -> List[Dict]:
        conn = self._conn()
        cur = conn.cursor()

        cur.execute("SELECT id, input_json, recommended, timestamp FROM history WHERE user_id=? ORDER BY id DESC", (user_id,))
        rows = cur.fetchall()
        conn.close()

        return [
            {"id": r[0], "input": r[1], "recommended": r[2], "timestamp": r[3]} 
            for r in rows
        ]
