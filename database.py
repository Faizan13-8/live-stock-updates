import sqlite3
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS nifty_5min (
    timestamp TEXT PRIMARY KEY,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    open_interest REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_nifty_5min_timestamp
ON nifty_5min(timestamp);
"""

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SCHEMA)
        con.commit()

def upsert_candles(rows):
    init_db()
    if not rows:
        return 0
    sql = """
    INSERT INTO nifty_5min(timestamp,open,high,low,close,volume,open_interest)
    VALUES(?,?,?,?,?,?,?)
    ON CONFLICT(timestamp) DO UPDATE SET
      open=excluded.open, high=excluded.high, low=excluded.low,
      close=excluded.close, volume=excluded.volume,
      open_interest=excluded.open_interest
    """
    with sqlite3.connect(DB_PATH) as con:
        con.executemany(sql, rows)
        con.commit()
    return len(rows)

def get_stats():
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM nifty_5min"
        ).fetchone()
    return {"count": int(row[0] or 0), "first": row[1], "last": row[2]}

def load_candles(limit=6000):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("""
            SELECT timestamp,open,high,low,close,volume,open_interest
            FROM nifty_5min ORDER BY timestamp ASC LIMIT ?
        """, (int(limit),)).fetchall()
