from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "nifty50.db"
TOKEN_FILE = DATA_DIR / "upstox_token.txt"

INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
API_BASE = "https://api.upstox.com/v3"
CANDLE_UNIT = "minutes"
CANDLE_INTERVAL = "5"
REQUEST_TIMEOUT = 20

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def read_server_token():
    if not TOKEN_FILE.exists():
        return None
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return token or None


def write_server_token(token):
    token = str(token or "").strip()
    if not token:
        return None
    TOKEN_FILE.write_text(token, encoding="utf-8")
    return token


def clear_server_token():
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    return True
