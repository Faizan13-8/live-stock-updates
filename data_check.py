from config import DB_PATH
from database import init_db, get_stats

def check_database():
    try:
        init_db()
        s = get_stats()
        ok = s["count"] > 0
        return {
            "ok": ok,
            "database_exists": DB_PATH.exists(),
            "count": s["count"],
            "first": s["first"],
            "last": s["last"],
            "message": "Database OK" if ok else "Database empty. Download historical data first."
        }
    except Exception as e:
        return {"ok": False, "database_exists": DB_PATH.exists(), "count": 0, "message": str(e)}
