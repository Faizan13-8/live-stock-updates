from datetime import date, timedelta
from urllib.parse import quote
import time
import requests

from config import (
    API_BASE, INSTRUMENT_KEY, CANDLE_UNIT, CANDLE_INTERVAL,
    REQUEST_TIMEOUT
)
from database import upsert_candles

_CACHE_TTL_SECONDS = 60
_CACHE = {}


def _cache_key(prefix, *parts):
    return prefix + "::" + "::".join(str(p) for p in parts)


def _read_cache(key):
    entry = _CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def _write_cache(key, value):
    _CACHE[key] = {"ts": time.time(), "value": value}

def _headers(token):
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

def _get(url, token):
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        detail = r.text[:500].replace("\n", " ")
        if r.status_code in (401, 403):
            raise RuntimeError("Upstox token expired/invalid. Test the token again.")
        raise RuntimeError(f"Upstox API {r.status_code}: {detail}")
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Upstox API returned: {body}")
    return body.get("data", {}).get("candles", [])

def _parse(rows):
    out=[]
    for row in rows:
        if len(row) < 7:
            continue
        try:
            ts = str(row[0])
            o,h,l,c = map(float, row[1:5])
            vol = float(row[5] or 0)
            oi = float(row[6] or 0)
            if not (l <= o <= h and l <= c <= h):
                continue
            out.append((ts,o,h,l,c,vol,oi))
        except (TypeError,ValueError):
            continue
    return out

def test_connection(token):
    # Current-day intraday endpoint is the lightest authenticated test.
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{API_BASE}/historical-candle/intraday/{key}/{CANDLE_UNIT}/{CANDLE_INTERVAL}"
    _get(url, token)
    return True

def fetch_intraday(token):
    cache_key = _cache_key("intraday", token)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{API_BASE}/historical-candle/intraday/{key}/{CANDLE_UNIT}/{CANDLE_INTERVAL}"
    rows = _parse(_get(url, token))
    _write_cache(cache_key, rows)
    return rows


def fetch_historical(token, from_date, to_date):
    cache_key = _cache_key("historical", token, from_date.isoformat(), to_date.isoformat())
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached
    key = quote(INSTRUMENT_KEY, safe="")
    url = (
        f"{API_BASE}/historical-candle/{key}/{CANDLE_UNIT}/"
        f"{CANDLE_INTERVAL}/{to_date:%Y-%m-%d}/{from_date:%Y-%m-%d}"
    )
    rows = _parse(_get(url, token))
    _write_cache(cache_key, rows)
    return rows


def fetch_range(token, days):
    """
    Fetch the requested range directly from Upstox.
    V3 limits 1-15 minute historical queries to one month, so
    ranges longer than one month are split into <=30-day chunks.
    """
    today = date.today()
    historical_end = today - timedelta(days=1)
    start = historical_end - timedelta(days=max(0, days-1))
    rows=[]
    cursor=start
    while cursor <= historical_end:
        end=min(cursor+timedelta(days=29), historical_end)
        rows.extend(fetch_historical(token, cursor, end))
        cursor=end+timedelta(days=1)
    # Deduplicate by timestamp while preserving chronological order.
    unique={r[0]:r for r in rows}
    return sorted(unique.values(), key=lambda x:x[0])

def download_history(token, progress=None, years=3):
    today=date.today()
    historical_end=today-timedelta(days=1)
    start=historical_end-timedelta(days=365*years)
    total=max(1,(historical_end-start).days+1)
    cursor=start
    done=0
    chunks=0
    saved=0
    while cursor <= historical_end:
        end=min(cursor+timedelta(days=29),historical_end)
        if progress:
            progress(
                message=f"Downloading {cursor} → {end}",
                progress=int(done/total*100),
                chunks_done=chunks,
                chunks_total=(total+29)//30,
                saved=saved,
            )
        rows=fetch_historical(token,cursor,end)
        saved += upsert_candles(rows)
        chunks += 1
        done += (end-cursor).days+1
        cursor=end+timedelta(days=1)
    if progress:
        progress(message=f"Downloaded {saved:,} candle rows",progress=100,
                 chunks_done=chunks,chunks_total=chunks,saved=saved)
    return saved
