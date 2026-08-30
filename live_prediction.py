
import json
import sqlite3
import time
from datetime import datetime, date, time as dt_time
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from config import (
    API_BASE, INSTRUMENT_KEY, CANDLE_UNIT, CANDLE_INTERVAL, REQUEST_TIMEOUT, MODEL_DIR, DB_PATH
)
from features import make_features, clean_X
from pattern_features import add_pattern_features, pattern_name_from_row
from chart_patterns import detect_chart_structures
from model_feedback import calibrate_live_forecast

INTRADAY_URL = f"{API_BASE}/historical-candle/intraday"
QUOTE_URL = f"{API_BASE.replace('/v3', '/v2')}/market-quote/quotes"
NEWS_URL = f"{API_BASE.replace('/v3', '/v2')}/news"

MODEL_5M = MODEL_DIR / "nifty_v5_5m_points.pkl"
MODEL_10M = MODEL_DIR / "nifty_v5_10m_points.pkl"
MODEL_DIR_MODEL = MODEL_DIR / "nifty_v5_5m_direction.pkl"
FEATURE_FILE = MODEL_DIR / "feature_columns.json"
PATTERN_MODEL = MODEL_DIR / "nifty_v5_pattern_direction.pkl"
PATTERN_STATS_FILE = MODEL_DIR / "nifty_v5_pattern_stats.json"
IST = ZoneInfo("Asia/Kolkata")

# NSE quotes NIFTY derivatives in 0.05 increments. Every price shown as an
# actionable level is snapped to this grid so it can be entered as-is.
TICK_SIZE = 0.05

# Share of the confluence score owned by the model itself (classifier plus the
# two point regressors). The remainder goes to the technical/news/depth context.
# Above 0.5 so the confirmation layer cannot invert a confident model call.
MODEL_WEIGHT = 0.70

# Trade signals require BOTH: raw out-of-sample directional accuracy at or above
# this floor, AND an edge over the majority-class baseline of at least
# MIN_EDGE_SIGMAS standard errors.
MIN_VALIDATION_DIRECTIONAL = 0.54
MIN_EDGE_SIGMAS = 2.0

# Widest stop the level builder may produce, as a multiple of ATR.
MAX_RISK_ATR_MULT = 2.5

_NEWS_CACHE = {"ts": 0.0, "items": [], "sentiment": 0.0}
_NEWS_CACHE_TTL_SECONDS = 300


def _headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


def _fetch_today(token):
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{INTRADAY_URL}/{key}/{CANDLE_UNIT}/{CANDLE_INTERVAL}"
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Upstox intraday API error {r.status_code}: {r.text[:400]}")
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Unexpected Upstox intraday response: {body}")
    rows = []
    for row in body.get("data", {}).get("candles", []):
        if len(row) < 5:
            continue
        try:
            ts = pd.to_datetime(row[0], errors="coerce")
            o, h, l, c = map(float, row[1:5])
            vol = float(row[5] or 0) if len(row) > 5 else 0.0
            oi = float(row[6] or 0) if len(row) > 6 else 0.0
            if pd.isna(ts) or not (l <= o <= h and l <= c <= h):
                continue
            rows.append({
                "timestamp": ts.isoformat(),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": vol,
                "open_interest": oi,
            })
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def _load_recent_db(limit=1200):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        d = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume, open_interest FROM nifty_5min ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    return d.sort_values("timestamp")


def _merge_frames(*frames):
    fs = [x for x in frames if x is not None and not x.empty]
    if not fs:
        return pd.DataFrame()
    d = pd.concat(fs, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "open_interest"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    d = d.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    d = d[(d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0)]
    d = d[(d["low"] <= d["open"]) & (d["open"] <= d["high"]) & (d["low"] <= d["close"]) & (d["close"] <= d["high"])]
    return d.reset_index(drop=True)


def get_live_frame(token):
    return _merge_frames(_load_recent_db(1200), _fetch_today(token))


def _fetch_full_quote(token):
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{QUOTE_URL}?instrument_key={key}"
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Upstox quote API error {r.status_code}: {r.text[:300]}")
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Unexpected quote response: {body}")
    data = body.get("data", {})
    if not data:
        return {}
    raw = next(iter(data.values()))
    depth = raw.get("depth", {}) or {}
    buy = depth.get("buy", []) or []
    sell = depth.get("sell", []) or []
    bid_qty = sum(float(x.get("quantity", 0) or 0) for x in buy[:5])
    ask_qty = sum(float(x.get("quantity", 0) or 0) for x in sell[:5])
    total_buy = float(raw.get("total_buy_quantity", 0) or 0)
    total_sell = float(raw.get("total_sell_quantity", 0) or 0)
    if bid_qty + ask_qty <= 0:
        bid_qty, ask_qty = total_buy, total_sell
    imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.0
    return {
        "last_price": float(raw.get("last_price")) if raw.get("last_price") is not None else None,
        "timestamp": raw.get("timestamp") or raw.get("last_trade_time"),
        "volume": float(raw.get("volume", 0) or 0),
        "average_price": float(raw.get("average_price", 0) or 0),
        "total_buy_quantity": total_buy,
        "total_sell_quantity": total_sell,
        "bid5_quantity": bid_qty,
        "ask5_quantity": ask_qty,
        "depth_imbalance": float(imbalance),
        "depth_supported": bool(bid_qty + ask_qty > 0),
    }


def _headline_sentiment(text):
    t = str(text).lower()
    positive = ("beat", "upgrade", "growth", "surge", "rally", "strong", "positive", "easing", "record", "inflow", "profit", "optimism", "gain")
    negative = ("fall", "drop", "downgrade", "weak", "risk", "war", "inflation", "hawkish", "outflow", "loss", "lawsuit", "crash", "volatility")
    p = sum(1 for w in positive if w in t)
    n = sum(1 for w in negative if w in t)
    if p + n == 0:
        return 0.0
    return float(np.clip((p - n) / (p + n), -1, 1))


def _fetch_news(token):
    global _NEWS_CACHE
    now = time.time()
    if now - _NEWS_CACHE["ts"] < _NEWS_CACHE_TTL_SECONDS:
        return _NEWS_CACHE
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{NEWS_URL}?category=instrument_keys&instrument_keys={key}"
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        _NEWS_CACHE = {"ts": now, "items": [], "sentiment": 0.0}
        return _NEWS_CACHE
    try:
        body = r.json()
    except ValueError:
        return _NEWS_CACHE
    items = body.get("data", {}).get("news", []) if isinstance(body.get("data"), dict) else []
    parsed = []
    scores = []
    for x in items[:20]:
        title = x.get("title") or x.get("headline") or x.get("name") or ""
        if not title:
            continue
        score = _headline_sentiment(title)
        scores.append(score)
        parsed.append({"title": title, "published_at": x.get("published_at") or x.get("timestamp"), "sentiment": round(score, 3)})
    sentiment = float(np.mean(scores)) if scores else 0.0
    _NEWS_CACHE = {"ts": now, "items": parsed, "sentiment": sentiment}
    return _NEWS_CACHE


def _technical_context(df):
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    ema5 = c.ewm(span=5, adjust=False).mean().iloc[-1]
    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    slope = np.polyfit(np.arange(min(20, len(c))), c.tail(20), 1)[0] if len(c) >= 3 else 0.0
    trend = "STRONG UP" if ema5 > ema20 > ema50 and slope > 0 else "UP" if ema5 > ema20 else "STRONG DOWN" if ema5 < ema20 < ema50 and slope < 0 else "DOWN" if ema5 < ema20 else "SIDEWAYS"
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    rsi = rsi if np.isfinite(rsi) else 50.0
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    atr = atr if np.isfinite(atr) and atr > 0 else float((h - l).tail(20).mean())
    typical = (h + l + c) / 3
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    # VWAP is a session statistic: it must restart at each open, not run across
    # the whole 1200-candle frame. And it is only meaningful when there is real
    # volume — the NIFTY 50 index itself publishes none (every row is 0), in
    # which case the honest answer is None rather than silently returning the
    # current price and letting vwap_bias look like a signal.
    if float(vol.sum()) > 0:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        session = ts.dt.date
        cum_pv = (typical * vol).groupby(session).cumsum()
        cum_v = vol.groupby(session).cumsum()
        last_v = float(cum_v.iloc[-1])
        vwap = float(cum_pv.iloc[-1] / last_v) if last_v > 0 else None
    else:
        vwap = None
    support = float(l.tail(30).min())
    resistance = float(h.tail(30).max())
    return {
        "trend": trend,
        "rsi": rsi,
        "atr": atr,
        "vwap": vwap,
        "has_volume": bool(float(vol.sum()) > 0),
        "support": support,
        "resistance": resistance,
        "ema5": float(ema5),
        "ema20": float(ema20),
        "ema50": float(ema50),
    }


def detect_support_resistance(df, lookback=30):
    if df is None or df.empty:
        return None, None, {"support": None, "resistance": None, "support_points": [], "resistance_points": []}

    recent = df.tail(lookback).copy()
    highs = pd.to_numeric(recent["high"], errors="coerce").astype(float)
    lows = pd.to_numeric(recent["low"], errors="coerce").astype(float)
    if highs.empty or lows.empty:
        return None, None, {"support": None, "resistance": None, "support_points": [], "resistance_points": []}

    idx = recent.index.to_numpy()
    recent_highs = highs.to_numpy()
    recent_lows = lows.to_numpy()
    support_candidates = []
    resistance_candidates = []

    if len(recent_lows) >= 3:
        for i in range(1, len(recent_lows) - 1):
            if recent_lows[i] <= recent_lows[i - 1] and recent_lows[i] <= recent_lows[i + 1]:
                support_candidates.append((idx[i], float(recent_lows[i])))
    if len(recent_highs) >= 3:
        for i in range(1, len(recent_highs) - 1):
            if recent_highs[i] >= recent_highs[i - 1] and recent_highs[i] >= recent_highs[i + 1]:
                resistance_candidates.append((idx[i], float(recent_highs[i])))

    if support_candidates:
        support = float(min(support_candidates, key=lambda x: x[1])[1])
        support_points = [(int(np.asarray([p[0] for p in support_candidates]).min()), support)]
    else:
        support = float(lows.min())
        support_points = [(int(recent.index.min()), support)]

    if resistance_candidates:
        resistance = float(max(resistance_candidates, key=lambda x: x[1])[1])
        resistance_points = [(int(np.asarray([p[0] for p in resistance_candidates]).max()), resistance)]
    else:
        resistance = float(highs.max())
        resistance_points = [(int(recent.index.max()), resistance)]

    meta = {
        "support": support,
        "resistance": resistance,
        "support_points": support_points,
        "resistance_points": resistance_points,
    }
    return support, resistance, meta


def build_auto_pattern_context(df, pattern_name=None, lookback=40):
    if df is None or df.empty:
        return {"pattern": pattern_name or "NONE", "support": None, "resistance": None, "trendline": [], "levels": []}

    recent = df.tail(lookback).copy()
    closes = pd.to_numeric(recent["close"], errors="coerce").astype(float).to_numpy()
    support, resistance, support_meta = detect_support_resistance(recent, lookback=lookback)

    trendline = []
    if len(closes) >= 2:
        xs = np.arange(len(closes))
        slope, intercept = np.polyfit(xs, closes, 1)
        x0 = xs[0]
        x1 = xs[-1]
        y0 = intercept + slope * x0
        y1 = intercept + slope * x1
        trendline = [{"x": int(x0), "y": float(y0)}, {"x": int(x1), "y": float(y1)}]

    levels = []
    if support is not None:
        levels.append({"type": "support", "value": float(support)})
    if resistance is not None:
        levels.append({"type": "resistance", "value": float(resistance)})

    if pattern_name and pattern_name.upper() in {"BREAKOUT_UP", "BREAKOUT_DOWN", "BULLISH_ENGULFING", "BEARISH_ENGULFING"}:
        current = float(recent["close"].iloc[-1])
        if pattern_name.upper() in {"BREAKOUT_UP", "BULLISH_ENGULFING"}:
            levels.append({"type": "breakout", "value": float(resistance)})
            levels.append({"type": "entry", "value": float(current)})
        elif pattern_name.upper() in {"BREAKOUT_DOWN", "BEARISH_ENGULFING"}:
            levels.append({"type": "breakdown", "value": float(support)})
            levels.append({"type": "entry", "value": float(current)})

    return {
        "pattern": (pattern_name or "NONE").upper(),
        "support": support,
        "resistance": resistance,
        "trendline": trendline,
        "levels": levels,
        "support_meta": support_meta,
    }


def _market_regime(df):
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    returns = close.pct_change().fillna(0.0)
    vol = returns.rolling(20, min_periods=5).std().iloc[-1]
    trend_strength = (close.iloc[-1] - close.rolling(20, min_periods=5).mean().iloc[-1]) / max(close.rolling(14, min_periods=5).std().iloc[-1], 1e-9)
    if trend_strength > 1.0 and vol > 0.0004:
        regime = "TRENDING_UP"
    elif trend_strength < -1.0 and vol > 0.0004:
        regime = "TRENDING_DOWN"
    elif abs(trend_strength) < 0.5:
        regime = "RANGE"
    else:
        regime = "TRANSITION"
    return {"regime": regime, "trend_strength": float(trend_strength), "volatility": float(vol)}


def _build_training_compatible_features(df):
    """Build exactly the feature columns used by train_model.py V5.1."""
    x, _ = make_features(df)
    x, labels = add_pattern_features(x)
    c = pd.to_numeric(x["close"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    o = pd.to_numeric(x["open"], errors="coerce")
    r = c.pct_change()
    for n in (1, 2, 3, 5, 10, 20):
        x[f"ret_lag_{n}"] = c.pct_change(n)
        x[f"close_lag_pct_{n}"] = c / c.shift(n) - 1.0
    x["high_dist_1"] = h / c - 1.0
    x["low_dist_1"] = l / c - 1.0
    x["open_dist_1"] = o / c - 1.0
    for n in (3, 5, 10):
        x[f"ret_mean_{n}"] = r.rolling(n).mean()
        x[f"ret_std_{n}"] = r.rolling(n).std()
    ts = pd.to_datetime(x["timestamp"], errors="coerce")
    mins = ts.dt.hour * 60 + ts.dt.minute
    x["time_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    x["time_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    return x, labels


def _make_input(df):
    x, labels = _build_training_compatible_features(df)
    cols = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    clean = clean_X(x, cols)
    return clean, cols, labels


def _pressure(df):
    recent = df.tail(20)
    close = recent["close"].astype(float)
    open_ = recent["open"].astype(float)
    vol = recent["volume"].fillna(0).astype(float)
    signed = np.sign(close - open_)
    up = float(vol.where(signed > 0, 0).sum())
    down = float(vol.where(signed < 0, 0).sum())
    total = up + down
    if total > 0:
        return 100 * up / total, 100 * down / total, "candle-volume proxy"
    body = (close - open_).fillna(0)
    score = float(body.sum())
    bp = float(np.clip(50 + score / max(float(body.abs().sum()), 1e-9) * 50, 0, 100))
    return bp, 100 - bp, "price-action proxy"


def _snap(price):
    """Round to the NSE tick grid.

    NIFTY derivatives quote in 0.05 increments. round(x, 2) produced levels like
    2.13 or 2.17 that cannot be entered as a limit order, so a stop shown on
    screen would not be the stop that actually got placed.
    """
    return round(round(float(price) / TICK_SIZE) * TICK_SIZE, 2)


def _levels(current, atr, support, resistance, p5, direction):
    # Cap the risk unit. abs(p5) is a model output with no upper bound, so a
    # single wild forecast could previously widen the stop without limit.
    atr = max(float(atr), 0.5)
    risk = max(atr * 0.9, min(abs(p5) * 0.75, atr * MAX_RISK_ATR_MULT), 1.0)
    risk = min(risk, atr * MAX_RISK_ATR_MULT)
    if direction == "UP":
        # The stop should be near the entry or just below the nearest support;
        # taking the lower of both can create an unusably distant stop.
        sl = max(current - risk, support - atr * 0.25)
        sl = min(sl, current - max(atr * 0.25, 0.5))
        t1 = max(current + abs(p5), current + risk * 1.5)
        t2 = max(current + risk * 2.4, resistance if resistance > current else current + risk * 2.4)
    else:
        sl = min(current + risk, resistance + atr * 0.25)
        sl = max(sl, current + max(atr * 0.25, 0.5))
        t1 = min(current - abs(p5), current - risk * 1.5)
        t2 = min(current - risk * 2.4, support if support < current else current - risk * 2.4)
    entry, sl, t1, t2 = _snap(current), _snap(sl), _snap(t1), _snap(t2)
    rr = abs(t1 - entry) / max(abs(entry - sl), 1e-9)
    return {
        "entry": entry,
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "risk_reward": round(float(rr), 2),
        "risk_points": round(abs(entry - sl), 2),
        "reward_points": round(abs(t1 - entry), 2),
    }


def predict_live(token):
    required = (MODEL_5M, MODEL_10M, MODEL_DIR_MODEL, FEATURE_FILE)
    for p in required:
        if not p.exists():
            raise RuntimeError(f"V5 model missing: {p.name}. Train model first.")

    df = get_live_frame(token)
    if len(df) < 60:
        raise RuntimeError(f"Need at least 60 candles. Found {len(df)}.")

    x, labels = _build_training_compatible_features(df)
    cols = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"Live/training feature mismatch: {missing[:10]}")
    row = clean_X(x, cols).iloc[[-1]]
    m5 = joblib.load(MODEL_5M)
    m10 = joblib.load(MODEL_10M)
    clf = joblib.load(MODEL_DIR_MODEL)
    for model_name, model in (("5m", m5), ("10m", m10), ("direction", clf)):
        expected = int(getattr(model, "n_features_in_", len(cols)))
        if expected != len(cols):
            raise RuntimeError(f"{model_name} model expects {expected} features but saved schema has {len(cols)}. Retrain V5.1.")

    current = float(df["close"].iloc[-1])
    p5_raw = float(m5.predict(row)[0])
    p10 = float(m10.predict(row)[0])
    probs = clf.predict_proba(row)[0]
    classes = list(getattr(clf, "classes_", []))
    prob_map = {int(k): float(v) for k, v in zip(classes, probs)}
    # Class 2 is UP, 1 is FLAT, 0 is DOWN. No cross-class fallback: reading the
    # FLAT probability as "up" when class 2 is absent would report a sideways
    # model as bullish.
    up = prob_map.get(2, 0.0)
    down = prob_map.get(0, 0.0)
    flat = prob_map.get(1, 0.0)
    ml_direction = "UP" if up >= max(down, flat) else "DOWN" if down >= flat else "FLAT"
    base_conf = max(up, down, flat)
    p5 = p5_raw

    # Two different things, kept apart. candle_pattern is the single-bar
    # candlestick label that pattern_stats and the pattern model are keyed on;
    # structure_pattern is the multi-bar chart formation. The old code assigned
    # the structure over the top of `pattern` after the stats lookup had already
    # happened, so the screen showed one pattern's name next to another
    # pattern's historical hit rate.
    candle_pattern = pattern_name_from_row(x.iloc[-1])
    pattern_model = joblib.load(PATTERN_MODEL) if PATTERN_MODEL.exists() else None
    pattern_stats = json.loads(PATTERN_STATS_FILE.read_text(encoding="utf-8")) if PATTERN_STATS_FILE.exists() else {}
    pattern_bias = 0.0
    pattern_conf = None
    pattern_dir = "NEUTRAL"
    pattern_count = 0
    pattern_hit = None
    if pattern_model is not None and candle_pattern != "NONE":
        pp = pattern_model.predict_proba(row)[0]
        pc = list(getattr(pattern_model, "classes_", range(len(pp))))
        pm = {int(k): float(v) for k, v in zip(pc, pp)}
        # Named pdn, not pd — `pd` is the pandas alias in this module and binding
        # it to a float here shadowed it for the rest of this function.
        pu, pn, pdn = pm.get(2, 0.0), pm.get(1, 0.0), pm.get(0, 0.0)
        pattern_conf = max(pu, pn, pdn)
        pattern_dir = "UP" if pu >= max(pn, pdn) else "DOWN" if pdn >= pn else "NEUTRAL"
        pattern_bias = pu - pdn
        st = pattern_stats.get(candle_pattern, {})
        pattern_count = int(st.get("count", 0))
        pattern_hit = st.get("hit_rate")

    quote_data = _fetch_full_quote(token)
    news = _fetch_news(token)
    tech = _technical_context(df)
    market_regime = _market_regime(df)
    pattern_context = detect_chart_structures(df, lookback=80)
    structure_pattern = pattern_context.get("pattern") or "NONE"
    bp, sp, pressure_basis = _pressure(df)
    depth_imb = float(quote_data.get("depth_imbalance", 0.0) or 0.0)
    min_move = max(1.0, tech["atr"] * 0.15)

    if pattern_context.get("support") is not None:
        tech["support"] = float(pattern_context["support"])
    if pattern_context.get("resistance") is not None:
        tech["resistance"] = float(pattern_context["resistance"])

    trend_bias = 1 if tech["trend"] in ("UP", "STRONG UP") else -1 if tech["trend"] in ("DOWN", "STRONG DOWN") else 0
    rsi_bias = 1 if tech["rsi"] > 58 else -1 if tech["rsi"] < 42 else 0
    # tech["vwap"] is None when the instrument reports no volume (the NIFTY 50
    # index never does). Treat that as no information rather than comparing the
    # price against itself and calling the result a signal.
    if tech["vwap"] is None:
        vwap_bias = 0
    else:
        vwap_bias = 1 if current > tech["vwap"] else -1 if current < tech["vwap"] else 0

    # Levels for break detection are taken from bars *before* the current one.
    # tech["support"] is the minimum low of a window that includes the current
    # bar, so current < support is arithmetically impossible and the "broke
    # support" branch below could never fire — it was dead in 100/100 samples.
    prior = df.iloc[:-1]
    prior_support = float(pd.to_numeric(prior["low"], errors="coerce").tail(30).min()) if len(prior) >= 5 else tech["support"]
    prior_resistance = float(pd.to_numeric(prior["high"], errors="coerce").tail(30).max()) if len(prior) >= 5 else tech["resistance"]
    atr_unit = max(tech["atr"], 1.0)
    support_distance = (current - prior_support) / atr_unit
    resistance_distance = (prior_resistance - current) / atr_unit
    support_bias = 1 if 0 <= support_distance <= 1.0 and p5 > 0 else -1 if support_distance < -0.15 else 0
    resistance_bias = -1 if 0 <= resistance_distance <= 1.0 and p5 < 0 else 1 if resistance_distance < -0.15 else 0
    depth_bias = float(np.clip(depth_imb * 2, -1, 1))
    news_sentiment = float(np.clip(news.get("sentiment", 0.0), -1, 1))
    news_direction = 1 if news_sentiment > 0.15 else -1 if news_sentiment < -0.15 else 0
    news_strength = min(abs(news_sentiment), 1.0)
    news_bias = news_direction * news_strength * (0.78 if abs(p5) >= max(min_move, tech["atr"] * 0.1) else 0.52)
    p5_bias = 1 if p5 > 0 else -1 if p5 < 0 else 0
    p10_bias = 1 if p10 > 0 else -1 if p10 < 0 else 0
    regime_bias = 1 if market_regime["regime"] in ("TRENDING_UP", "TRANSITION") else -1 if market_regime["regime"] in ("TRENDING_DOWN",) else 0
    horizon_consensus = 0.65 * p5_bias + 0.35 * p10_bias

    # The model must be able to outvote the confirmation layer. In the previous
    # blend the model carried 0.36 of the weight against 0.94 of heuristics, so a
    # maximally confident bullish model (up=1.0, down=0.0) with every indicator
    # bearish produced confluence -0.82 and the screen printed DOWN. That is not
    # an ML prediction with a confirmation layer, it is an indicator vote.
    #
    # Now: the model owns MODEL_WEIGHT, the context terms share the remainder as
    # a weighted average bounded to [-1, 1], so context can shade confidence but
    # never flip a confident call.
    context_terms = (
        (0.20, pattern_bias),
        (0.16, trend_bias),
        (0.10, rsi_bias),
        (0.10, vwap_bias),
        (0.10, support_bias),
        (0.10, resistance_bias),
        (0.10, depth_bias),
        (0.08, news_bias),
        (0.06, regime_bias),
    )
    context_score = float(np.clip(sum(w * float(v) for w, v in context_terms), -1.0, 1.0))
    model_score = float(np.clip(0.78 * (up - down) + 0.22 * horizon_consensus, -1.0, 1.0))
    confluence = MODEL_WEIGHT * model_score + (1.0 - MODEL_WEIGHT) * context_score
    final_prob = float(np.clip(0.5 + 0.5 * confluence, 0.10, 0.90))
    p5, final_prob, calibration = calibrate_live_forecast(p5, final_prob)
    metadata_path = MODEL_DIR / "model_metadata.json"
    model_metrics = {}
    all_metrics = {}
    verdict = {}
    trained_model_version = "unknown"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            all_metrics = metadata.get("metrics", {}) or {}
            model_metrics = all_metrics.get("5m", {})
            verdict = metadata.get("verdict", {}) or {}
            trained_model_version = str(metadata.get("model_version", trained_model_version))
        except (OSError, ValueError, TypeError):
            model_metrics = {}
    validation_direction = float(model_metrics.get("directional_accuracy", 0.0) or 0.0)
    validation_baseline = model_metrics.get("directional_baseline")
    dir_metrics = all_metrics.get("direction", {}) or {}
    edge_sigmas = float(dir_metrics.get("EdgeSigmas", 0.0) or 0.0)
    # Two independent conditions, both required. The absolute-accuracy floor is
    # the original gate; the edge test is what actually matters, because 0.54
    # accuracy against a 0.55 baseline is worse than a coin flip dressed up as
    # skill. If metadata predates the edge fields, edge_sigmas is 0.0 and the
    # gate stays closed — failing safe is correct for something wired to money.
    quality_ok = validation_direction >= MIN_VALIDATION_DIRECTIONAL and edge_sigmas >= MIN_EDGE_SIGMAS
    if not quality_ok:
        # Do not convert a weakly validated model into a real trading call.
        final_prob = 0.5 + (final_prob - 0.5) * 0.25
        calibration["quality_gate"] = "blocked"
        calibration["validation_directional_accuracy"] = round(validation_direction, 4)
        calibration["edge_sigmas"] = round(edge_sigmas, 2)
    final_dir = "UP" if final_prob >= 0.62 else "DOWN" if final_prob <= 0.38 else "FLAT"

    # The probability view and the points view come from two different models and
    # can disagree. Showing "DOWN" beside "expected +6.2 points" is not a
    # prediction, it is two predictions contradicting each other. When they
    # disagree the honest output is FLAT — stand aside.
    points_dir = "UP" if p5 > 0 else "DOWN" if p5 < 0 else "FLAT"
    directions_agree = final_dir == "FLAT" or points_dir == final_dir
    if not directions_agree:
        final_dir = "FLAT"
    confidence = max(final_prob, 1 - final_prob) if final_dir != "FLAT" else 0.5

    signal = "WAIT"
    score = (final_prob * 100) if final_dir == "UP" else ((1 - final_prob) * 100) if final_dir == "DOWN" else 50
    score += (trend_bias * 8) + (depth_bias * 7) + (news_bias * 10) + (pattern_bias * 8)
    score = float(np.clip(score, 0, 100))
    market = "OPEN" if datetime.now(IST).weekday() < 5 and dt_time(9, 15) <= datetime.now(IST).time() <= dt_time(15, 30) else "CLOSED"
    if quality_ok and market == "OPEN" and abs(p5) >= min_move and confidence >= 0.80:
        if final_dir == "UP" and p5 > 0:
            signal = "BUY"
        elif final_dir == "DOWN" and p5 < 0:
            signal = "SELL"

    levels = (_levels(current, tech["atr"], tech["support"], tech["resistance"], p5, final_dir)
              if signal != "WAIT"
              else {"entry": None, "stop_loss": None, "target_1": None, "target_2": None,
                    "risk_reward": None, "risk_points": None, "reward_points": None})

    # The 10-minute view goes through the same dead band as the 5-minute view.
    # Previously it printed raw sign ("UP" if p10 >= 0), so a p10 of +0.03 points
    # was displayed as an UP call, and it could contradict next_5m on screen.
    dir_10m = "UP" if p10 > min_move else "DOWN" if p10 < -min_move else "FLAT"

    trade_call = {
        "bias": signal,
        "setup": "Bullish continuation" if signal == "BUY" else "Bearish continuation" if signal == "SELL" else "Wait for confirmation",
        "entry": levels.get("entry"),
        "stop_loss": levels.get("stop_loss"),
        "target_1": levels.get("target_1"),
        "target_2": levels.get("target_2"),
        "risk_reward": levels.get("risk_reward"),
        "confidence": round(float(confidence), 4),
        "reason": f"{tech['trend']} / RSI {tech['rsi']:.1f} / news {news_sentiment:+.2f} / depth {depth_imb:+.2f} / regime {market_regime['regime']}",
    }

    next_price = current + p5
    return {
        "model_version": trained_model_version,
        "calibration": calibration,
        "timestamp": str(df["timestamp"].iloc[-1]),
        "updated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "current_price": current,
        # The bar the forecast was actually computed from. Shown on the dashboard so
        # a stale or wrong candle is visible instead of silently driving a signal.
        "latest_candle": {
            "open": round(float(df["open"].iloc[-1]), 2),
            "high": round(float(df["high"].iloc[-1]), 2),
            "low": round(float(df["low"].iloc[-1]), 2),
            "close": round(float(df["close"].iloc[-1]), 2),
        },
        "market_status": market,
        "market_regime": market_regime,
        "model_quality": {
            "trade_signals_enabled": quality_ok,
            "validation_directional_accuracy": round(validation_direction, 4),
            "validation_directional_baseline": round(float(validation_baseline), 4) if validation_baseline is not None else None,
            "edge_sigmas": round(edge_sigmas, 2),
            "minimum_required": MIN_VALIDATION_DIRECTIONAL,
            "minimum_edge_sigmas": MIN_EDGE_SIGMAS,
            "verdict": verdict.get("headline") or "Model metadata predates edge testing — retrain to get a verdict.",
            "verdict_reasons": verdict.get("reasons", []),
            "tradeable": bool(verdict.get("tradeable", False)),
        },
        "next_5m": {
            "available": quality_ok,
            "direction": final_dir,
            "expected_points": round(p5, 2),
            "expected_price": round(next_price, 2),
            "confidence": round(float(confidence), 4),
            "model_probability": round(float(base_conf), 4),
            "confluence_probability": round(float(final_prob), 4),
            "points_direction": points_dir,
            "directions_agree": bool(directions_agree),
        },
        "next_10m": {
            "available": quality_ok,
            "direction": dir_10m,
            "expected_points": round(p10, 2),
            "expected_price": round(current + p10, 2),
        },
        "signal": signal,
        "signal_score": round(score, 1),
        "pattern": candle_pattern,
        "candle_pattern": candle_pattern,
        "structure_pattern": structure_pattern,
        "pattern_context": pattern_context,
        "pattern_learning": {
            "enabled": pattern_model is not None,
            "pattern": candle_pattern,
            "direction": pattern_dir,
            "confidence": round(pattern_conf, 4) if pattern_conf is not None else None,
            "historical_hit_rate": round(float(pattern_hit), 4) if pattern_hit is not None else None,
            "training_samples": pattern_count,
            "weight": 0.15 + 0.20 * min(1, pattern_count / 100),
        },
        "trend": tech["trend"],
        "rsi": round(tech["rsi"], 2),
        "vwap": round(tech["vwap"], 2) if tech["vwap"] is not None else None,
        "vwap_note": None if tech["has_volume"] else "NIFTY 50 is an index and publishes no volume, so VWAP is undefined.",
        "support": round(tech["support"], 2),
        "resistance": round(tech["resistance"], 2),
        "buyer_pressure": round(bp, 1),
        "seller_pressure": round(sp, 1),
        "pressure_basis": pressure_basis,
        "market_depth": {
            "available": bool(quote_data.get("depth_supported", False)),
            "buyers": round(float(quote_data.get("bid5_quantity", 0)), 0),
            "sellers": round(float(quote_data.get("ask5_quantity", 0)), 0),
            "imbalance": round(depth_imb, 4),
            "total_buy_quantity": round(float(quote_data.get("total_buy_quantity", 0)), 0),
            "total_sell_quantity": round(float(quote_data.get("total_sell_quantity", 0)), 0),
        },
        "news": {"sentiment": round(float(news.get("sentiment", 0.0)), 4), "items": news.get("items", [])[:8], "updated_at": news.get("ts")},
        "levels": levels,
        "entry": levels["entry"],
        "stop_loss": levels["stop_loss"],
        "target_1": levels["target_1"],
        "target_2": levels["target_2"],
        "trade_call": trade_call,
        "reasons": [
            f"ML base direction={ml_direction}, probability={base_conf:.2f}",
            f"Candle pattern={candle_pattern}, learned bias={pattern_bias:+.2f}",
            f"Chart structure={structure_pattern}",
            f"Trend={tech['trend']}, RSI={tech['rsi']:.1f}, "
            + (f"VWAP bias={vwap_bias:+d}" if tech["has_volume"] else "VWAP unavailable (index has no volume)"),
            f"Buyer/Seller pressure={bp:.1f}/{sp:.1f} ({pressure_basis})",
            f"Depth imbalance={depth_imb:+.2f}" if quote_data.get("depth_supported") else "Order-book depth unavailable for this instrument",
            f"News sentiment={news.get('sentiment', 0):+.2f}",
            f"Market regime={market_regime['regime']}" if market_regime else "Market regime unavailable",
            f"Model weight={MODEL_WEIGHT:.2f} (model {model_score:+.2f}, context {context_score:+.2f}) -> confluence {confluence:+.2f}",
        ] + ([] if directions_agree else [
            f"Direction forced FLAT: probability says {points_dir == 'UP' and 'DOWN' or 'UP'} while the point model says {points_dir}."
        ]),
        "note": "Trade signals are disabled when out-of-sample directional validation is below the required threshold. Historical ML features remain causal. Research/paper-trading only.",
    }


def get_chart_history(token, rng="1D"):
    from upstox_api import fetch_intraday, fetch_range
    from database import load_candles

    range_key = str(rng).upper()
    rows = []
    try:
        if range_key == "1D":
            rows = fetch_intraday(token)
        else:
            days = {"5D": 5, "1M": 30, "3M": 90}.get(range_key, 1)
            rows = fetch_range(token, days)
    except Exception:
        rows = []

    if not rows:
        db_rows = load_candles(limit=6000)
        if db_rows:
            rows = db_rows
        else:
            return {"range": range_key, "candles": [], "count": 0}

    out = []
    for r in rows:
        if len(r) < 5:
            continue
        try:
            ts = pd.to_datetime(r[0], errors="coerce")
            if pd.isna(ts):
                continue
            open_v = float(r[1])
            high_v = float(r[2])
            low_v = float(r[3])
            close_v = float(r[4])
            if not (open_v > 0 and high_v > 0 and low_v > 0 and close_v > 0):
                continue
            if not (low_v <= open_v <= high_v and low_v <= close_v <= high_v):
                continue
            out.append({
                "timestamp": ts.isoformat(),
                "open": open_v,
                "high": high_v,
                "low": low_v,
                "close": close_v,
                "volume": float(r[5] or 0) if len(r) > 5 else 0.0,
                "open_interest": float(r[6] or 0) if len(r) > 6 else 0.0,
            })
        except (TypeError, ValueError):
            pass

    if not out:
        db_rows = load_candles(limit=6000)
        if db_rows:
            return {"range": range_key, "candles": [
                {
                    "timestamp": pd.to_datetime(r[0], errors="coerce").isoformat() if pd.notna(pd.to_datetime(r[0], errors="coerce")) else str(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5] or 0) if len(r) > 5 else 0.0,
                    "open_interest": float(r[6] or 0) if len(r) > 6 else 0.0,
                }
                for r in db_rows if len(r) >= 5 and float(r[1]) > 0 and float(r[2]) > 0 and float(r[3]) > 0 and float(r[4]) > 0 and float(r[3]) <= float(r[1]) <= float(r[2]) and float(r[3]) <= float(r[4]) <= float(r[2])
            ], "count": sum(1 for r in db_rows if len(r) >= 5 and float(r[1]) > 0 and float(r[2]) > 0 and float(r[3]) > 0 and float(r[4]) > 0 and float(r[3]) <= float(r[1]) <= float(r[2]) and float(r[3]) <= float(r[4]) <= float(r[2]))}

    if out:
        # Database fallback has no API range filtering. Keep the latest portion
        # appropriate to the selected control rather than mixing years of data.
        timestamps = pd.to_datetime([row["timestamp"] for row in out], errors="coerce")
        valid_ts = timestamps[~timestamps.isna()]
        if len(valid_ts):
            if range_key == "1D":
                last_date = valid_ts.max().date()
                out = [row for row, ts in zip(out, timestamps) if pd.notna(ts) and ts.date() == last_date]
            else:
                days = {"5D": 5, "1M": 31, "3M": 93}.get(range_key, 1)
                cutoff = valid_ts.max() - pd.Timedelta(days=days)
                out = [row for row, ts in zip(out, timestamps) if pd.notna(ts) and ts >= cutoff]

    return {"range": range_key, "candles": out, "count": len(out)}
