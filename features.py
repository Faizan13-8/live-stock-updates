import numpy as np
import pandas as pd

FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_20",
    "body_pct","upper_wick_pct","lower_wick_pct","range_pct",
    "ema5_dist","ema9_dist","ema20_dist","ema50_dist",
    "sma20_dist","sma50_dist",
    "rsi14","rsi_delta",
    "macd","macd_signal","macd_hist","atr_pct",
    "vol_ratio","close_pos_20","high_breakout_20","low_breakdown_20",
    "trend_strength","regime_trend","regime_volatility","volatility_bucket",
    "time_sin","time_cos",
]

# Trailing window for the volatility percentile buckets. Must be comfortably
# smaller than the live frame (live_prediction pulls 1200 candles) so the last
# row gets the same window at inference as it did during training.
VOL_BUCKET_WINDOW = 500
VOL_BUCKET_MIN_PERIODS = 100


def make_features(df):
    x = df.copy()
    for c in ["open","high","low","close","volume","open_interest"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    c = x["close"]
    o = x["open"]
    h = x["high"]
    l = x["low"]

    r = c.pct_change()
    for n in [1, 3, 5, 10, 20]:
        x[f"ret_{n}"] = c.pct_change(n)

    rng = (h - l).replace(0, np.nan)
    x["body_pct"] = (c - o) / c.replace(0, np.nan)
    x["upper_wick_pct"] = (h - np.maximum(o, c)) / c.replace(0, np.nan)
    x["lower_wick_pct"] = (np.minimum(o, c) - l) / c.replace(0, np.nan)
    x["range_pct"] = rng / c.replace(0, np.nan)

    for n in [5, 9, 20, 50]:
        ema = c.ewm(span=n, adjust=False).mean()
        x[f"ema{n}_dist"] = c / ema.replace(0, np.nan) - 1.0
    for n in [20, 50]:
        sma = c.rolling(n, min_periods=5).mean()
        x[f"sma{n}_dist"] = c / sma.replace(0, np.nan) - 1.0

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    x["rsi_delta"] = x["rsi14"].diff()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    x["atr_pct"] = atr / c.replace(0, np.nan)

    vm = x["volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    x["vol_ratio"] = x["volume"] / vm.replace(0, np.nan)

    rh = h.shift(1).rolling(20, min_periods=5).max()
    rl = l.shift(1).rolling(20, min_periods=5).min()
    denom = (rh - rl).replace(0, np.nan)
    x["close_pos_20"] = (c - rl) / denom
    x["high_breakout_20"] = (c > rh).astype(int)
    x["low_breakdown_20"] = (c < rl).astype(int)

    ma20 = c.rolling(20, min_periods=5).mean()
    ma50 = c.rolling(50, min_periods=10).mean()
    x["trend_strength"] = (c - ma20) / atr.replace(0, np.nan)
    x["regime_trend"] = ((c > ma20) & (ma20 > ma50)).astype(int) - ((c < ma20) & (ma20 < ma50)).astype(int)
    x["regime_volatility"] = r.rolling(20, min_periods=5).std().fillna(0.0)
    # Bucket against a trailing window, never against the whole series. A global
    # .quantile() reads future rows (look-ahead) and — worse — is computed over
    # 3 years at training but over the ~1200-candle live frame at inference, so
    # the same candle lands in a different bucket on each path. A fixed trailing
    # window makes the value identical on both.
    vol_hi = x["regime_volatility"].rolling(VOL_BUCKET_WINDOW, min_periods=VOL_BUCKET_MIN_PERIODS).quantile(0.75)
    vol_lo = x["regime_volatility"].rolling(VOL_BUCKET_WINDOW, min_periods=VOL_BUCKET_MIN_PERIODS).quantile(0.25)
    x["volatility_bucket"] = np.where(
        x["regime_volatility"] > vol_hi, 1.0,
        np.where(x["regime_volatility"] < vol_lo, -1.0, 0.0),
    )

    ts = pd.to_datetime(x["timestamp"], errors="coerce")
    minute = ts.dt.hour * 60 + ts.dt.minute
    x["time_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    x["time_cos"] = np.cos(2 * np.pi * minute / 1440.0)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x, FEATURES.copy()


def clean_X(x, cols):
    y = x[cols].apply(pd.to_numeric, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    # Forward-fill only. bfill would pull a later candle's value backwards into a
    # warm-up row, which is look-ahead leakage at training time and a different
    # value than the live frame would produce at inference time.
    y = y.ffill().fillna(0.0)
    return y
