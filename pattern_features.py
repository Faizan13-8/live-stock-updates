import numpy as np
import pandas as pd

from chart_patterns import structure_row_features

PATTERN_NAMES = [
    "NONE", "DOJI", "HAMMER", "SHOOTING_STAR",
    "BULLISH_ENGULFING", "BEARISH_ENGULFING", "INSIDE_BAR",
    "BREAKOUT_UP", "BREAKOUT_DOWN", "HIGHER_HIGH_LOWER_LOW",
    "LOWER_HIGH_LOWER_LOW"
]


def add_pattern_features(df):
    """Causal price-action features with conservative pattern detection and no look-ahead leakage."""
    x = df.copy()
    o = pd.to_numeric(x["open"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")

    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    x["pattern_body_ratio"] = (body / rng).replace([np.inf, -np.inf], np.nan)
    x["pattern_upper_wick_ratio"] = (upper / rng).replace([np.inf, -np.inf], np.nan)
    x["pattern_lower_wick_ratio"] = (lower / rng).replace([np.inf, -np.inf], np.nan)
    x["pattern_bullish"] = (c > o).astype(int)
    x["pattern_bearish"] = (c < o).astype(int)

    prev_close = c.shift(1)
    prev_open = o.shift(1)
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    prev_body = (prev_close - prev_open).abs()

    doji = (body / rng <= 0.12)
    hammer = (
        (lower >= 2.0 * body) &
        (upper <= 0.8 * body) &
        (body / rng <= 0.45) &
        (c > o)
    )
    shooting_star = (
        (upper >= 2.0 * body) &
        (lower <= 0.8 * body) &
        (body / rng <= 0.45) &
        (c < o)
    )
    bullish_engulfing = (
        (prev_close < prev_open) &
        (c > o) &
        (o <= prev_close) &
        (c >= prev_open) &
        (body >= prev_body)
    )
    bearish_engulfing = (
        (prev_close > prev_open) &
        (c < o) &
        (o >= prev_close) &
        (c <= prev_open) &
        (body >= prev_body)
    )
    inside_bar = (h <= prev_high) & (l >= prev_low)

    prev_high_20 = h.shift(1).rolling(20, min_periods=5).max()
    prev_low_20 = l.shift(1).rolling(20, min_periods=5).min()
    breakout_up = c > prev_high_20
    breakout_down = c < prev_low_20
    hh_hl = (h > prev_high) & (l > prev_low)
    lh_ll = (h < prev_high) & (l < prev_low)

    for name, series in {
        "pattern_doji": doji,
        "pattern_hammer": hammer,
        "pattern_shooting_star": shooting_star,
        "pattern_bullish_engulfing": bullish_engulfing,
        "pattern_bearish_engulfing": bearish_engulfing,
        "pattern_inside_bar": inside_bar,
        "pattern_breakout_up": breakout_up,
        "pattern_breakout_down": breakout_down,
        "pattern_higher_high_lower_high": hh_hl,
        "pattern_lower_high_lower_low": lh_ll,
    }.items():
        x[name] = series.fillna(False).astype(int)

    # Most specific pattern first. A bar often matches several masks at once, and
    # Series.mask overwrites on every hit, so this list is applied in reverse: the
    # first entry is written last and therefore wins. Applying it forwards lets the
    # generic higher-high/lower-low masks bury every breakout and engulfing bar.
    pattern_priority = [
        (breakout_up, "BREAKOUT_UP"),
        (breakout_down, "BREAKOUT_DOWN"),
        (bullish_engulfing, "BULLISH_ENGULFING"),
        (bearish_engulfing, "BEARISH_ENGULFING"),
        (hammer, "HAMMER"),
        (shooting_star, "SHOOTING_STAR"),
        (inside_bar, "INSIDE_BAR"),
        (doji, "DOJI"),
        (hh_hl, "HIGHER_HIGH_LOWER_LOW"),
        (lh_ll, "LOWER_HIGH_LOWER_LOW"),
    ]
    label = pd.Series("NONE", index=x.index, dtype="object")
    for mask, name in reversed(pattern_priority):
        label = label.mask(mask.fillna(False), name)

    x["pattern_code"] = label.map({name: i for i, name in enumerate(PATTERN_NAMES)}).astype(int)

    c5 = c.rolling(5).mean()
    c20 = c.rolling(20).mean()
    x["structure_up"] = ((c5 > c20) & (c5.diff() > 0)).astype(int)
    x["structure_down"] = ((c5 < c20) & (c5.diff() < 0)).astype(int)
    x["structure_strength"] = ((c5 - c20) / c20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return x, label


def pattern_name_from_row(row):
    code = row.get("pattern_code", 0)
    try:
        code = int(code)
    except Exception:
        code = 0
    return PATTERN_NAMES[code] if 0 <= code < len(PATTERN_NAMES) else "NONE"
