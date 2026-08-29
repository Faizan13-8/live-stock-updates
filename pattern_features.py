import numpy as np
import pandas as pd

PATTERN_NAMES = [
    "NONE", "DOJI", "HAMMER", "SHOOTING_STAR",
    "BULLISH_ENGULFING", "BEARISH_ENGULFING", "INSIDE_BAR",
    "BREAKOUT_UP", "BREAKOUT_DOWN", "HIGHER_HIGH_LOWER_LOW",
    "LOWER_HIGH_LOWER_LOW"
]


def add_pattern_features(df):
    """Causal price-action features. Uses current and past candles only."""
    x = df.copy()
    o = pd.to_numeric(x["open"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")

    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    # Basic candle anatomy.
    x["pattern_body_ratio"] = body / rng
    x["pattern_upper_wick_ratio"] = upper / rng
    x["pattern_lower_wick_ratio"] = lower / rng
    x["pattern_bullish"] = (c > o).astype(int)
    x["pattern_bearish"] = (c < o).astype(int)
    x["pattern_range_atr"] = rng / pd.to_numeric(x.get("atr_14"), errors="coerce").replace(0, np.nan) if "atr_14" in x else rng / c

    po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    pbody = (pc - po).abs()

    # Recognizable textbook patterns. Thresholds are intentionally conservative.
    doji = (body / rng <= 0.12)
    hammer = (
        (lower >= 2.0 * body) &
        (upper <= 0.8 * body) &
        (body / rng <= 0.45)
    )
    shooting_star = (
        (upper >= 2.0 * body) &
        (lower <= 0.8 * body) &
        (body / rng <= 0.45)
    )
    bullish_engulfing = (
        (pc < po) & (c > o) &
        (o <= pc) & (c >= po) &
        (body >= pbody)
    )
    bearish_engulfing = (
        (pc > po) & (c < o) &
        (o >= pc) & (c <= po) &
        (body >= pbody)
    )
    inside_bar = (h <= ph) & (l >= pl)

    # Breakout uses prior rolling levels only, avoiding current-bar leakage.
    prev_high_20 = h.shift(1).rolling(20, min_periods=5).max()
    prev_low_20 = l.shift(1).rolling(20, min_periods=5).min()
    breakout_up = c > prev_high_20
    breakout_down = c < prev_low_20

    hh_hl = (h > ph) & (l > pl)
    lh_ll = (h < ph) & (l < pl)

    flags = {
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
    }
    for name, series in flags.items():
        x[name] = series.fillna(False).astype(int)

    # Priority order for a single human-readable label.
    label = pd.Series("NONE", index=x.index, dtype="object")
    for mask, name in [
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
    ]:
        label = label.mask(mask.fillna(False), name)

    x["pattern_code"] = label.map({name: i for i, name in enumerate(PATTERN_NAMES)}).astype(int)

    # Trend structure over recent closes.
    c5 = c.rolling(5).mean()
    c20 = c.rolling(20).mean()
    x["structure_up"] = ((c5 > c20) & (c5.diff() > 0)).astype(int)
    x["structure_down"] = ((c5 < c20) & (c5.diff() < 0)).astype(int)
    x["structure_strength"] = (c5 - c20) / c20.replace(0, np.nan)

    return x, label


def pattern_name_from_row(row):
    code = row.get("pattern_code", 0)
    try:
        code = int(code)
    except Exception:
        code = 0
    return PATTERN_NAMES[code] if 0 <= code < len(PATTERN_NAMES) else "NONE"
