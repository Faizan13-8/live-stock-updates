import numpy as np
import pandas as pd

from live_prediction import detect_support_resistance, build_auto_pattern_context
from pattern_features import add_pattern_features


def test_detect_support_resistance_finds_wellformed_levels():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min"),
            "open": [100.0] * 20,
            "high": [101.0, 102.0, 102.5, 101.5, 103.0, 103.5, 104.0, 104.5, 104.0, 105.0,
                      105.2, 105.8, 106.0, 105.5, 106.5, 107.0, 107.3, 106.8, 108.0, 108.5],
            "low": [99.1, 99.5, 99.8, 99.0, 100.2, 100.4, 101.2, 101.0, 101.8, 102.4,
                    102.0, 102.8, 103.0, 102.5, 103.2, 104.0, 104.1, 103.8, 104.6, 105.2],
            "close": [100.4, 101.2, 100.8, 101.9, 102.4, 103.0, 103.9, 102.8, 103.6, 104.2,
                      104.5, 105.0, 105.3, 104.8, 106.0, 106.4, 106.9, 106.3, 107.5, 108.1],
            "volume": [1000] * 20,
            "open_interest": [10] * 20,
        }
    )
    support, resistance, meta = detect_support_resistance(df)
    assert support < resistance
    assert support == meta["support"]
    assert resistance == meta["resistance"]


def test_build_auto_pattern_context_returns_trendline_and_support_resistance():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=18, freq="5min"),
            "open": [100.0, 100.6, 101.1, 101.4, 101.9, 102.2, 102.5, 101.8, 102.6, 103.1,
                     103.8, 104.0, 104.4, 104.7, 105.0, 105.6, 106.0, 106.5],
            "high": [101.0, 101.3, 101.9, 102.5, 102.7, 103.0, 103.3, 102.8, 103.4, 104.0,
                     104.3, 104.8, 105.1, 105.6, 105.8, 106.4, 106.9, 107.1],
            "low": [99.0, 99.8, 100.4, 100.9, 101.2, 101.7, 101.9, 101.3, 102.0, 102.8,
                    103.0, 103.2, 103.7, 104.1, 104.4, 104.8, 105.2, 105.9],
            "close": [100.4, 101.0, 101.5, 102.0, 102.4, 102.8, 103.0, 102.5, 103.0, 103.6,
                      104.1, 104.5, 104.9, 105.2, 105.5, 106.1, 106.4, 106.8],
            "volume": [800] * 18,
            "open_interest": [10] * 18,
        }
    )
    ctx = build_auto_pattern_context(df, "BREAKOUT_UP")
    assert "support" in ctx
    assert "resistance" in ctx
    assert "trendline" in ctx
    assert len(ctx["trendline"]) >= 2


def test_specific_pattern_wins_over_generic_higher_high_label():
    """A breakout bar is also a higher-high/higher-low bar; the breakout must win."""
    n = 30
    highs = list(np.linspace(100.0, 105.0, n - 1))
    lows = [v - 1.0 for v in highs]
    opens = [v - 0.5 for v in highs]
    closes = [v - 0.2 for v in highs]
    # Final bar clears the 20-bar high while also printing a higher high and low.
    highs.append(120.0)
    lows.append(112.0)
    opens.append(113.0)
    closes.append(119.0)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * n,
        "open_interest": [0] * n,
    })

    x, label = add_pattern_features(df)

    assert int(x["pattern_breakout_up"].iloc[-1]) == 1
    assert int(x["pattern_higher_high_lower_high"].iloc[-1]) == 1
    assert label.iloc[-1] == "BREAKOUT_UP"
