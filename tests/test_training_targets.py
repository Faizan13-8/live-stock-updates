"""Targets must never bridge a session break, and features must stay causal."""
import numpy as np
import pandas as pd

import train_model as tm
from features import make_features


def _two_day_frame():
    """Two sessions of 5-minute bars with an overnight gap between them."""
    rows = []
    price = 100.0
    for day in ("2026-08-27", "2026-08-28"):
        for i in range(6):
            stamp = pd.Timestamp(f"{day}T09:15:00+05:30") + pd.Timedelta(minutes=5 * i)
            price += 1.0
            rows.append({"timestamp": stamp.isoformat(), "open": price, "high": price + 1,
                         "low": price - 1, "close": price, "volume": 0, "open_interest": 0})
        price += 500.0  # overnight gap
    return pd.DataFrame(rows)


def test_forward_move_does_not_bridge_the_overnight_gap():
    df = _two_day_frame()
    close = df["close"].astype(float)
    stamps = pd.to_datetime(df["timestamp"], utc=True)

    move = tm._forward_move(close, stamps, 1)

    # The last bar of day one has no valid "next 5 minutes".
    assert pd.isna(move.iloc[5]), "overnight gap was treated as a 5-minute move"
    assert move.iloc[0] == 1.0
    assert not move.iloc[:5].isna().any()


def test_unresolvable_target_rows_are_dropped_not_zero_filled():
    """fillna(0.0) previously relabelled gap rows as 'price did not move'."""
    df = _two_day_frame()
    close = df["close"].astype(float)
    stamps = pd.to_datetime(df["timestamp"], utc=True)
    move = tm._forward_move(close, stamps, 1)
    assert (move == 0.0).sum() == 0, "a gap row was labelled FLAT instead of dropped"


def test_direction_band_is_never_zero():
    """A zero threshold forces every non-zero move into UP or DOWN."""
    band = np.maximum(tm.DIRECTION_ATR_MULT * np.array([0.0, 0.0, 50.0]),
                      tm.MIN_DIRECTION_BAND_PTS)
    assert (band > 0).all()
    assert band[0] == tm.MIN_DIRECTION_BAND_PTS


def test_volatility_bucket_uses_only_past_rows():
    """The bucket must not change when future candles are appended.

    A global .quantile() made this value depend on the whole series, so the same
    candle was bucketed differently at training time (3 years of data) and at
    inference time (a 1200-candle live frame).
    """
    rng = np.random.default_rng(0)
    n = 900
    close = 24000 + np.cumsum(rng.normal(0, 5, n))
    stamps = pd.date_range("2026-01-01T09:15:00+05:30", periods=n, freq="5min")
    frame = pd.DataFrame({"timestamp": stamps.astype(str), "open": close, "high": close + 3,
                          "low": close - 3, "close": close, "volume": 0, "open_interest": 0})

    full, _ = make_features(frame)
    truncated, _ = make_features(frame.iloc[:700].copy())

    cut = 699
    assert full["volatility_bucket"].iloc[cut] == truncated["volatility_bucket"].iloc[cut]
