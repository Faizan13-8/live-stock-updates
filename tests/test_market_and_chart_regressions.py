from datetime import datetime

from zoneinfo import ZoneInfo

import app
import live_prediction
import upstox_api


def test_get_market_status_reports_open_market_window():
    now = datetime(2026, 8, 31, 10, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
    status = app.get_market_status(now)
    assert status["is_open"] is True
    assert status["label"] == "OPEN"


def test_sync_prediction_feedback_uses_correct_chart_history_signature(monkeypatch):
    calls = []

    def fake_get_chart_history(token, rng="1D"):
        calls.append((token, rng))
        return {"candles": [{"close": 105.0}]}

    monkeypatch.setattr(app, "get_chart_history", fake_get_chart_history)
    monkeypatch.setattr(
        app,
        "PREDICTION_LEDGER",
        [{
            "resolved": False,
            "expected_points": 2.0,
            "current_price": 100.0,
            "direction": "UP",
            "model_version": "V5.1",
        }],
    )
    monkeypatch.setattr(app, "record_prediction", lambda **kwargs: None)

    app._sync_prediction_feedback("abc-token")

    assert calls == [("abc-token", "1D")]


def test_get_chart_history_uses_intraday_for_1d_and_historical_for_longer_ranges(monkeypatch):
    calls = []

    def fake_fetch_intraday(token):
        calls.append(("intraday", token))
        return [("2026-08-30T09:15:00+05:30", 100.0, 101.0, 99.5, 100.5, 200.0, 0.0)]

    def fake_fetch_range(token, days):
        calls.append(("range", token, days))
        return [("2026-08-01T09:15:00+05:30", 98.0, 102.0, 96.0, 101.0, 500.0, 0.0)]

    monkeypatch.setattr(upstox_api, "fetch_intraday", fake_fetch_intraday)
    monkeypatch.setattr(upstox_api, "fetch_range", fake_fetch_range)

    assert live_prediction.get_chart_history("abc-token", "1D")["count"] == 1
    assert live_prediction.get_chart_history("abc-token", "5D")["count"] == 1
    assert calls[0] == ("intraday", "abc-token")
    assert calls[1] == ("range", "abc-token", 5)


def test_get_chart_history_falls_back_to_db_when_api_has_no_rows(monkeypatch):
    import upstox_api

    monkeypatch.setattr(upstox_api, "fetch_intraday", lambda token: [])
    monkeypatch.setattr(upstox_api, "fetch_range", lambda token, days: [])

    db_rows = [
        ("2026-08-30T09:15:00+05:30", 100.0, 101.5, 99.5, 100.8, 250.0, 0.0),
        ("2026-08-30T09:20:00+05:30", 100.8, 102.0, 100.0, 101.5, 320.0, 0.0),
    ]
    monkeypatch.setattr("database.load_candles", lambda limit=6000: db_rows)

    result = live_prediction.get_chart_history("abc-token", "1D")

    assert result["count"] == 2
    assert result["candles"][0]["close"] == 100.8


def test_get_chart_history_falls_back_to_db_when_intraday_api_raises(monkeypatch):
    import upstox_api

    def fake_fetch_intraday(token):
        raise RuntimeError("Upstox token expired")

    monkeypatch.setattr(upstox_api, "fetch_intraday", fake_fetch_intraday)
    monkeypatch.setattr(upstox_api, "fetch_range", lambda token, days: [])
    monkeypatch.setattr("database.load_candles", lambda limit=6000: [
        ("2026-08-30T09:15:00+05:30", 100.0, 101.0, 99.5, 100.5, 200.0, 0.0),
        ("2026-08-30T09:20:00+05:30", 100.5, 101.5, 99.8, 101.0, 220.0, 0.0),
    ])

    result = live_prediction.get_chart_history("abc-token", "1D")

    assert result["count"] == 2
    assert result["candles"][-1]["close"] == 101.0


def test_get_chart_history_works_without_any_token(monkeypatch):
    monkeypatch.setattr("upstox_api.fetch_intraday", lambda token: [])
    monkeypatch.setattr("upstox_api.fetch_range", lambda token, days: [])
    monkeypatch.setattr("database.load_candles", lambda limit=6000: [
        ("2026-08-30T09:15:00+05:30", 100.0, 101.0, 99.5, 100.5, 200.0, 0.0),
        ("2026-08-30T09:20:00+05:30", 100.5, 101.5, 99.8, 101.0, 220.0, 0.0),
    ])

    result = live_prediction.get_chart_history(None, "1D")

    assert result["count"] == 2
    assert result["candles"][-1]["close"] == 101.0


def test_db_fallback_1d_keeps_only_latest_trading_session(monkeypatch):
    import upstox_api

    monkeypatch.setattr(upstox_api, "fetch_intraday", lambda token: [])
    monkeypatch.setattr(upstox_api, "fetch_range", lambda token, days: [])
    monkeypatch.setattr("database.load_candles", lambda limit=6000: [
        ("2026-08-27T15:25:00+05:30", 100.0, 101.0, 99.5, 100.5, 200.0, 0.0),
        ("2026-08-28T09:15:00+05:30", 101.0, 102.0, 100.5, 101.5, 220.0, 0.0),
        ("2026-08-28T09:20:00+05:30", 101.5, 102.5, 101.0, 102.0, 230.0, 0.0),
    ])

    result = live_prediction.get_chart_history(None, "1D")

    assert result["count"] == 2
    assert all(row["timestamp"].startswith("2026-08-28") for row in result["candles"])


def test_trade_levels_do_not_choose_an_unreasonably_distant_stop():
    long_levels = live_prediction._levels(100.0, 4.0, 70.0, 110.0, 3.0, "UP")
    short_levels = live_prediction._levels(100.0, 4.0, 90.0, 135.0, -3.0, "DOWN")

    assert 96.0 <= long_levels["stop_loss"] < 100.0
    assert 100.0 < short_levels["stop_loss"] <= 104.0


def test_trade_levels_land_on_the_nse_tick_grid():
    """Every level must be enterable as a limit order. NSE ticks are 0.05."""
    levels = live_prediction._levels(24812.37, 18.4, 24760.0, 24870.0, 12.6, "UP")
    for field in ("entry", "stop_loss", "target_1", "target_2"):
        value = levels[field]
        assert abs(round(value / 0.05) - value / 0.05) < 1e-9, f"{field}={value} is off the tick grid"


def test_trade_level_risk_is_capped_relative_to_atr():
    """A wild point forecast must not be able to widen the stop without limit."""
    atr = 18.4
    levels = live_prediction._levels(24812.37, atr, 24760.0, 24870.0, 500.0, "UP")
    assert levels["risk_points"] <= atr * live_prediction.MAX_RISK_ATR_MULT + 0.05


def test_confirmation_layer_cannot_invert_a_confident_model():
    """The context terms may shade confidence, never flip the model's call.

    The old blend gave the model 0.36 of the weight against 0.94 of heuristics,
    so a maximally bullish model with every indicator bearish printed DOWN.
    """
    import numpy as np

    weights = [0.20, 0.16, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08, 0.06]
    mw = live_prediction.MODEL_WEIGHT

    def blend(up, down, horizon, context_value):
        context = float(np.clip(sum(w * context_value for w in weights), -1.0, 1.0))
        model = float(np.clip(0.78 * (up - down) + 0.22 * horizon, -1.0, 1.0))
        confluence = mw * model + (1.0 - mw) * context
        return float(np.clip(0.5 + 0.5 * confluence, 0.10, 0.90))

    assert blend(1.0, 0.0, 1.0, -1) >= 0.62, "bullish model outvoted by bearish context"
    assert blend(0.0, 1.0, -1.0, 1) <= 0.38, "bearish model outvoted by bullish context"
    # But an undecided model must still let context decide.
    assert blend(0.34, 0.33, 0.0, -1) < 0.5


def _capture_feedback(monkeypatch, candles, ledger):
    recorded = []
    monkeypatch.setattr(app, "get_chart_history", lambda token, rng="1D": {"candles": candles})
    monkeypatch.setattr(app, "PREDICTION_LEDGER", ledger)
    monkeypatch.setattr(app, "record_prediction", lambda **kwargs: recorded.append(kwargs))
    app._sync_prediction_feedback("abc-token")
    return recorded


def test_feedback_scores_the_next_candle_not_the_latest_close(monkeypatch):
    """A 5-minute call must be settled by the candle right after it."""
    candles = [
        {"timestamp": "2026-08-28T10:00:00+05:30", "close": 100.0},
        {"timestamp": "2026-08-28T10:05:00+05:30", "close": 103.0},
        {"timestamp": "2026-08-28T10:30:00+05:30", "close": 150.0},
    ]
    ledger = [{
        "resolved": False,
        "timestamp": "2026-08-28 10:00:00+05:30",
        "expected_points": 2.0,
        "current_price": 100.0,
        "direction": "UP",
        "model_version": "V5.1",
    }]

    recorded = _capture_feedback(monkeypatch, candles, ledger)

    assert len(recorded) == 1
    # 103 - 100, not 150 - 100.
    assert recorded[0]["actual_points"] == 3.0
    assert recorded[0]["actual_direction"] == "UP"
    assert ledger[0]["resolved"] is True


def test_feedback_discards_forecasts_with_no_adjacent_candle(monkeypatch):
    """An overnight gap is not an observed 5-minute outcome."""
    candles = [
        {"timestamp": "2026-08-28T15:25:00+05:30", "close": 100.0},
        {"timestamp": "2026-08-31T09:15:00+05:30", "close": 140.0},
    ]
    ledger = [{
        "resolved": False,
        "timestamp": "2026-08-28 15:25:00+05:30",
        "expected_points": 2.0,
        "current_price": 100.0,
        "direction": "UP",
        "model_version": "V5.1",
    }]

    recorded = _capture_feedback(monkeypatch, candles, ledger)

    assert recorded == []
    assert ledger[0]["resolved"] is True
    assert ledger[0]["unscored_reason"] == "no adjacent candle"


def test_feedback_leaves_unsettled_forecasts_open(monkeypatch):
    """The newest forecast has no following candle yet, so it must stay pending."""
    candles = [{"timestamp": "2026-08-28T10:00:00+05:30", "close": 100.0}]
    ledger = [{
        "resolved": False,
        "timestamp": "2026-08-28 10:00:00+05:30",
        "expected_points": 2.0,
        "current_price": 100.0,
        "direction": "UP",
        "model_version": "V5.1",
    }]

    recorded = _capture_feedback(monkeypatch, candles, ledger)

    assert recorded == []
    assert ledger[0]["resolved"] is False
