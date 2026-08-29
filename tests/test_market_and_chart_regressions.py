from datetime import datetime

from zoneinfo import ZoneInfo

import app


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
