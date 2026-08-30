import inspect
import json
from pathlib import Path

from model_feedback import calculate_accuracy, record_prediction, should_retrain


def test_should_retrain_uses_80_percent_target():
    params = inspect.signature(should_retrain).parameters
    assert params["min_accuracy"].default == 0.80


def test_record_prediction_and_accuracy(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback.json"
    monkeypatch.setattr("model_feedback.FEEDBACK_PATH", feedback_path)

    record_prediction(
        predicted_direction="UP",
        expected_points=12.5,
        actual_points=15.2,
        actual_direction="UP",
        model_version="V5.1",
    )
    record_prediction(
        predicted_direction="DOWN",
        expected_points=-8.0,
        actual_points=-2.0,
        actual_direction="UP",
        model_version="V5.1",
    )

    summary = calculate_accuracy(window=10)
    assert summary["samples"] >= 2
    assert summary["direction_accuracy"] >= 0.0
    assert summary["mape"] >= 0.0
    assert summary["best_model_version"] == "V5.1"

    data = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_should_retrain_after_five_prediction_batch(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback.json"
    monkeypatch.setattr("model_feedback.FEEDBACK_PATH", feedback_path)

    for i in range(5):
        actual_dir = "DOWN" if i % 2 == 0 else "UP"
        record_prediction(
            predicted_direction="UP",
            expected_points=12.5,
            actual_points=5.0,
            actual_direction=actual_dir,
            model_version="V5.2",
        )

    should_train, summary = should_retrain(window=10, min_samples=5, min_accuracy=0.80, max_mape=0.50, batch_size=5)
    assert should_train is True
    assert summary["samples"] == 5


def test_should_retrain_respects_cooldown(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback.json"
    monkeypatch.setattr("model_feedback.FEEDBACK_PATH", feedback_path)

    for i in range(5):
        record_prediction(
            predicted_direction="UP",
            expected_points=12.5,
            actual_points=5.0,
            actual_direction="DOWN",
            model_version="V5.2",
        )

    from datetime import datetime

    should_train, summary = should_retrain(
        window=10,
        min_samples=5,
        min_accuracy=0.80,
        max_mape=0.50,
        batch_size=5,
        last_retrain_at=datetime.now().isoformat(),
        cooldown_minutes=60,
    )

    assert should_train is False
    assert summary["samples"] == 5


def test_run_live_prediction_cycle_returns_safe_wait_when_data_is_unavailable(monkeypatch):
    import app

    def fail(_token):
        raise RuntimeError("Upstox token expired")

    monkeypatch.setattr(app, "predict_live", fail)

    result = app._run_live_prediction_cycle("fake-token")

    assert result is not None
    assert result["signal"] == "WAIT"
    assert result["market_status"] in {"OPEN", "CLOSED"}
    assert "Prediction unavailable" in result["note"]
