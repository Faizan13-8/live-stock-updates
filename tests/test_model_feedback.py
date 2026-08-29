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
