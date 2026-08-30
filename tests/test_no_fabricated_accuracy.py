"""The dashboard must never be handed a number the system did not measure."""
import json

import model_feedback


def test_accuracy_reports_unknown_not_eighty_percent(tmp_path, monkeypatch):
    """With no resolved outcomes, accuracy is None — it used to be a literal 0.8."""
    monkeypatch.setattr(model_feedback, "FEEDBACK_PATH", tmp_path / "fb.json")
    monkeypatch.setattr(model_feedback, "MODEL_DIR", tmp_path)

    summary = model_feedback.calculate_accuracy()
    assert summary["samples"] == 0
    assert summary["has_data"] is False
    assert summary["direction_accuracy"] is None


def test_placeholder_records_do_not_shrink_live_forecasts(tmp_path, monkeypatch):
    """Unresolved rows must be invisible to the calibrator.

    141 rows of expected=+84.76 / actual=0.0 previously survived into
    calibrate_live_forecast and pinned point_scale to the 0.35 floor, silently
    cutting every forecast to a third.
    """
    path = tmp_path / "fb.json"
    path.write_text(json.dumps([
        {"timestamp": "2026-08-29T10:00:00+05:30", "predicted_direction": "FLAT",
         "expected_points": 84.76, "actual_points": 0.0, "actual_direction": "FLAT",
         "model_version": "V5.1"}
        for _ in range(141)
    ]), encoding="utf-8")
    monkeypatch.setattr(model_feedback, "FEEDBACK_PATH", path)
    monkeypatch.setattr(model_feedback, "MODEL_DIR", tmp_path)

    assert model_feedback.load_resolved_feedback() == []

    points, prob, info = model_feedback.calibrate_live_forecast(13.0, 0.72)
    assert points == 13.0, "forecast was shrunk by unresolved placeholder records"
    assert prob == 0.72
    assert info["enabled"] is False
    assert info["point_scale"] == 1.0


def test_should_retrain_does_not_fire_on_unknown_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(model_feedback, "FEEDBACK_PATH", tmp_path / "fb.json")
    monkeypatch.setattr(model_feedback, "MODEL_DIR", tmp_path)
    fire, summary = model_feedback.should_retrain()
    assert fire is False
    assert summary["has_data"] is False


def test_mape_ignores_rows_with_a_zero_actual(tmp_path, monkeypatch):
    """MAPE is a ratio. Appending raw point errors mixed units into the average."""
    path = tmp_path / "fb.json"
    path.write_text(json.dumps([
        {"timestamp": "2026-08-29T10:00:00+05:30", "predicted_direction": "UP",
         "expected_points": 10.0, "actual_points": 20.0, "actual_direction": "UP",
         "model_version": "V5.1"},
        {"timestamp": "2026-08-29T10:05:00+05:30", "predicted_direction": "FLAT",
         "expected_points": 0.0, "actual_points": 0.0, "actual_direction": "FLAT",
         "model_version": "V5.1"},
    ]), encoding="utf-8")
    monkeypatch.setattr(model_feedback, "FEEDBACK_PATH", path)
    monkeypatch.setattr(model_feedback, "MODEL_DIR", tmp_path)

    summary = model_feedback.calculate_accuracy()
    assert summary["samples"] == 2
    assert summary["mape_samples"] == 1
    assert summary["mape"] == 0.5
