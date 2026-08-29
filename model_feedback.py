import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEEDBACK_PATH = MODEL_DIR / "model_feedback.json"


def _ensure_dir():
    MODEL_DIR.mkdir(exist_ok=True)


def _load_feedback():
    _ensure_dir()
    if not FEEDBACK_PATH.exists():
        return []
    try:
        data = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_feedback(items):
    _ensure_dir()
    FEEDBACK_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def record_prediction(predicted_direction, expected_points, actual_points, actual_direction, model_version="unknown"):
    item = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted_direction": str(predicted_direction).upper(),
        "expected_points": float(expected_points),
        "actual_points": float(actual_points),
        "actual_direction": str(actual_direction).upper(),
        "model_version": str(model_version),
    }
    items = _load_feedback()
    items.append(item)
    _save_feedback(items[-5000:])
    return item


def calculate_accuracy(window=100):
    items = _load_feedback()
    if not items:
        return {
            "samples": 0,
            "direction_accuracy": 0.8,
            "mape": 0.0,
            "mae": 0.0,
            "best_model_version": "N/A",
            "window": window,
            "note": "No live feedback yet; using the configured confidence target until enough samples exist.",
        }

    recent = items[-window:]
    direction_matches = 0
    abs_errors = []
    model_counts = {}
    for row in recent:
        model_counts[row.get("model_version", "unknown")] = model_counts.get(row.get("model_version", "unknown"), 0) + 1
        if str(row.get("predicted_direction", "")).upper() == str(row.get("actual_direction", "")).upper():
            direction_matches += 1

        actual = float(row.get("actual_points", 0) or 0)
        expected = float(row.get("expected_points", 0) or 0)
        if actual == 0:
            abs_errors.append(abs(expected))
        else:
            abs_errors.append(abs(expected - actual) / abs(actual))

    direction_accuracy = direction_matches / len(recent) if recent else 0.0
    mape = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    mae = sum(abs(float(row.get("expected_points", 0) or 0) - float(row.get("actual_points", 0) or 0)) for row in recent) / len(recent) if recent else 0.0
    best_model_version = max(model_counts.items(), key=lambda kv: kv[1])[0] if model_counts else "N/A"

    return {
        "samples": len(recent),
        "direction_accuracy": round(float(direction_accuracy), 4),
        "mape": round(float(mape), 4),
        "mae": round(float(mae), 4),
        "best_model_version": best_model_version,
        "window": window,
    }


def should_retrain(window=100, min_samples=5, min_accuracy=0.80, max_mape=0.50, batch_size=5):
    summary = calculate_accuracy(window=window)
    recent_count = summary["samples"]
    required_samples = max(min_samples, batch_size)
    if recent_count < required_samples:
        return False, summary
    if recent_count >= batch_size and (summary["direction_accuracy"] < min_accuracy or summary["mape"] > max_mape):
        return True, summary
    return False, summary


def upsert_prediction_feedback(prediction_record):
    items = _load_feedback()
    for i, item in enumerate(items):
        if item.get("timestamp") == prediction_record.get("timestamp") and item.get("model_version") == prediction_record.get("model_version"):
            items[i] = prediction_record
            _save_feedback(items[-5000:])
            return prediction_record
    items.append(prediction_record)
    _save_feedback(items[-5000:])
    return prediction_record
