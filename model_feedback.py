import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEEDBACK_PATH = MODEL_DIR / "model_feedback.json"

# India has no daylight saving, so a fixed offset is exactly right and avoids
# depending on a tz database being present. The rest of the app stamps IST; this
# module used to stamp naive local time, so records could not be compared.
IST = timezone(timedelta(hours=5, minutes=30))

# Lower bound on the shrink factor applied to point forecasts.
MIN_POINT_SCALE = 0.35
MAX_POINT_SCALE = 1.25


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


def _is_resolved(row):
    """True when this record represents a genuinely observed outcome.

    A row claiming a non-zero forecast but a dead-flat actual was never resolved
    against a real future candle — it is a placeholder the monitor loop wrote and
    never filled in. Both the accuracy report and the calibrator must apply this
    same test; previously only the report did, so the calibrator fitted its scale
    factor on rows whose actual_points were all 0.0 and shrank every live
    forecast to the floor.
    """
    try:
        expected = abs(float(row.get("expected_points", 0) or 0))
        actual = abs(float(row.get("actual_points", 0) or 0))
    except (TypeError, ValueError):
        return False
    if expected > 0.01 and actual < 1e-9:
        return False
    return str(row.get("actual_direction", "")).upper() in {"UP", "DOWN", "FLAT"}


def load_resolved_feedback(window=None):
    items = [row for row in _load_feedback() if _is_resolved(row)]
    return items[-window:] if window else items


def record_prediction(predicted_direction, expected_points, actual_points, actual_direction, model_version="unknown"):
    item = {
        "timestamp": datetime.now(IST).isoformat(timespec="seconds"),
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
    items = load_resolved_feedback()
    if not items:
        # No fabricated stand-in. This function previously returned
        # direction_accuracy = 0.8 here, which the dashboard rendered as a
        # measured 80% hit rate on zero observations.
        return {
            "samples": 0,
            "has_data": False,
            "direction_accuracy": None,
            "mape": None,
            "mae": None,
            "best_model_version": "N/A",
            "window": window,
            "note": "No resolved live outcomes yet. Accuracy is unknown, not high.",
        }

    recent = items[-window:]
    direction_matches = 0
    ratio_errors = []
    point_errors = []
    model_counts = {}
    for row in recent:
        version = row.get("model_version", "unknown")
        model_counts[version] = model_counts.get(version, 0) + 1
        if str(row.get("predicted_direction", "")).upper() == str(row.get("actual_direction", "")).upper():
            direction_matches += 1

        actual = float(row.get("actual_points", 0) or 0)
        expected = float(row.get("expected_points", 0) or 0)
        point_errors.append(abs(expected - actual))
        # Only rows with a non-zero actual can contribute to a percentage error.
        # Appending abs(expected) when actual == 0 mixed points into a list of
        # ratios and reported the average as though it were a percentage.
        if abs(actual) > 1e-9:
            ratio_errors.append(abs(expected - actual) / abs(actual))

    direction_accuracy = direction_matches / len(recent)
    best_model_version = max(model_counts.items(), key=lambda kv: kv[1])[0] if model_counts else "N/A"

    return {
        "samples": len(recent),
        "has_data": True,
        "direction_accuracy": round(float(direction_accuracy), 4),
        "mape": round(sum(ratio_errors) / len(ratio_errors), 4) if ratio_errors else None,
        "mape_samples": len(ratio_errors),
        "mae": round(sum(point_errors) / len(point_errors), 4),
        "best_model_version": best_model_version,
        "window": window,
    }


def calibrate_live_forecast(expected_points, probability, window=100, min_samples=12):
    """Shrink live forecasts using only completed, chronological feedback."""
    items = load_resolved_feedback(window)
    expected = float(expected_points)
    probability = float(probability)
    if len(items) < min_samples:
        return expected, probability, {"enabled": False, "samples": len(items), "point_scale": 1.0}

    predicted = [float(x.get("expected_points", 0.0) or 0.0) for x in items]
    actual = [float(x.get("actual_points", 0.0) or 0.0) for x in items]
    denom = sum(p * p for p in predicted)
    scale = sum(p * a for p, a in zip(predicted, actual)) / denom if denom > 1e-9 else 1.0
    scale = max(MIN_POINT_SCALE, min(MAX_POINT_SCALE, scale))
    direction_accuracy = sum((p > 0) == (a > 0) for p, a in zip(predicted, actual) if p != 0 and a != 0)
    directional_samples = sum(p != 0 and a != 0 for p, a in zip(predicted, actual))
    observed = direction_accuracy / directional_samples if directional_samples else 0.5
    calibrated_probability = 0.5 + (probability - 0.5) * max(0.25, min(1.0, observed / 0.70))
    return expected * scale, max(0.02, min(0.98, calibrated_probability)), {
        "enabled": True, "samples": len(items), "point_scale": round(scale, 4),
        "direction_accuracy": round(observed, 4),
    }


def should_retrain(window=100, min_samples=5, min_accuracy=0.80, max_mape=0.50, batch_size=5, last_retrain_at=None, cooldown_minutes=60):
    summary = calculate_accuracy(window=window)
    recent_count = summary["samples"]
    required_samples = max(min_samples, batch_size)
    # Never retrain off an unknown. With has_data False the accuracy field is
    # None, and comparing None to min_accuracy would raise on Python 3.
    if not summary.get("has_data") or recent_count < required_samples:
        return False, summary

    if last_retrain_at:
        try:
            last_time = datetime.fromisoformat(str(last_retrain_at))
            now = datetime.now(IST) if last_time.tzinfo else datetime.now()
            if now - last_time < timedelta(minutes=cooldown_minutes):
                return False, summary
        except Exception:
            pass

    mape = summary.get("mape")
    degraded = summary["direction_accuracy"] < min_accuracy or (mape is not None and mape > max_mape)
    if recent_count >= batch_size and degraded:
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
