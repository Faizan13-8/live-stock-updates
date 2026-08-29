
from flask import Flask, render_template, request, jsonify
from threading import Thread, Lock
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import json
import importlib
import sqlite3
import math
import time

from upstox_api import test_connection, download_history
from database import get_stats
from data_check import check_database
from model_feedback import calculate_accuracy, should_retrain, record_prediction
import train_model as train_module
from live_prediction import predict_live, get_chart_history
from config import read_server_token, write_server_token, clear_server_token

app = Flask(__name__)
status_lock = Lock()

TOKEN = read_server_token()
PREDICTION_LEDGER = []

status = {
    "running": False,
    "message": "Ready",
    "progress": 0,
    "saved": 0,
    "chunks_done": 0,
    "chunks_total": 0,
    "error": None,
    "metadata": None,
}

LIVE_MONITOR_THREAD = None
LIVE_MONITOR_LOCK = Lock()


def market_is_open():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.weekday() >= 5:
        return False
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    return market_open <= now.time() <= market_close


def set_status(**kwargs):
    with status_lock:
        status.update(kwargs)

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/test")
def api_test():
    global TOKEN
    token = ((request.json or {}).get("token") or "").strip()
    if not token:
        return jsonify({"ok": False, "message": "Token is required"}), 400
    try:
        test_connection(token)
        TOKEN = write_server_token(token)
        return jsonify({"ok": True, "message": "Upstox API connected. Token saved on the backend server."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400

@app.post("/api/save-token")
def api_save_token():
    global TOKEN
    token = ((request.json or {}).get("token") or "").strip()
    if not token:
        return jsonify({"ok": False, "message": "Token is required"}), 400
    TOKEN = write_server_token(token)
    return jsonify({"ok": True, "message": "Token saved on the backend server."})

@app.post("/api/clear-token")
def api_clear_token():
    global TOKEN
    TOKEN = None
    clear_server_token()
    return jsonify({"ok": True, "message": "Token cleared."})

@app.get("/api/token-status")
def api_token_status():
    TOKEN = read_server_token()
    return jsonify({"saved": bool(TOKEN)})

def _token_from_request():
    global TOKEN
    supplied = ((request.json or {}).get("token") or "").strip() if request.is_json else ""
    if supplied:
        TOKEN = write_server_token(supplied)
    return TOKEN or read_server_token()

@app.post("/api/download")
def api_download():
    token = _token_from_request()
    if not token:
        return jsonify({"ok": False, "message": "Enter/test the Upstox token first."}), 400

    with status_lock:
        if status["running"]:
            return jsonify({"ok": False, "message": "Another operation is running"}), 409
        status.update({
            "running": True, "message": "Starting download...", "progress": 0,
            "saved": 0, "chunks_done": 0, "chunks_total": 0, "error": None
        })

    def worker():
        try:
            def progress_callback(**info):
                set_status(**info)
            download_history(token, progress_callback)
            stats = get_stats()
            set_status(
                running=False,
                message=f"Completed. {stats['count']:,} candles in database.",
                progress=100,
                saved=stats["count"],
            )
        except Exception as e:
            set_status(running=False, message="Download failed", error=str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "message": "Download started"})

@app.get("/api/status")
def api_status():
    with status_lock:
        return jsonify(status)

def start_training_job():
    with status_lock:
        if status["running"]:
            return False
        status.update({
            "running": True,
            "message": "Starting model training...",
            "progress": 0,
            "saved": 0,
            "chunks_done": 0,
            "chunks_total": 0,
            "error": None,
            "metadata": None,
        })

    def worker():
        try:
            global train_module
            train_module = importlib.reload(train_module)

            def progress(message, percent):
                set_status(message=message, progress=percent)

            metadata = train_module.train_model(progress)
            version = metadata.get("model_version", "MODEL") if isinstance(metadata, dict) else "MODEL"
            set_status(
                running=False,
                message=f"MODEL {version} READY",
                progress=100,
                error=None,
                metadata=metadata,
            )
        except Exception as e:
            set_status(running=False, message="Training failed", error=str(e))

    Thread(target=worker, daemon=True).start()
    return True

@app.post("/api/train")
def api_train():
    started = start_training_job()
    if not started:
        return jsonify({"ok": False, "message": "Another operation is running"}), 409
    return jsonify({"ok": True, "message": "Training started"})

@app.post("/api/retrain-if-needed")
def api_retrain_if_needed():
    needs, summary = should_retrain(window=100, min_samples=25, min_accuracy=0.80, max_mape=0.50)
    if not needs:
        return jsonify({"ok": True, "triggered": False, "summary": summary})
    started = start_training_job()
    if not started:
        return jsonify({"ok": True, "triggered": False, "message": "Model update skipped because another job is running", "summary": summary})
    return jsonify({"ok": True, "triggered": True, "summary": summary})

def _sync_prediction_feedback(token):
    try:
        current = get_chart_history("1D", token)
        candles = current.get("candles", []) if isinstance(current, dict) else []
        if not candles:
            return
        latest = candles[-1]
        latest_close = float(latest.get("close", 0.0) or 0.0)
        for item in list(PREDICTION_LEDGER):
            if item.get("resolved"):
                continue
            predicted = float(item.get("expected_points", 0.0) or 0.0)
            base = float(item.get("current_price", 0.0) or 0.0)
            actual_points = latest_close - base
            actual_direction = "UP" if actual_points > 0 else "DOWN" if actual_points < 0 else "FLAT"
            record_prediction(
                predicted_direction=item.get("direction", "FLAT"),
                expected_points=predicted,
                actual_points=actual_points,
                actual_direction=actual_direction,
                model_version=item.get("model_version", "unknown"),
            )
            item["resolved"] = True
            item["actual_points"] = actual_points
            item["actual_direction"] = actual_direction
    except Exception:
        pass

def _run_live_prediction_cycle(token):
    if not token:
        return None
    try:
        result = predict_live(token)
        next_5m = result.get("next_5m", {}) if isinstance(result, dict) else {}
        record = {
            "timestamp": result.get("timestamp"),
            "current_price": float(result.get("current_price", 0.0) or 0.0),
            "direction": str(next_5m.get("direction", "FLAT")).upper(),
            "expected_points": float(next_5m.get("expected_points", 0.0) or 0.0),
            "model_version": str(result.get("model_version", "unknown")),
            "resolved": False,
        }
        PREDICTION_LEDGER.append(record)
        if len(PREDICTION_LEDGER) > 120:
            PREDICTION_LEDGER[:] = PREDICTION_LEDGER[-120:]
        _sync_prediction_feedback(token)
        summary = calculate_accuracy(window=100)
        should_retrain_flag, retrain_summary = should_retrain(window=100, min_samples=25, min_accuracy=0.80, max_mape=0.50)
        if should_retrain_flag and not status.get("running"):
            try:
                start_training_job()
            except Exception:
                pass
        result["accuracy"] = summary
        result["retrain_needed"] = should_retrain_flag
        result["retrain_summary"] = retrain_summary
        return result
    except Exception:
        return None


def _live_monitor_loop():
    while True:
        time.sleep(60)
        if not market_is_open():
            continue
        token = read_server_token()
        if not token:
            continue
        _run_live_prediction_cycle(token)


def start_live_monitor():
    global LIVE_MONITOR_THREAD
    with LIVE_MONITOR_LOCK:
        if LIVE_MONITOR_THREAD is not None and LIVE_MONITOR_THREAD.is_alive():
            return
        LIVE_MONITOR_THREAD = Thread(target=_live_monitor_loop, daemon=True)
        LIVE_MONITOR_THREAD.start()


@app.post("/api/predict")
def api_predict():
    token = _token_from_request()
    if not token:
        return jsonify({"ok": False, "message": "Enter/test the Upstox token first."}), 400
    try:
        result = _run_live_prediction_cycle(token)
        if result is None:
            raise RuntimeError("Prediction cycle failed. Check token and market data.")
        start_live_monitor()
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/chart-predictions")
def api_chart_predictions():
    try:
        rng = request.args.get("range", "1D").upper()
        if not TOKEN:
            return jsonify({"ok": False, "message": "Token not saved."}), 400

        payload = get_chart_history(rng, TOKEN)
        rows = payload.get("candles", []) if isinstance(payload, dict) else []
        if not rows:
            return jsonify({"ok": True, "predictions": [], "count": 0})

        import joblib
        import pandas as pd
        import live_prediction as lp

        model_dir = Path(__file__).resolve().parent / "models"
        m5 = joblib.load(model_dir / "nifty_v5_5m_points.pkl")
        dm = joblib.load(model_dir / "nifty_v5_5m_direction.pkl")

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ("open","high","low","close","volume","open_interest"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df=(df.dropna(subset=["timestamp","open","high","low","close"])
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True))

        results=[]
        warmup=60
        for idx in range(warmup-1, len(df)):
            try:
                x, cols, _ = lp._make_input(df.iloc[:idx+1].copy())
                row=x.iloc[[-1]][cols]
                pts=float(m5.predict(row)[0])
                probs=dm.predict_proba(row)[0]
                classes=list(getattr(dm,"classes_",range(len(probs))))
                pm={int(k):float(v) for k,v in zip(classes,probs)}
                choices=[(pm.get(0,0.0),"DOWN"),(pm.get(1,0.0),"FLAT"),(pm.get(2,0.0),"UP")]
                conf,direction=max(choices,key=lambda z:z[0])
                cur=float(df["close"].iloc[idx])
                results.append({
                    "index":idx,
                    "timestamp":df["timestamp"].iloc[idx].isoformat(),
                    "prediction":round(cur+pts,2),
                    "expected_points":round(pts,2),
                    "direction":direction,
                    "confidence":round(float(conf),4)
                })
            except Exception:
                continue
        return jsonify({"ok":True,"range":rng,"predictions":results,"count":len(results)})
    except Exception as e:
        return jsonify({"ok":False,"message":str(e)}),400

@app.get("/api/chart-history")
def api_chart_history():
    try:
        rng = request.args.get("range", "1D").upper()
        return jsonify({"ok": True, **get_chart_history(rng, TOKEN)})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400

@app.get("/api/check-data")
def api_check_data():
    return jsonify(check_database())

@app.get("/api/model-status")
def api_model_status():
    metadata_path = Path(__file__).resolve().parent / "models" / "model_metadata.json"
    if not metadata_path.exists():
        return jsonify({"ready": False})
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return jsonify({
            "ready": True,
            "metadata": metadata,
            "old": False,
        })
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)})

@app.get("/api/model-feedback")
def api_model_feedback():
    return jsonify({"ok": True, "summary": calculate_accuracy(window=100)})

@app.get("/api/retrain-status")
def api_retrain_status():
    needs, summary = should_retrain(window=100, min_samples=25, min_accuracy=0.80, max_mape=0.50)
    return jsonify({"ok": True, "needs_retrain": needs, "summary": summary})

@app.get("/api/stats")
def api_stats():
    return jsonify(get_stats())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
