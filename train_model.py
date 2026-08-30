
import json
import sqlite3
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from config import DB_PATH, MODEL_DIR
from features import make_features
from pattern_features import add_pattern_features

MODEL_VERSION = "V5.1"

# Two 5-minute bars are treated as consecutive only if their timestamps are at
# most 5*horizon + this many minutes apart. Absorbs the odd missing bar without
# ever bridging the overnight break.
SESSION_GAP_SLACK_MIN = 10

# UP/DOWN require a move larger than DIRECTION_ATR_MULT * ATR, floored so a
# zero-ATR warm-up row still gets a real dead band.
DIRECTION_ATR_MULT = 0.18
MIN_DIRECTION_BAND_PTS = 1.0

# Round-trip friction assumed by the backtest, in NIFTY points. NIFTY 50 itself
# is an index and cannot be traded; the tradeable instrument is the future or an
# option, so a fill costs at least the bid/ask plus brokerage and taxes. 2 points
# is a deliberately optimistic figure for the front-month future.
ROUND_TRIP_COST_PTS = 2.0

# Held-out rounds without improvement before a booster stops adding trees.
EARLY_STOPPING_ROUNDS = 50

# An edge must clear this many standard errors before it is called real rather
# than luck. 2.0 is the usual two-sigma convention.
MIN_EDGE_SIGMAS = 2.0


def _verdict(metrics):
    """Turn the raw metrics into a plain statement of whether this is tradeable.

    Written so the dashboard cannot show an accuracy number without the verdict
    beside it. The reasons list is what the trader actually needs to read.
    """
    d = metrics.get("direction", {})
    r = metrics.get("5m", {})
    edge = float(d.get("Edge", 0.0))
    sigmas = float(d.get("EdgeSigmas", 0.0))
    r2 = float(r.get("R2", 0.0))

    reasons = []
    if sigmas < MIN_EDGE_SIGMAS:
        reasons.append(
            f"Direction edge over the always-guess-the-common-class baseline is "
            f"{edge * 100:+.2f}% ({sigmas:.1f} sigma) — not statistically distinguishable from chance."
        )
    if r2 <= 0:
        reasons.append(
            f"Point forecast R2 is {r2:+.4f}. Anything at or below zero means the "
            f"model predicts the size of the next move worse than just guessing the average move."
        )
    if float(r.get("directional_edge", 0.0)) <= 0:
        reasons.append(
            "The point model's sign is no better than the majority direction in the test window."
        )

    tradeable = not reasons
    return {
        "tradeable": tradeable,
        "headline": ("Edge detected — validate on paper before risking money."
                     if tradeable else
                     "NO TRADEABLE EDGE. Do not risk money on these signals."),
        "reasons": reasons,
        "min_edge_sigmas": MIN_EDGE_SIGMAS,
    }


def load_data():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume, open_interest FROM nifty_5min ORDER BY timestamp ASC",
            con,
        )


def _safe_sign(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def build_features(df):
    x, _ = make_features(df)
    x, labels = add_pattern_features(x)
    c = x["close"].astype(float)
    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)

    for n in (1, 2, 3, 5, 10, 20):
        x[f"ret_lag_{n}"] = c.pct_change(n)
        x[f"close_lag_pct_{n}"] = (c / c.shift(n)) - 1.0
    x["high_dist_1"] = h / c - 1.0
    x["low_dist_1"] = l / c - 1.0
    x["open_dist_1"] = o / c - 1.0
    r = c.pct_change()
    for n in (3, 5, 10):
        x[f"ret_mean_{n}"] = r.rolling(n).mean()
        x[f"ret_std_{n}"] = r.rolling(n).std()

    ts = pd.to_datetime(x["timestamp"], errors="coerce")
    mins = ts.dt.hour * 60 + ts.dt.minute
    x["time_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    x["time_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    x["pattern_name"] = labels.astype(str)
    return x


def _forward_move(c, ts, horizon):
    """Close-to-close move `horizon` bars ahead, NaN when the two bars are not in
    the same trading session.

    Without the gap check, the last bar of each day (15:25) gets tomorrow's 09:15
    as its "next 5 minutes". That overnight jump is not a 5-minute move: those
    rows carry moves roughly 7x larger than a real intraday bar, and the model
    spends its capacity trying to predict gaps it cannot see.
    """
    move = c.shift(-horizon) - c
    gap = ts.shift(-horizon) - ts
    max_gap = pd.Timedelta(minutes=5 * horizon + SESSION_GAP_SLACK_MIN)
    return move.where(gap <= max_gap)


def _prepare_training_frame(df):
    x = build_features(df)
    c = x["close"].astype(float)
    ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)

    # No .fillna(0.0) here: a row whose future is unknown (end of series) or
    # unusable (overnight gap) must be dropped, not relabelled as "price did not
    # move". Filling with 0.0 previously made the dropna() below a no-op and
    # taught the classifier that 752 session-boundary rows were FLAT.
    x["target_5m"] = _forward_move(c, ts, 1)
    x["target_10m"] = _forward_move(c, ts, 2)

    atr = (x["atr_pct"].astype(float) * c).abs()
    # Floor the band so a warm-up row with atr == 0 cannot get threshold 0, which
    # would force every non-zero move into UP or DOWN and never FLAT.
    threshold = np.maximum(DIRECTION_ATR_MULT * atr.fillna(0.0), MIN_DIRECTION_BAND_PTS)
    x["direction_5m"] = np.where(
        x["target_5m"] > threshold, 2,
        np.where(x["target_5m"] < -threshold, 0, 1),
    )
    x["pattern_code"] = x.get("pattern_code", 0)
    excluded = {
        "timestamp", "open", "high", "low", "close", "volume", "open_interest",
        "target_5m", "target_10m", "direction_5m", "pattern_name", "pattern_code"
    }
    cols = [col for col in x.columns if col not in excluded and pd.api.types.is_numeric_dtype(x[col])]
    work = x.dropna(subset=["target_5m", "target_10m"]).copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    for col in cols:
        # ffill only. bfill copies a later candle's value backwards into a warm-up
        # row, which is look-ahead at training time and unreproducible at
        # inference time (the live frame has no "later" candle to borrow from).
        work[col] = pd.to_numeric(work[col], errors="coerce").ffill().fillna(0.0)
    return work, cols


def _build_regressor(early_stopping_rounds=None):
    return XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.025,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def _build_classifier(early_stopping_rounds=None):
    return XGBClassifier(
        n_estimators=900,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def walk_forward_backtest(df, test_size=0.2, window=2000, step=250, horizon_bars=50,
                          cost_points=ROUND_TRIP_COST_PTS, min_conf=0.40):
    """Retrain on a rolling window, trade the next `horizon_bars`, repeat.

    Every fold trains only on bars strictly before the bars it is scored on, so
    the equity curve here is the honest answer to "would this have made money".
    A trade is taken only when the classifier picks UP or DOWN (never FLAT) with
    probability >= min_conf, and each trade pays `cost_points` round-trip.
    """
    work, cols = _prepare_training_frame(df)
    empty = {
        "accuracy": 0.0, "mae": 0.0, "rmse": 0.0, "precision": 0.0, "recall": 0.0,
        "f1": 0.0, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "drawdown": 0.0, "gross_points": 0.0, "net_points": 0.0,
        "expectancy_points": 0.0, "cost_points": float(cost_points),
    }
    if len(work) < window + horizon_bars * 2:
        return {"status": "not_enough_data", "metrics": empty}

    n = len(work)
    start_at = max(window, int(n * (1 - test_size)))
    rows = []

    for start in range(start_at, n - horizon_bars, step):
        train_slice = work.iloc[start - window:start]
        test_slice = work.iloc[start:start + horizon_bars]
        if len(train_slice) < window // 2 or len(test_slice) < horizon_bars // 2:
            continue
        if train_slice["direction_5m"].nunique() < 3:
            continue

        reg_5 = _build_regressor()
        clf = _build_classifier()
        reg_5.fit(train_slice[cols], train_slice["target_5m"])
        clf.fit(train_slice[cols], train_slice["direction_5m"])

        pred_5 = reg_5.predict(test_slice[cols])
        proba = clf.predict_proba(test_slice[cols])
        pred_dir = clf.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1)

        actual_5 = test_slice["target_5m"].to_numpy(dtype=float)
        actual_dir = test_slice["direction_5m"].to_numpy(dtype=int)

        # Positional indexing throughout. The previous version indexed the
        # prediction arrays with len(predictions) % len(pred_5), which paired each
        # actual with an unrelated prediction and made every metric below noise.
        for i in range(len(test_slice)):
            rows.append({
                "actual_5m": float(actual_5[i]),
                "predicted_5m": float(pred_5[i]),
                "actual_dir": int(actual_dir[i]),
                "predicted_dir": int(pred_dir[i]),
                "confidence": float(conf[i]),
            })

    if not rows:
        return {"status": "no_predictions", "metrics": empty}

    actual_dir = np.array([r["actual_dir"] for r in rows])
    pred_dir = np.array([r["predicted_dir"] for r in rows])
    actual_5m = np.array([r["actual_5m"] for r in rows], dtype=float)
    pred_5m = np.array([r["predicted_5m"] for r in rows], dtype=float)
    conf = np.array([r["confidence"] for r in rows], dtype=float)

    accuracy = accuracy_score(actual_dir, pred_dir)
    mae = mean_absolute_error(actual_5m, pred_5m)
    rmse = float(np.sqrt(mean_squared_error(actual_5m, pred_5m)))
    precision = precision_score(actual_dir, pred_dir, average="macro", zero_division=0)
    recall = recall_score(actual_dir, pred_dir, average="macro", zero_division=0)
    f1 = f1_score(actual_dir, pred_dir, average="macro", zero_division=0)

    # Only directional calls above the confidence floor become trades. FLAT is a
    # decision to stay out, so it must not be scored as a winning trade — the old
    # np.sign() comparison collapsed FLAT(1) and UP(2) to the same value and
    # counted "predicted FLAT, price rose" as a win.
    take = (pred_dir != 1) & (conf >= min_conf)
    side = np.where(pred_dir == 2, 1.0, -1.0)[take]
    gross = side * actual_5m[take]
    net = gross - cost_points
    trades = int(take.sum())

    if trades:
        win_rate = float((net > 0).mean())
        gross_profit = float(net[net > 0].sum())
        gross_loss = float(abs(net[net < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        cumulative = np.cumsum(net)
        peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        drawdown = float(np.max(peak - cumulative))
        expectancy = float(net.mean())
        dir_hit = float((side == np.where(actual_5m[take] > 0, 1.0, -1.0)).mean())
    else:
        win_rate = profit_factor = drawdown = expectancy = dir_hit = 0.0
        gross = net = np.array([])

    return {
        "status": "ok",
        "metrics": {
            "accuracy": float(accuracy),
            "mae": float(mae),
            "rmse": rmse,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "folds": int(len(rows) // horizon_bars),
            "scored_bars": int(len(rows)),
            "trades": trades,
            "trade_rate": float(trades / len(rows)),
            "directional_hit_rate": dir_hit,
            "win_rate": win_rate,
            "profit_factor": float(profit_factor),
            "drawdown": drawdown,
            "gross_points": float(gross.sum()) if trades else 0.0,
            "net_points": float(net.sum()) if trades else 0.0,
            "expectancy_points": expectancy,
            "cost_points": float(cost_points),
        },
    }


def train_model(progress=None):
    df = load_data()
    if len(df) < 2000:
        raise RuntimeError(f"Not enough candles: {len(df)}")
    if progress:
        progress(f"Preparing V5.1 features from {len(df):,} candles...", 10)

    work, cols = _prepare_training_frame(df)
    n = len(work)
    tr = int(n * 0.70)
    va = int(n * 0.85)

    Xtr, Xva, Xte = work[cols].iloc[:tr], work[cols].iloc[tr:va], work[cols].iloc[va:]

    if progress:
        progress("Training 5-minute point model...", 30)
    # The 70-85% slice used to be cut out and never used. Feeding it as an
    # eval_set stops each booster at the round where held-out error stops
    # improving instead of always burning all 1000 trees into the training set.
    m5 = _build_regressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    m10 = _build_regressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    m5.fit(Xtr, work["target_5m"].iloc[:tr], eval_set=[(Xva, work["target_5m"].iloc[tr:va])], verbose=False)
    m10.fit(Xtr, work["target_10m"].iloc[:tr], eval_set=[(Xva, work["target_10m"].iloc[tr:va])], verbose=False)

    if progress:
        progress("Training UP/FLAT/DOWN classifier...", 60)
    clf = _build_classifier(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    clf.fit(Xtr, work["direction_5m"].iloc[:tr],
            eval_set=[(Xva, work["direction_5m"].iloc[tr:va])], verbose=False)

    if progress:
        progress("Training learned pattern model...", 75)
    pattern_mask = work.get("pattern_code", 0).astype(int) > 0
    pw = work.loc[pattern_mask]
    pmodel = None
    if len(pw) >= 300 and pw["direction_5m"].nunique() >= 3:
        pmodel = _build_classifier()
        cut = int(len(pw) * 0.85)
        pmodel.fit(pw[cols].iloc[:cut], pw["direction_5m"].iloc[:cut])

    y5 = work["target_5m"].iloc[va:]
    p5 = m5.predict(Xte)
    yd = work["direction_5m"].iloc[va:]
    pd_ = clf.predict(Xte)

    # The comparators. An accuracy figure with no baseline next to it is not
    # interpretable: on this label set "always predict the most common class"
    # already scores ~0.37, so a 0.38 classifier has essentially no edge. Store
    # both so the dashboard can never present accuracy as if it were skill.
    dir_baseline = float(yd.value_counts(normalize=True).max()) if len(yd) else 0.0
    dir_accuracy = float(accuracy_score(yd, pd_))
    nz = y5 != 0
    sign_accuracy = float((np.sign(y5[nz]) == np.sign(p5[nz.to_numpy()])).mean()) if nz.any() else 0.0
    sign_baseline = float(max((y5[nz] > 0).mean(), (y5[nz] < 0).mean())) if nz.any() else 0.0
    # Standard error of a proportion at this sample size. An edge smaller than
    # ~2 SE is indistinguishable from luck.
    dir_se = float(np.sqrt(0.25 / len(yd))) if len(yd) else 0.0

    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "rows_total": int(len(df)),
        "rows_used": int(len(work)),
        "rows_dropped_unusable_target": int(len(df) - len(work)),
        "feature_count": int(len(cols)),
        "historical_start": str(df["timestamp"].min()),
        "historical_end": str(df["timestamp"].max()),
        "train_rows": tr,
        "validation_rows": va - tr,
        "test_rows": n - va,
        "best_iteration": {
            "5m": int(getattr(m5, "best_iteration", 0) or 0),
            "10m": int(getattr(m10, "best_iteration", 0) or 0),
            "direction": int(getattr(clf, "best_iteration", 0) or 0),
        },
        "metrics": {
            "5m": {
                "MAE": float(mean_absolute_error(y5, p5)),
                "RMSE": float(np.sqrt(mean_squared_error(y5, p5))),
                "R2": float(r2_score(y5, p5)),
                "directional_accuracy": sign_accuracy,
                "directional_baseline": sign_baseline,
                "directional_edge": sign_accuracy - sign_baseline,
            },
            "direction": {
                "Accuracy": dir_accuracy,
                "Baseline": dir_baseline,
                "Edge": dir_accuracy - dir_baseline,
                "StdErr": dir_se,
                "EdgeSigmas": (dir_accuracy - dir_baseline) / dir_se if dir_se else 0.0,
                "F1": float(f1_score(yd, pd_, average="macro", zero_division=0)),
                "Precision": float(precision_score(yd, pd_, average="macro", zero_division=0)),
                "Recall": float(recall_score(yd, pd_, average="macro", zero_division=0)),
            },
            "pattern": {"enabled": pmodel is not None, "samples": int(len(pw))},
        },
    }
    meta["verdict"] = _verdict(meta["metrics"])

    stats = {}
    if len(pw):
        trainp = pw.iloc[: max(1, int(len(pw) * 0.70))]
        for _, grp in trainp.groupby("pattern_code"):
            name = str(grp["pattern_name"].iloc[0])
            vals = grp["direction_5m"].astype(int)
            implied = 2 if name in {"HAMMER", "BULLISH_ENGULFING", "BREAKOUT_UP", "HIGHER_HIGH_LOWER_LOW"} else 0 if name in {"SHOOTING_STAR", "BEARISH_ENGULFING", "BREAKOUT_DOWN", "LOWER_HIGH_LOWER_LOW"} else 1
            stats[name] = {"count": int(len(grp)), "hit_rate": float((vals == implied).mean()), "implied_direction": int(implied)}

    joblib.dump(m5, MODEL_DIR / "nifty_v5_5m_points.pkl")
    joblib.dump(m10, MODEL_DIR / "nifty_v5_10m_points.pkl")
    joblib.dump(clf, MODEL_DIR / "nifty_v5_5m_direction.pkl")
    if pmodel is not None:
        joblib.dump(pmodel, MODEL_DIR / "nifty_v5_pattern_direction.pkl")
    elif (MODEL_DIR / "nifty_v5_pattern_direction.pkl").exists():
        (MODEL_DIR / "nifty_v5_pattern_direction.pkl").unlink()

    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
    (MODEL_DIR / "nifty_v5_pattern_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if progress:
        progress("MODEL V5.1 READY — ML + pattern learning + live confirmation layer", 100)
    gc.collect()
    return meta


if __name__ == "__main__":
    print(json.dumps(train_model(lambda m, p: print(f"[{p}%] {m}")), indent=2))
