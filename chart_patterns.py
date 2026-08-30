"""Causal chart-structure detection for overlays and live confirmation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ts_iso(value) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.isoformat()


def _local_extrema(values: np.ndarray, order: int = 3) -> tuple[list[int], list[int]]:
    peaks: list[int] = []
    troughs: list[int] = []
    n = len(values)
    if n < order * 2 + 1:
        return peaks, troughs
    for i in range(order, n - order):
        window = values[i - order : i + order + 1]
        center = values[i]
        if not np.isfinite(center):
            continue
        if center >= np.nanmax(window) and center >= values[i - 1] and center >= values[i + 1]:
            peaks.append(i)
        if center <= np.nanmin(window) and center <= values[i - 1] and center <= values[i + 1]:
            troughs.append(i)
    return peaks, troughs


def _line_from_points(xs: list[int], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _cluster_levels(prices: list[float], tolerance: float) -> list[dict[str, Any]]:
    if not prices:
        return []
    ordered = sorted(float(p) for p in prices if np.isfinite(p))
    clusters: list[dict[str, Any]] = []
    for price in ordered:
        if clusters and abs(price - clusters[-1]["mean"]) <= max(tolerance, 1e-9):
            clusters[-1]["vals"].append(price)
            clusters[-1]["mean"] = float(np.mean(clusters[-1]["vals"]))
        else:
            clusters.append({"vals": [price], "mean": price})
    clusters.sort(key=lambda c: (-len(c["vals"]), -len(c["vals"]) * 0 + abs(c["mean"])))
    clusters.sort(key=lambda c: -len(c["vals"]))
    return clusters


def _trendline_overlay(kind: str, role: str, i0: int, i1: int, y0: float, y1: float, timestamps) -> dict[str, Any]:
    return {
        "kind": kind,
        "role": role,
        "points": [
            {"index": int(i0), "timestamp": _ts_iso(timestamps[i0]), "price": float(y0)},
            {"index": int(i1), "timestamp": _ts_iso(timestamps[i1]), "price": float(y1)},
        ],
    }


def _horizontal_overlay(role: str, price: float, i0: int, i1: int, timestamps) -> dict[str, Any]:
    return {
        "kind": "horizontal",
        "role": role,
        "price": float(price),
        "points": [
            {"index": int(i0), "timestamp": _ts_iso(timestamps[i0]), "price": float(price)},
            {"index": int(i1), "timestamp": _ts_iso(timestamps[i1]), "price": float(price)},
        ],
    }


def _label_overlay(text: str, index: int, price: float, timestamps, tone: str = "neutral") -> dict[str, Any]:
    return {
        "kind": "label",
        "role": tone,
        "text": text,
        "points": [{"index": int(index), "timestamp": _ts_iso(timestamps[index]), "price": float(price)}],
    }


def _box_overlay(role: str, i0: int, i1: int, y0: float, y1: float, timestamps) -> dict[str, Any]:
    return {
        "kind": "box",
        "role": role,
        "y0": float(min(y0, y1)),
        "y1": float(max(y0, y1)),
        "points": [
            {"index": int(i0), "timestamp": _ts_iso(timestamps[i0]), "price": float(max(y0, y1))},
            {"index": int(i1), "timestamp": _ts_iso(timestamps[i1]), "price": float(min(y0, y1))},
        ],
    }


def detect_chart_structures(df: pd.DataFrame, lookback: int = 80) -> dict[str, Any]:
    empty = {
        "pattern": "NONE",
        "bias": "NEUTRAL",
        "support": None,
        "resistance": None,
        "overlays": [],
        "levels": [],
        "trendline": [],
        "signal_hint": "WAIT",
        "stop_candidates": [],
    }
    if df is None or df.empty:
        return empty

    work = df.tail(max(24, int(lookback))).copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close"])
    if len(work) < 16:
        return empty

    highs = work["high"].to_numpy(dtype=float)
    lows = work["low"].to_numpy(dtype=float)
    closes = work["close"].to_numpy(dtype=float)
    timestamps = work["timestamp"].to_numpy() if "timestamp" in work.columns else np.arange(len(work))
    n = len(work)
    last = n - 1
    current = float(closes[-1])
    atr = float(np.nanmean(highs - lows))
    atr = atr if np.isfinite(atr) and atr > 0 else max(abs(current) * 0.002, 1.0)
    order = 2 if n < 40 else 3

    peaks, troughs = _local_extrema(highs, order=order)
    _, troughs_low = _local_extrema(lows, order=order)
    troughs = sorted(set(troughs + troughs_low))

    overlays: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    stop_candidates: list[float] = []
    pattern = "NONE"
    bias = "NEUTRAL"
    signal_hint = "WAIT"

    peak_pts = [(i, float(highs[i])) for i in peaks]
    trough_pts = [(i, float(lows[i])) for i in troughs]

    recent_peaks = peak_pts[-4:] if len(peak_pts) >= 2 else peak_pts
    recent_troughs = trough_pts[-4:] if len(trough_pts) >= 2 else trough_pts

    upper = _line_from_points([p[0] for p in recent_peaks], [p[1] for p in recent_peaks]) if len(recent_peaks) >= 2 else None
    lower = _line_from_points([p[0] for p in recent_troughs], [p[1] for p in recent_troughs]) if len(recent_troughs) >= 2 else None

    i0, i1 = 0, last
    upper_y0 = upper_y1 = lower_y0 = lower_y1 = None
    if upper is not None:
        us, ui = upper
        upper_y0 = ui + us * i0
        upper_y1 = ui + us * i1
        overlays.append(_trendline_overlay("trendline", "resistance", i0, i1, upper_y0, upper_y1, timestamps))
    if lower is not None:
        ls, li = lower
        lower_y0 = li + ls * i0
        lower_y1 = li + ls * i1
        overlays.append(_trendline_overlay("trendline", "support", i0, i1, lower_y0, lower_y1, timestamps))

    if upper is not None and lower is not None and upper_y0 is not None and lower_y0 is not None:
        start_gap = abs(upper_y0 - lower_y0)
        end_gap = abs(upper_y1 - lower_y1)
        converging = end_gap < start_gap * 0.78 and start_gap > atr * 1.2
        us, _ = upper
        ls, _ = lower
        if converging and us < -1e-6 and ls > 1e-6:
            pattern = "SYMMETRICAL_TRIANGLE"
        elif converging and us < -1e-6 and ls < 1e-4:
            pattern = "DESCENDING_TRIANGLE" if abs(ls) < abs(us) * 0.35 else "FALLING_WEDGE"
        elif converging and ls > 1e-6 and us > -1e-4:
            pattern = "ASCENDING_TRIANGLE" if abs(us) < ls * 0.35 else "RISING_WEDGE"
        elif converging:
            pattern = "CONVERGING_WEDGE"

        buffer = atr * 0.15
        if pattern != "NONE":
            if current > upper_y1 + buffer:
                bias = "UP"
                signal_hint = "BUY"
                overlays.append(_box_overlay("breakout", max(0, last - 1), last, min(current, upper_y1), max(current, highs[-1]), timestamps))
                overlays.append(_label_overlay(f"{pattern.replace('_', ' ').title()} breakout", last, current, timestamps, "up"))
            elif current < lower_y1 - buffer:
                bias = "DOWN"
                signal_hint = "SELL"
                overlays.append(_box_overlay("breakdown", max(0, last - 1), last, min(current, lows[-1]), max(current, lower_y1), timestamps))
                overlays.append(_label_overlay(f"{pattern.replace('_', ' ').title()} breakdown", last, current, timestamps, "down"))

    clusters = _cluster_levels([p[1] for p in trough_pts] + [p[1] for p in peak_pts], tolerance=atr * 0.35)
    support = None
    resistance = None
    below = [c["mean"] for c in clusters if c["mean"] <= current]
    above = [c["mean"] for c in clusters if c["mean"] >= current]
    if below:
        support = float(max(below))
    else:
        support = float(np.nanmin(lows[-30:]))
    if above:
        resistance = float(min(above))
    else:
        resistance = float(np.nanmax(highs[-30:]))

    if support is not None:
        overlays.append(_horizontal_overlay("support", support, 0, last, timestamps))
        levels.append({"type": "support", "value": support})
        stop_candidates.append(support)
    if resistance is not None:
        overlays.append(_horizontal_overlay("resistance", resistance, 0, last, timestamps))
        levels.append({"type": "resistance", "value": resistance})

    cup = _detect_cup_and_handle(highs, lows, closes, timestamps, atr, current, last)
    if cup["found"] and pattern in {"NONE", "CONVERGING_WEDGE"}:
        pattern = cup["pattern"]
        overlays.extend(cup["overlays"])
        stop_candidates.extend(cup["stops"])
        if cup["bias"] != "NEUTRAL":
            bias = cup["bias"]
            signal_hint = cup["signal_hint"]
        if cup.get("levels"):
            levels.extend(cup["levels"])

    trendline = []
    if upper is not None:
        trendline = overlays[0]["points"] if overlays else []
    elif lower is not None and overlays:
        trendline = overlays[0]["points"]

    return {
        "pattern": pattern,
        "bias": bias,
        "support": None if support is None else float(support),
        "resistance": None if resistance is None else float(resistance),
        "overlays": overlays,
        "levels": levels,
        "trendline": trendline,
        "signal_hint": signal_hint,
        "stop_candidates": [float(x) for x in stop_candidates if np.isfinite(x)],
        "current": current,
        "lookback": int(n),
    }


def _detect_cup_and_handle(highs, lows, closes, timestamps, atr, current, last) -> dict[str, Any]:
    result = {"found": False, "pattern": "NONE", "bias": "NEUTRAL", "signal_hint": "WAIT", "overlays": [], "stops": [], "levels": []}
    n = len(closes)
    if n < 28:
        return result

    search = closes[: max(18, int(n * 0.82))]
    bottom_i = int(np.argmin(search))
    if bottom_i < 6 or bottom_i > n - 10:
        return result

    left_i = int(np.argmax(highs[: bottom_i + 1]))
    right_slice_end = min(n - 4, bottom_i + max(8, (last - bottom_i) * 2 // 3 + 8))
    if right_slice_end <= bottom_i + 4:
        return result
    right_i = bottom_i + 1 + int(np.argmax(highs[bottom_i + 1 : right_slice_end + 1]))
    left_p = float(highs[left_i])
    right_p = float(highs[right_i])
    bottom_p = float(lows[bottom_i])
    rim = (left_p + right_p) / 2.0
    depth = rim - bottom_p
    if depth < atr * 2.2:
        return result
    if abs(left_p - right_p) > max(atr * 1.8, rim * 0.012):
        return result
    if right_i <= left_i + 8:
        return result

    handle_highs = highs[right_i:]
    handle_lows = lows[right_i:]
    if len(handle_highs) < 3:
        return result
    handle_res = float(np.max(handle_highs[:-1])) if len(handle_highs) > 1 else float(handle_highs[0])
    handle_sup = float(np.min(handle_lows))
    cup_mid = bottom_p + depth * 0.5
    if handle_sup < cup_mid:
        return result

    xs = np.linspace(left_i, right_i, max(8, right_i - left_i + 1))
    # U-shape through left rim, bottom, right rim
    curve_pts = []
    span = max(right_i - left_i, 1)
    for x in xs:
        t = (x - left_i) / span
        price = left_p * (1 - t) + right_p * t + 4 * (bottom_p - (left_p + right_p) / 2.0) * t * (1 - t)
        xi = int(round(x))
        xi = min(max(xi, 0), last)
        curve_pts.append({"index": xi, "timestamp": _ts_iso(timestamps[xi]), "price": float(price)})

    overlays = [{
        "kind": "curve",
        "role": "cup",
        "points": curve_pts,
    }]
    if len(handle_highs) >= 2:
        overlays.append(_trendline_overlay("trendline", "handle_resistance", right_i, last, float(handle_highs[0]), float(handle_highs[-1]), timestamps))
        overlays.append(_trendline_overlay("trendline", "handle_support", right_i, last, float(handle_lows[0]), float(handle_lows[-1]), timestamps))
    overlays.append(_horizontal_overlay("handle_stop", handle_sup, right_i, last, timestamps))
    overlays.append(_horizontal_overlay("handle_stop", handle_sup + atr * 0.25, right_i, last, timestamps))
    overlays.append(_label_overlay("Possible stop loss locations", right_i, handle_sup + atr * 0.12, timestamps, "neutral"))

    bias = "NEUTRAL"
    hint = "WAIT"
    pattern = "CUP_AND_HANDLE"
    if current > handle_res + atr * 0.05:
        bias = "UP"
        hint = "BUY"
        overlays.append(_box_overlay("breakout", max(right_i, last - 1), last, handle_res, max(current, highs[-1]), timestamps))
        overlays.append(_label_overlay("Buy handle breakout", last, current, timestamps, "up"))
        pattern = "CUP_HANDLE_BREAKOUT"

    result.update({
        "found": True,
        "pattern": pattern,
        "bias": bias,
        "signal_hint": hint,
        "overlays": overlays,
        "stops": [handle_sup, handle_sup + atr * 0.25],
        "levels": [
            {"type": "support", "value": float(handle_sup)},
            {"type": "breakout", "value": float(handle_res)},
        ],
    })
    return result


def structure_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling structure flags used by training and live features."""
    x = df.copy()
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")
    rng20 = (h.rolling(20, min_periods=8).max() - l.rolling(20, min_periods=8).min()).replace(0, np.nan)
    rng10 = (h.rolling(10, min_periods=5).max() - l.rolling(10, min_periods=5).min()).replace(0, np.nan)
    x["range_compression"] = (rng10 / rng20 - 1.0).replace([np.inf, -np.inf], np.nan)
    prev_high20 = h.shift(1).rolling(20, min_periods=8).max()
    prev_low20 = l.shift(1).rolling(20, min_periods=8).min()
    x["dist_to_20_high"] = (c / prev_high20.replace(0, np.nan) - 1.0)
    x["dist_to_20_low"] = (c / prev_low20.replace(0, np.nan) - 1.0)
    x["triangle_compressing"] = (x["range_compression"] < -0.12).astype(int)
    return x


def attach_overlays_to_history(candles: list[dict[str, Any]]) -> dict[str, Any]:
    if not candles:
        return detect_chart_structures(pd.DataFrame())
    df = pd.DataFrame(candles)
    return detect_chart_structures(df, lookback=min(120, max(40, len(df))))
