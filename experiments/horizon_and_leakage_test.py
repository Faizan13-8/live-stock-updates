"""Head-to-head test: does fixing the target bugs make the model actually predict?

Compares the current training recipe against session-aware targets, no-bfill,
binary framing, and longer horizons. Every variant uses the same temporal
70/15/15 split and reports metrics on the final 15% only.

Run: .venv/Scripts/python.exe experiments/horizon_and_leakage_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_model as tm

EXCLUDED = {
    "timestamp", "open", "high", "low", "close", "volume", "open_interest",
    "target", "direction", "pattern_name", "pattern_code",
}


def prep(x, ts, c, horizon, session_aware, use_bfill, thr_mult=0.18):
    """Build (frame, cols) for one variant. x/ts/c are precomputed once."""
    x = x.copy()
    tgt = c.shift(-horizon) - c
    valid = tgt.notna()
    if session_aware:
        # A 5-minute bar horizon of h spans 5*h minutes. Allow two missing bars
        # of slack, but never bridge an overnight or lunch-length break.
        gap = ts.shift(-horizon) - ts
        valid &= gap <= pd.Timedelta(minutes=5 * horizon + 10)

    atr = x["atr_pct"].astype(float) * c
    thr = (thr_mult * atr).abs()
    x["target"] = tgt
    x["direction"] = np.where(tgt > thr, 2, np.where(tgt < -thr, 0, 1))

    cols = [col for col in x.columns
            if col not in EXCLUDED and pd.api.types.is_numeric_dtype(x[col])]

    work = x.loc[valid].replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        s = pd.to_numeric(work[col], errors="coerce").ffill()
        if use_bfill:
            s = s.bfill()
        work[col] = s.fillna(0.0)
    return work.reset_index(drop=True), cols


def evaluate(work, cols, label, binary=False):
    n = len(work)
    tr, va = int(n * 0.70), int(n * 0.85)

    Xtr, Xte = work[cols].iloc[:tr], work[cols].iloc[va:]
    ytr, yte = work["target"].iloc[:tr], work["target"].iloc[va:]

    reg = tm._build_regressor()
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)

    if binary:
        # Drop FLAT rows entirely and ask the easier UP-vs-DOWN question.
        mtr = work["direction"].iloc[:tr] != 1
        mte = work["direction"].iloc[va:] != 1
        dtr = (work["direction"].iloc[:tr][mtr] == 2).astype(int)
        dte = (work["direction"].iloc[va:][mte] == 2).astype(int)
        clf = tm._build_classifier()
        clf.set_params(objective="binary:logistic", num_class=None, eval_metric="logloss")
        clf.fit(Xtr[mtr.values], dtr)
        pdir = clf.predict(Xte[mte.values])
        acc = accuracy_score(dte, pdir)
        f1 = f1_score(dte, pdir, average="macro", zero_division=0)
        base = max(float((dte == 1).mean()), float((dte == 0).mean()))
        nclass = 2
    else:
        dtr = work["direction"].iloc[:tr]
        dte = work["direction"].iloc[va:]
        clf = tm._build_classifier()
        clf.fit(Xtr, dtr)
        pdir = clf.predict(Xte)
        acc = accuracy_score(dte, pdir)
        f1 = f1_score(dte, pdir, average="macro", zero_division=0)
        base = float(dte.value_counts(normalize=True).max())
        nclass = 3

    nz = yte != 0
    sign_acc = float((np.sign(yte[nz]) == np.sign(pred[nz])).mean())

    return {
        "variant": label,
        "rows": int(n),
        "test_rows": int(n - va),
        "classes": nclass,
        "MAE": round(float(mean_absolute_error(yte, pred)), 3),
        "RMSE": round(float(np.sqrt(np.mean((yte - pred) ** 2))), 3),
        "R2": round(float(r2_score(yte, pred)), 5),
        "sign_acc": round(sign_acc, 4),
        "clf_acc": round(float(acc), 4),
        "baseline": round(base, 4),
        "edge": round(float(acc) - base, 4),
        "F1": round(float(f1), 4),
    }


def main():
    df = tm.load_data()
    print(f"loaded {len(df):,} candles", flush=True)
    x = tm.build_features(df)
    ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    c = x["close"].astype(float)

    variants = [
        ("A baseline (current code: bfill + cross-session targets)", dict(horizon=1, session_aware=False, use_bfill=True), False),
        ("B + session-aware 5m target",                              dict(horizon=1, session_aware=True,  use_bfill=True), False),
        ("C + no bfill leak",                                        dict(horizon=1, session_aware=True,  use_bfill=False), False),
        ("D C but binary UP vs DOWN (FLAT dropped)",                 dict(horizon=1, session_aware=True,  use_bfill=False), True),
        ("E horizon 2 bars = 10m, binary",                           dict(horizon=2, session_aware=True,  use_bfill=False), True),
        ("F horizon 6 bars = 30m, binary",                           dict(horizon=6, session_aware=True,  use_bfill=False), True),
        ("G horizon 12 bars = 60m, binary",                          dict(horizon=12, session_aware=True, use_bfill=False), True),
    ]

    rows = []
    for label, kw, binary in variants:
        work, cols = prep(x, ts, c, **kw)
        r = evaluate(work, cols, label, binary=binary)
        rows.append(r)
        print(f"  done: {label}  clf_acc={r['clf_acc']} base={r['baseline']} edge={r['edge']:+.4f} R2={r['R2']}", flush=True)

    print()
    hdr = f"{'variant':<58}{'rows':>7}{'R2':>10}{'sign':>8}{'clf':>8}{'base':>8}{'edge':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<58}{r['rows']:>7}{r['R2']:>10.5f}{r['sign_acc']:>8.4f}"
              f"{r['clf_acc']:>8.4f}{r['baseline']:>8.4f}{r['edge']:>+9.4f}")

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
