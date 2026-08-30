"""READ-ONLY AUDIT: does the window-dependent quantile change the LIVE model output?
Trains a fresh classifier on current code's columns, then feeds the SAME bar twice:
 (a) features built on the 1200-candle live frame (exactly get_live_frame's DB slice)
 (b) features built on the full frame the model was trained on
and compares predicted direction / confidence.
"""
import sys, sqlite3, collections
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/Faizan/Downloads/NIFTY50_AI_CLEAN_REFACTORED")
from config import DB_PATH
import train_model as tm
import live_prediction as lp
from features import clean_X

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT timestamp,open,high,low,close,volume,open_interest FROM nifty_5min ORDER BY timestamp ASC", con)
con.close()

work, cols = tm._prepare_training_frame(df)
n = len(work); tr = int(n*0.70); va = int(n*0.85)
clf = tm._build_classifier()
clf.fit(work[cols].iloc[:tr], work["direction_5m"].iloc[:tr])
print("classifier trained on %d rows, %d features" % (tr, len(cols)))

# full-frame ("training view") feature rows, exactly as train_model built them
full_rows = work[cols]

flips = 0; total = 0; conf_deltas = []; changed_feats = collections.Counter()
examples = []
for e in range(va+1200, len(df), 53):
    live_df = df.iloc[e-1200:e].reset_index(drop=True)      # == _load_recent_db(1200) slice
    x, _ = lp._build_training_compatible_features(live_df)
    live_row = clean_X(x, cols).iloc[[-1]]
    train_row = full_rows.iloc[[e-1]]
    total += 1
    dl = int(clf.predict(live_row)[0]); dt = int(clf.predict(train_row)[0])
    pl = float(clf.predict_proba(live_row)[0].max()); pt = float(clf.predict_proba(train_row)[0].max())
    conf_deltas.append(abs(pl-pt))
    diff = [c for c in cols
            if not np.isclose(float(live_row[c].iloc[0]), float(train_row[c].iloc[0]), rtol=1e-7, atol=1e-10)]
    for c in diff: changed_feats[c]+=1
    if dl != dt:
        flips += 1
        if len(examples) < 6:
            examples.append((str(df["timestamp"].iloc[e-1]), dl, pl, dt, pt, diff))

name = {0:"DOWN",1:"FLAT",2:"UP"}
print("\nbars evaluated                        : %d" % total)
print("features that differ live vs training : %s" % dict(changed_feats))
print("DIRECTION FLIPS (live vs training view): %d = %.2f%%" % (flips, 100*flips/total))
print("mean |confidence difference|          : %.4f   max: %.4f" % (np.mean(conf_deltas), np.max(conf_deltas)))
print("\nexamples (same candle, two histories):")
for ts, dl, pl, dt, pt, diff in examples:
    print("  %s  live-window -> %-4s conf %.3f   |   training-view -> %-4s conf %.3f   (only diff: %s)"
          % (ts, name[dl], pl, name[dt], pt, diff))
