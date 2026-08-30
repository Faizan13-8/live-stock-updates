"""READ-ONLY AUDIT: quantify the money-facing effect of the window-dependent volatility_bucket
on the 5m POINT regressor (p5 drives stop_loss / target_1 / target_2 in live_prediction._levels)."""
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
m5 = tm._build_regressor(); m5.fit(work[cols].iloc[:tr], work["target_5m"].iloc[:tr])
clf = tm._build_classifier(); clf.fit(work[cols].iloc[:tr], work["direction_5m"].iloc[:tr])
print("m5 regressor + direction clf trained on %d rows / %d features" % (tr, len(cols)))
full_rows = work[cols]

recs = []
for e in range(va+1200, len(df), 11):     # dense sample
    live_df = df.iloc[e-1200:e].reset_index(drop=True)
    x, _ = lp._build_training_compatible_features(live_df)
    lr = clean_X(x, cols).iloc[[-1]]
    trr = full_rows.iloc[[e-1]]
    vb_l = float(lr["volatility_bucket"].iloc[0]); vb_t = float(trr["volatility_bucket"].iloc[0])
    p5l = float(m5.predict(lr)[0]); p5t = float(m5.predict(trr)[0])
    dl = int(clf.predict(lr)[0]); dt = int(clf.predict(trr)[0])
    recs.append((str(df["timestamp"].iloc[e-1]), vb_l, vb_t, p5l, p5t, dl, dt))

a = pd.DataFrame(recs, columns=["ts","vb_live","vb_train","p5_live","p5_train","dir_live","dir_train"])
skew = a[a.vb_live != a.vb_train]
print("\nbars evaluated                : %d" % len(a))
print("volatility_bucket differs     : %d = %.2f%%" % (len(skew), 100*len(skew)/len(a)))
print("direction argmax flips        : %d = %.2f%%" % ((a.dir_live!=a.dir_train).sum(), 100*(a.dir_live!=a.dir_train).mean()))
sgn = (np.sign(a.p5_live)!=np.sign(a.p5_train))
print("p5 SIGN flips (long vs short) : %d = %.2f%%" % (sgn.sum(), 100*sgn.mean()))
d = (a.p5_live - a.p5_train).abs()
print("|p5_live - p5_train| median=%.3f  p90=%.3f  max=%.3f NIFTY points" % (d.median(), d.quantile(.9), d.max()))
if len(skew):
    ds = (skew.p5_live - skew.p5_train).abs()
    print("  on skewed bars only: median=%.3f p90=%.3f max=%.3f" % (ds.median(), ds.quantile(.9), ds.max()))
    sg = (np.sign(skew.p5_live)!=np.sign(skew.p5_train))
    print("  on skewed bars only: p5 sign flips %d/%d = %.2f%%" % (sg.sum(), len(skew), 100*sg.mean()))
    print("\n  worst 8 skewed bars:")
    for _, r in skew.assign(d=ds).sort_values("d", ascending=False).head(8).iterrows():
        print("   %s  vb live=%+.0f train=%+.0f   p5 live=%+.2f  train=%+.2f  (delta %.2f pts)"
              % (r.ts, r.vb_live, r.vb_train, r.p5_live, r.p5_train, abs(r.p5_live-r.p5_train)))
