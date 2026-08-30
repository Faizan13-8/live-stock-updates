"""READ-ONLY AUDIT: A/B impact of the volatility_bucket look-ahead on trained-model behaviour."""
import sys, time, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/Faizan/Downloads/NIFTY50_AI_CLEAN_REFACTORED")
from config import DB_PATH
import train_model as tm
from sklearn.metrics import accuracy_score

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT timestamp,open,high,low,close,volume,open_interest FROM nifty_5min ORDER BY timestamp ASC", con)
con.close()

work, cols = tm._prepare_training_frame(df)
n = len(work); tr = int(n*0.70); va = int(n*0.85)
print("rows=%d train=%d val=%d test=%d  features=%d" % (n, tr, va-tr, n-va, len(cols)))

# causal replacement: expanding quantile
rv = work["regime_volatility"].astype(float)
q75 = rv.expanding(min_periods=200).quantile(0.75)
q25 = rv.expanding(min_periods=200).quantile(0.25)
causal_vb = pd.Series(np.where(rv > q75, 1.0, np.where(rv < q25, -1.0, 0.0)), index=work.index)

variants = {
    "A_leaky_as_shipped": work[cols].copy(),
}
b = work[cols].copy(); b["volatility_bucket"] = causal_vb.to_numpy()
variants["B_causal_expanding_q"] = b

for name, X in variants.items():
    t0 = time.time()
    clf = tm._build_classifier()
    clf.fit(X.iloc[:tr], work["direction_5m"].iloc[:tr])
    pred_test = clf.predict(X.iloc[va:])
    acc = accuracy_score(work["direction_5m"].iloc[va:], pred_test)
    imp = dict(zip(cols, clf.feature_importances_))
    rank = sorted(imp.items(), key=lambda kv: -kv[1])
    vb_rank = [i for i,(k,_) in enumerate(rank) if k=="volatility_bucket"][0]+1
    print("\n[%s] fit %.0fs" % (name, time.time()-t0))
    print("  test accuracy (iloc[va:]) = %.6f" % acc)
    print("  volatility_bucket gain-importance = %.6f  rank %d / %d" % (imp["volatility_bucket"], vb_rank, len(cols)))
    print("  vol_ratio importance = %.6f (dead-feature check)" % imp["vol_ratio"])
    print("  top 8:", [(k, round(v,4)) for k,v in rank[:8]])

# how many TEST-set rows have a different volatility_bucket under the causal definition?
d = (work["volatility_bucket"].to_numpy() != causal_vb.to_numpy())
print("\nrows whose volatility_bucket is future-dependent : %d / %d = %.2f%%" % (d.sum(), n, 100*d.mean()))
print("  within TRAIN slice iloc[:%d]      : %d = %.2f%%" % (tr, d[:tr].sum(), 100*d[:tr].mean()))
print("  within TEST  slice iloc[%d:]      : %d = %.2f%%" % (va, d[va:].sum(), 100*d[va:].mean()))
