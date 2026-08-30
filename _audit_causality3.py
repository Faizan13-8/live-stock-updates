"""READ-ONLY AUDIT: full-dataset prefix-vs-full causality test + bfill row count."""
import sqlite3, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/Faizan/Downloads/NIFTY50_AI_CLEAN_REFACTORED")
from config import DB_PATH
import train_model as tm
import live_prediction as lp
from features import clean_X

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT timestamp, open, high, low, close, volume, open_interest FROM nifty_5min ORDER BY timestamp ASC",
    con,
)
con.close()
print("FULL dataset rows:", len(df))

full = tm.build_features(df)
numcols = [c for c in full.columns if c not in
           {"timestamp","open","high","low","close","volume","open_interest","pattern_name"}
           and pd.api.types.is_numeric_dtype(full[c])]
print("columns under test:", len(numcols))

# ---------- bfill blast radius in the training matrix ----------
raw = full[numcols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
ff  = raw.ffill()
ffbf = ff.bfill()
changed = (~ff.isna()) & False
mask = ff.isna() & (~ffbf.isna())          # cells filled by BACKWARD fill = future value
print("\n--- bfill (future-fill) blast radius in training matrix ---")
print("cells filled from a FUTURE row :", int(mask.to_numpy().sum()))
rows_hit = mask.any(axis=1)
print("rows touched                   :", int(rows_hit.sum()), "of", len(raw))
if rows_hit.any():
    idxs = np.flatnonzero(rows_hit.to_numpy())
    print("row index range touched        :", idxs.min(), "..", idxs.max())
    print("columns touched                :", sorted(mask.columns[mask.any(axis=0)].tolist()))

# ---------- dense prefix-vs-full over the WHOLE real dataset ----------
ks = list(range(1000, len(df), 1400))
print("\nprobing %d rows across the full %d-row DB ..." % (len(ks), len(df)))
fails = {}
t0 = time.time()
for k in ks:
    pre = tm.build_features(df.iloc[:k+1].reset_index(drop=True))
    for c in numcols:
        a, b = full[c].iloc[k], pre[c].iloc[k]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
            fails.setdefault(c, []).append((k, None if pd.isna(b) else float(b),
                                                None if pd.isna(a) else float(a)))
print("elapsed %.1fs" % (time.time()-t0))
print("\n==== NON-CAUSAL COLUMNS OVER FULL REAL DB ====")
if not fails:
    print("(none)")
for c, rec in sorted(fails.items(), key=lambda kv: -len(kv[1])):
    print("\n%s : %d/%d probes CHANGED" % (c, len(rec), len(ks)))
    for k, b, a in rec[:10]:
        print("    row %6d  ts=%s  causal=%r  ->  with-future=%r"
              % (k, df["timestamp"].iloc[k], b, a))
print("\nFAIL LIST:", sorted(fails.keys()))
