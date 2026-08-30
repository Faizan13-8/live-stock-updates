"""READ-ONLY AUDIT: dense prefix-vs-full causality test. Writes nothing to repo state."""
import sqlite3, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/Faizan/Downloads/NIFTY50_AI_CLEAN_REFACTORED")
from config import DB_PATH
import train_model as tm

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT timestamp, open, high, low, close, volume, open_interest FROM nifty_5min ORDER BY timestamp ASC",
    con,
)
con.close()

N = 8000
sub = df.iloc[:N].reset_index(drop=True)

t0 = time.time()
full = tm.build_features(sub)
print("build_features(8000) took %.2fs" % (time.time() - t0))

numcols = [c for c in full.columns if c not in
           {"timestamp","open","high","low","close","volume","open_interest","pattern_name"}
           and pd.api.types.is_numeric_dtype(full[c])]

# ---- also emulate the post-clean training matrix exactly as _prepare_training_frame does
def cleaned(frame, cols):
    w = frame.replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        w[col] = pd.to_numeric(w[col], errors="coerce").ffill().bfill().fillna(0.0)
    return w

full_clean = cleaned(full, numcols)

ks = list(range(2000, N, 75))   # 80 dense probe rows
print("probing %d rows" % len(ks))

fails_raw, fails_clean = {}, {}
for k in ks:
    pre = tm.build_features(sub.iloc[:k+1].reset_index(drop=True))
    pre_clean = cleaned(pre, numcols)
    for c in numcols:
        a, b = full[c].iloc[k], pre[c].iloc[k]
        if not (pd.isna(a) and pd.isna(b)):
            if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
                fails_raw.setdefault(c, []).append((k, float(b) if not pd.isna(b) else None,
                                                    float(a) if not pd.isna(a) else None))
        a2, b2 = full_clean[c].iloc[k], pre_clean[c].iloc[k]
        if not np.isclose(float(a2), float(b2), rtol=1e-9, atol=1e-12):
            fails_clean.setdefault(c, []).append((k, float(b2), float(a2)))

for title, fails in (("RAW build_features()", fails_raw),
                     ("POST-CLEAN (ffill/bfill) training matrix", fails_clean)):
    print("\n" + "=" * 70)
    print("=== %s : NON-CAUSAL COLUMNS ===" % title)
    if not fails:
        print("(none)")
    for c, rec in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        print("\n%s : %d/%d probe rows CHANGED when future rows were appended" % (c, len(rec), len(ks)))
        for k, b, a in rec[:8]:
            print("    row %5d  causal(prefix 0..k)=%r   ->  with future rows=%r" % (k, b, a))
        if len(rec) > 8:
            print("    ... %d more" % (len(rec) - 8))
    print("\nFAIL LIST [%s]: %s" % (title, sorted(fails.keys())))
