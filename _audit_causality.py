"""READ-ONLY AUDIT SCRIPT - prefix vs full causality test. Does not modify repo files."""
import sqlite3, sys, json
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
print("rows loaded:", len(df))

N = 6000
sub = df.iloc[:N].reset_index(drop=True)

full = tm.build_features(sub)
numcols = [c for c in full.columns if c not in
           {"timestamp","open","high","low","close","volume","open_interest","pattern_name"}
           and pd.api.types.is_numeric_dtype(full[c])]
print("feature cols tested:", len(numcols))

ks = [1500, 2500, 3500, 4500, 5500]
fails = {}
for k in ks:
    pre = tm.build_features(sub.iloc[:k+1].reset_index(drop=True))
    for c in numcols:
        a = full[c].iloc[k]
        b = pre[c].iloc[k]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
            fails.setdefault(c, []).append((k, float(b) if not pd.isna(b) else None,
                                            float(a) if not pd.isna(a) else None))

print("\n=== RAW build_features(): NON-CAUSAL COLUMNS ===")
if not fails:
    print("(none)")
for c, rec in sorted(fails.items()):
    print(f"\n{c}:  {len(rec)}/{len(ks)} probe rows changed")
    for k, b, a in rec:
        print(f"   row {k}: prefix-only={b!r}   full-data={a!r}")
print("\nSUMMARY raw fail list:", sorted(fails.keys()))
