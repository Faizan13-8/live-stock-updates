import numpy as np
import pandas as pd

FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_20",
    "body_pct","upper_wick_pct","lower_wick_pct","range_pct",
    "ema5_dist","ema9_dist","ema20_dist","ema50_dist",
    "sma20_dist","sma50_dist","rsi14","rsi_delta",
    "macd","macd_signal","macd_hist","atr_pct",
    "vol_ratio","close_pos_20","high_breakout_20","low_breakdown_20",
    "time_sin","time_cos",
]

def make_features(df):
    x=df.copy()
    for c in ["open","high","low","close","volume","open_interest"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    c,o,h,l=x["close"],x["open"],x["high"],x["low"]

    r=c.pct_change()
    for n in [1,3,5,10,20]:
        x[f"ret_{n}"]=c.pct_change(n)

    rng=(h-l).replace(0,np.nan)
    x["body_pct"]=(c-o)/c
    x["upper_wick_pct"]=(h-np.maximum(o,c))/c
    x["lower_wick_pct"]=(np.minimum(o,c)-l)/c
    x["range_pct"]=rng/c

    for n in [5,9,20,50]:
        ema=c.ewm(span=n,adjust=False).mean()
        x[f"ema{n}_dist"]=c/ema-1
    for n in [20,50]:
        sma=c.rolling(n).mean()
        x[f"sma{n}_dist"]=c/sma-1

    delta=c.diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi14"]=100-(100/(1+rs))
    x["rsi_delta"]=x["rsi14"].diff()

    ema12=c.ewm(span=12,adjust=False).mean()
    ema26=c.ewm(span=26,adjust=False).mean()
    x["macd"]=ema12-ema26
    x["macd_signal"]=x["macd"].ewm(span=9,adjust=False).mean()
    x["macd_hist"]=x["macd"]-x["macd_signal"]

    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["atr_pct"]=tr.rolling(14).mean()/c

    vm=x["volume"].rolling(20).mean().replace(0,np.nan)
    x["vol_ratio"]=x["volume"]/vm

    rh=h.shift(1).rolling(20,min_periods=5).max()
    rl=l.shift(1).rolling(20,min_periods=5).min()
    x["close_pos_20"]=(c-rl)/(rh-rl).replace(0,np.nan)
    x["high_breakout_20"]=(c>rh).astype(int)
    x["low_breakdown_20"]=(c<rl).astype(int)

    ts=pd.to_datetime(x["timestamp"],errors="coerce")
    minute=ts.dt.hour*60+ts.dt.minute
    x["time_sin"]=np.sin(2*np.pi*minute/1440)
    x["time_cos"]=np.cos(2*np.pi*minute/1440)

    x=x.replace([np.inf,-np.inf],np.nan)
    return x, FEATURES.copy()

def clean_X(x, cols):
    y=x[cols].apply(pd.to_numeric,errors="coerce")
    y=y.replace([np.inf,-np.inf],np.nan)
    y=y.ffill().bfill().fillna(0.0)
    return y
