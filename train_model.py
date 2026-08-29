
import json, sqlite3, gc
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor
from config import DB_PATH, MODEL_DIR
from features import make_features
from pattern_features import add_pattern_features

MODEL_VERSION="V5.1"

def load_data():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT timestamp,open,high,low,close,volume,open_interest FROM nifty_5min ORDER BY timestamp ASC",con)

def build_features(df):
    x,_=make_features(df)
    x,labels=add_pattern_features(x)
    c=x["close"].astype(float); o=x["open"].astype(float); h=x["high"].astype(float); l=x["low"].astype(float)
    r=c.pct_change()
    for n in (1,2,3,5,10,20):
        x[f"ret_lag_{n}"]=c.pct_change(n)
        x[f"close_lag_pct_{n}"]=c/c.shift(n)-1
    x["high_dist_1"]=h/c-1; x["low_dist_1"]=l/c-1; x["open_dist_1"]=o/c-1
    for n in (3,5,10):
        x[f"ret_mean_{n}"]=r.rolling(n).mean(); x[f"ret_std_{n}"]=r.rolling(n).std()
    ts=pd.to_datetime(x["timestamp"],errors="coerce"); mins=ts.dt.hour*60+ts.dt.minute
    x["time_sin"]=np.sin(2*np.pi*mins/1440); x["time_cos"]=np.cos(2*np.pi*mins/1440)
    x["pattern_name"]=labels.astype(str)
    return x

def train_model(progress=None):
    df=load_data()
    if len(df)<2000: raise RuntimeError(f"Not enough candles: {len(df)}")
    if progress: progress(f"Preparing V5.1 features from {len(df):,} candles...",10)
    x=build_features(df); c=x["close"].astype(float)
    x["target_5m"]=c.shift(-1)-c; x["target_10m"]=c.shift(-2)-c
    atr=x["atr_pct"].astype(float)*c
    thr=(0.20*atr).fillna(x["target_5m"].abs().rolling(50).median()).fillna(1.0)
    x["direction_5m"]=np.where(x["target_5m"]>thr,2,np.where(x["target_5m"]<-thr,0,1))
    excluded={"timestamp","open","high","low","close","volume","open_interest","target_5m","target_10m","direction_5m","pattern_name"}
    cols=[c for c in x.columns if c not in excluded and pd.api.types.is_numeric_dtype(x[c])]
    work=x.dropna(subset=["target_5m","target_10m"]).copy()
    work=work.replace([np.inf,-np.inf],np.nan)
    for col in cols: work[col]=pd.to_numeric(work[col],errors="coerce").ffill().bfill().fillna(0)
    n=len(work); tr=int(n*.70); va=int(n*.85)
    if progress: progress("Training 5-minute point model...",30)
    def reg():
        return XGBRegressor(n_estimators=1000,max_depth=5,learning_rate=.025,min_child_weight=8,subsample=.85,colsample_bytree=.85,reg_alpha=.05,reg_lambda=2.0,objective="reg:squarederror",eval_metric="rmse",tree_method="hist",random_state=42,n_jobs=-1)
    m5=reg(); m10=reg()
    m5.fit(work[cols].iloc[:tr],work.target_5m.iloc[:tr],eval_set=[(work[cols].iloc[tr:va],work.target_5m.iloc[tr:va])],verbose=False)
    if progress: progress("Training 10-minute point model...",45)
    m10.fit(work[cols].iloc[:tr],work.target_10m.iloc[:tr],eval_set=[(work[cols].iloc[tr:va],work.target_10m.iloc[tr:va])],verbose=False)
    if progress: progress("Training UP/FLAT/DOWN classifier...",60)
    clf=XGBClassifier(n_estimators=900,max_depth=4,learning_rate=.03,min_child_weight=8,subsample=.85,colsample_bytree=.85,reg_alpha=.05,reg_lambda=2.0,objective="multi:softprob",num_class=3,eval_metric="mlogloss",tree_method="hist",random_state=42,n_jobs=-1)
    clf.fit(work[cols].iloc[:tr],work.direction_5m.iloc[:tr],eval_set=[(work[cols].iloc[tr:va],work.direction_5m.iloc[tr:va])],verbose=False)
    if progress: progress("Training learned pattern model...",75)
    pmask=work.pattern_code.astype(int)>0 if "pattern_code" in work else pd.Series(False,index=work.index)
    pw=work.loc[pmask]
    pmodel=None
    if len(pw)>=300 and pw.direction_5m.nunique()>=3:
        pmodel=XGBClassifier(n_estimators=600,max_depth=4,learning_rate=.03,min_child_weight=8,subsample=.85,colsample_bytree=.85,reg_alpha=.05,reg_lambda=2.0,objective="multi:softprob",num_class=3,eval_metric="mlogloss",tree_method="hist",random_state=42,n_jobs=-1)
        cut=int(len(pw)*.85)
        pmodel.fit(pw[cols].iloc[:cut],pw.direction_5m.iloc[:cut],eval_set=[(pw[cols].iloc[cut:],pw.direction_5m.iloc[cut:])],verbose=False)
    # Test metrics
    y5=work.target_5m.iloc[va:]; p5=m5.predict(work[cols].iloc[va:])
    yd=work.direction_5m.iloc[va:]; pd_=clf.predict(work[cols].iloc[va:])
    meta={
        "model_version":MODEL_VERSION,
        "trained_at":pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "rows_total":int(len(df)),"rows_used":int(len(work)),"feature_count":int(len(cols)),
        "historical_start":str(df.timestamp.min()),"historical_end":str(df.timestamp.max()),
        "train_rows":tr,"validation_rows":va-tr,"test_rows":n-va,
        "metrics":{
            "5m":{"MAE":float(mean_absolute_error(y5,p5)),"RMSE":float(np.sqrt(mean_squared_error(y5,p5))),"R2":float(r2_score(y5,p5)),
                  "directional_accuracy":float(np.mean(np.sign(y5)==np.sign(p5)))},
            "direction":{"Accuracy":float(accuracy_score(yd,pd_)),"F1":float(f1_score(yd,pd_,average="macro",zero_division=0)),
                         "Precision":float(precision_score(yd,pd_,average="macro",zero_division=0)),"Recall":float(recall_score(yd,pd_,average="macro",zero_division=0))},
            "pattern":{"enabled":pmodel is not None,"samples":int(len(pw))}
        }
    }
    # Train-only pattern hit-rate priors
    stats={}
    if len(pw):
        trainp=pw.iloc[:max(1,int(len(pw)*.70))]
        for code,grp in trainp.groupby("pattern_code"):
            name=str(grp["pattern_name"].iloc[0]); vals=grp.direction_5m.astype(int)
            implied=2 if name in {"HAMMER","BULLISH_ENGULFING","BREAKOUT_UP","HIGHER_HIGH_LOWER_LOW"} else 0 if name in {"SHOOTING_STAR","BEARISH_ENGULFING","BREAKOUT_DOWN","LOWER_HIGH_LOWER_LOW"} else 1
            stats[name]={"count":int(len(grp)),"hit_rate":float((vals==implied).mean()),"implied_direction":int(implied)}
    joblib.dump(m5,MODEL_DIR/"nifty_v5_5m_points.pkl"); joblib.dump(m10,MODEL_DIR/"nifty_v5_10m_points.pkl"); joblib.dump(clf,MODEL_DIR/"nifty_v5_5m_direction.pkl")
    if pmodel is not None: joblib.dump(pmodel,MODEL_DIR/"nifty_v5_pattern_direction.pkl")
    elif (MODEL_DIR/"nifty_v5_pattern_direction.pkl").exists(): (MODEL_DIR/"nifty_v5_pattern_direction.pkl").unlink()
    (MODEL_DIR/"feature_columns.json").write_text(json.dumps(cols,indent=2),encoding="utf-8")
    (MODEL_DIR/"nifty_v5_pattern_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    (MODEL_DIR/"model_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    if progress: progress("MODEL V5.1 READY — ML + pattern learning + live confirmation layer",100)
    gc.collect()
    return meta

if __name__=="__main__":
    print(json.dumps(train_model(lambda m,p:print(f"[{p}%] {m}")),indent=2))
