
import json, sqlite3, time
from datetime import datetime, date, time as dt_time
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from config import (
    API_BASE, INSTRUMENT_KEY, CANDLE_UNIT, CANDLE_INTERVAL, REQUEST_TIMEOUT, MODEL_DIR, DB_PATH
)
from features import make_features, clean_X
from pattern_features import add_pattern_features, pattern_name_from_row

INTRADAY_URL = f"{API_BASE}/historical-candle/intraday"
QUOTE_URL = f"{API_BASE.replace('/v3','/v2')}/market-quote/quotes"
NEWS_URL = f"{API_BASE.replace('/v3','/v2')}/news"

MODEL_5M = MODEL_DIR / "nifty_v5_5m_points.pkl"
MODEL_10M = MODEL_DIR / "nifty_v5_10m_points.pkl"
MODEL_DIR_MODEL = MODEL_DIR / "nifty_v5_5m_direction.pkl"
FEATURE_FILE = MODEL_DIR / "feature_columns.json"
PATTERN_MODEL = MODEL_DIR / "nifty_v5_pattern_direction.pkl"
PATTERN_STATS_FILE = MODEL_DIR / "nifty_v5_pattern_stats.json"
IST = ZoneInfo("Asia/Kolkata")

_NEWS_CACHE = {"ts": 0.0, "items": [], "sentiment": 0.0}

def _headers(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}

def _fetch_today(token):
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{INTRADAY_URL}/{key}/{CANDLE_UNIT}/{CANDLE_INTERVAL}"
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Upstox intraday API error {r.status_code}: {r.text[:400]}")
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Unexpected Upstox intraday response: {body}")
    rows = []
    for row in body.get("data", {}).get("candles", []):
        if len(row) < 5: continue
        try:
            ts = pd.to_datetime(row[0], errors="coerce")
            o,h,l,c = map(float,row[1:5])
            vol = float(row[5] or 0) if len(row)>5 else 0.0
            oi = float(row[6] or 0) if len(row)>6 else 0.0
            if pd.isna(ts) or not (l<=o<=h and l<=c<=h): continue
            rows.append({"timestamp": ts.isoformat(),"open":o,"high":h,"low":l,"close":c,"volume":vol,"open_interest":oi})
        except (TypeError,ValueError):
            continue
    return pd.DataFrame(rows)

def _load_recent_db(limit=1200):
    if not DB_PATH.exists(): return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        d = pd.read_sql_query(
            "SELECT timestamp,open,high,low,close,volume,open_interest FROM nifty_5min ORDER BY timestamp DESC LIMIT ?",
            conn, params=(int(limit),)
        )
    return d.sort_values("timestamp")

def _merge_frames(*frames):
    fs=[x for x in frames if x is not None and not x.empty]
    if not fs: return pd.DataFrame()
    d=pd.concat(fs,ignore_index=True)
    d["timestamp"]=pd.to_datetime(d["timestamp"],errors="coerce")
    for c in ["open","high","low","close","volume","open_interest"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
    d=d.drop_duplicates("timestamp",keep="last").reset_index(drop=True)
    d=d[(d["open"]>0)&(d["high"]>0)&(d["low"]>0)&(d["close"]>0)]
    d=d[(d["low"]<=d["open"])&(d["open"]<=d["high"])&(d["low"]<=d["close"])&(d["close"]<=d["high"])]
    return d.reset_index(drop=True)

def get_live_frame(token):
    return _merge_frames(_load_recent_db(1200), _fetch_today(token))

def _fetch_full_quote(token):
    key = quote(INSTRUMENT_KEY, safe="")
    url = f"{QUOTE_URL}?instrument_key={key}"
    r = requests.get(url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Upstox quote API error {r.status_code}: {r.text[:300]}")
    body=r.json()
    if body.get("status")!="success":
        raise RuntimeError(f"Unexpected quote response: {body}")
    data=body.get("data",{})
    if not data: return {}
    raw=next(iter(data.values()))
    depth=raw.get("depth",{}) or {}
    buy=depth.get("buy",[]) or []
    sell=depth.get("sell",[]) or []
    bid_qty=sum(float(x.get("quantity",0) or 0) for x in buy[:5])
    ask_qty=sum(float(x.get("quantity",0) or 0) for x in sell[:5])
    total_buy=float(raw.get("total_buy_quantity",0) or 0)
    total_sell=float(raw.get("total_sell_quantity",0) or 0)
    if bid_qty+ask_qty<=0:
        bid_qty,ask_qty=total_buy,total_sell
    imbalance=(bid_qty-ask_qty)/(bid_qty+ask_qty) if bid_qty+ask_qty>0 else 0.0
    return {
        "last_price": float(raw.get("last_price")) if raw.get("last_price") is not None else None,
        "timestamp": raw.get("timestamp") or raw.get("last_trade_time"),
        "volume": float(raw.get("volume",0) or 0),
        "average_price": float(raw.get("average_price",0) or 0),
        "total_buy_quantity": total_buy,
        "total_sell_quantity": total_sell,
        "bid5_quantity": bid_qty,
        "ask5_quantity": ask_qty,
        "depth_imbalance": float(imbalance),
        "depth_supported": bool(bid_qty+ask_qty>0),
    }

def _headline_sentiment(text):
    t=str(text).lower()
    positive=("beat","upgrade","growth","surge","rally","strong","positive","easing","record","inflow","profit","optimism","gain")
    negative=("fall","drop","downgrade","weak","risk","war","inflation","hawkish","outflow","loss","lawsuit","crash","volatility")
    p=sum(1 for w in positive if w in t); n=sum(1 for w in negative if w in t)
    if p+n==0:return 0.0
    return float(np.clip((p-n)/(p+n),-1,1))

def _fetch_news(token):
    global _NEWS_CACHE
    now=time.time()
    if now-_NEWS_CACHE["ts"]<60:
        return _NEWS_CACHE
    key=quote(INSTRUMENT_KEY,safe="")
    url=f"{NEWS_URL}?category=instrument_keys&instrument_keys={key}"
    r=requests.get(url,headers=_headers(token),timeout=REQUEST_TIMEOUT)
    if r.status_code!=200:
        _NEWS_CACHE={"ts":now,"items":[],"sentiment":0.0}
        return _NEWS_CACHE
    try:
        body=r.json()
    except ValueError:
        return _NEWS_CACHE
    items=body.get("data",{}).get("news",[]) if isinstance(body.get("data"),dict) else []
    parsed=[]
    scores=[]
    for x in items[:20]:
        title=x.get("title") or x.get("headline") or x.get("name") or ""
        if not title: continue
        score=_headline_sentiment(title); scores.append(score)
        parsed.append({"title":title,"published_at":x.get("published_at") or x.get("timestamp"),"sentiment":round(score,3)})
    sentiment=float(np.mean(scores)) if scores else 0.0
    _NEWS_CACHE={"ts":now,"items":parsed,"sentiment":sentiment}
    return _NEWS_CACHE

def _technical_context(df):
    c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float); o=df["open"].astype(float)
    ema5=c.ewm(span=5,adjust=False).mean().iloc[-1]
    ema20=c.ewm(span=20,adjust=False).mean().iloc[-1]
    ema50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    slope=np.polyfit(np.arange(min(20,len(c))),c.tail(20),1)[0] if len(c)>=3 else 0.0
    trend="STRONG UP" if ema5>ema20>ema50 and slope>0 else "UP" if ema5>ema20 else "STRONG DOWN" if ema5<ema20<ema50 and slope<0 else "DOWN" if ema5<ema20 else "SIDEWAYS"
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); rsi=float((100-100/(1+rs)).iloc[-1]); rsi=rsi if np.isfinite(rsi) else 50.0
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=float(tr.rolling(14).mean().iloc[-1]); atr=atr if np.isfinite(atr) and atr>0 else float((h-l).tail(20).mean())
    typical=(h+l+c)/3; vol=pd.to_numeric(df["volume"],errors="coerce").fillna(0)
    vwap=float((typical*vol).sum()/vol.sum()) if float(vol.sum())>0 else float(c.iloc[-1])
    support=float(l.tail(30).min()); resistance=float(h.tail(30).max())
    return {"trend":trend,"rsi":rsi,"atr":atr,"vwap":vwap,"support":support,"resistance":resistance,
            "ema5":float(ema5),"ema20":float(ema20),"ema50":float(ema50)}


def _build_training_compatible_features(df):
    """Build exactly the feature columns used by train_model.py V5.1."""
    x,_=make_features(df)
    x,labels=add_pattern_features(x)
    c=pd.to_numeric(x["close"],errors="coerce")
    h=pd.to_numeric(x["high"],errors="coerce")
    l=pd.to_numeric(x["low"],errors="coerce")
    o=pd.to_numeric(x["open"],errors="coerce")
    r=c.pct_change()
    for n in (1,2,3,5,10,20):
        x[f"ret_lag_{n}"]=c.pct_change(n)
        x[f"close_lag_pct_{n}"]=c/c.shift(n)-1.0
    x["high_dist_1"]=h/c-1.0
    x["low_dist_1"]=l/c-1.0
    x["open_dist_1"]=o/c-1.0
    for n in (3,5,10):
        x[f"ret_mean_{n}"]=r.rolling(n).mean()
        x[f"ret_std_{n}"]=r.rolling(n).std()
    ts=pd.to_datetime(x["timestamp"],errors="coerce")
    mins=ts.dt.hour*60+ts.dt.minute
    x["time_sin"]=np.sin(2*np.pi*mins/1440.0)
    x["time_cos"]=np.cos(2*np.pi*mins/1440.0)
    return x, labels

def _pressure(df):
    recent=df.tail(20); close=recent["close"].astype(float); open_=recent["open"].astype(float); vol=recent["volume"].fillna(0).astype(float)
    signed=np.sign(close-open_)
    up=float(vol.where(signed>0,0).sum()); down=float(vol.where(signed<0,0).sum())
    total=up+down
    if total>0:return 100*up/total,100*down/total,"candle-volume proxy"
    body=(close-open_).fillna(0); score=float(body.sum())
    bp=float(np.clip(50+score/max(float(body.abs().sum()),1e-9)*50,0,100))
    return bp,100-bp,"price-action proxy"

def _pattern_context(df):
    x,_=make_features(df)
    x,_=add_pattern_features(x)
    row=x.iloc[-1]
    return pattern_name_from_row(row), row

def _levels(current, atr, support, resistance, p5, direction, confidence):
    risk=max(atr*0.9,abs(p5)*0.75,1.0)
    if direction=="UP":
        sl=min(current-risk, support)
        t1=max(current+abs(p5), current+risk*1.5)
        t2=max(current+risk*2.4, resistance if resistance>current else current+risk*2.4)
    else:
        sl=max(current+risk, resistance)
        t1=min(current-abs(p5), current-risk*1.5)
        t2=min(current-risk*2.4, support if support<current else current-risk*2.4)
    # Keep levels tradable around the current market.
    rr=abs(t1-current)/max(abs(current-sl),1e-9)
    return {"entry":round(current,2),"stop_loss":round(sl,2),"target_1":round(t1,2),"target_2":round(t2,2),
            "risk_reward":round(float(rr),2),"risk_points":round(abs(current-sl),2),
            "reward_points":round(abs(t1-current),2)}

def predict_live(token):
    required=(MODEL_5M,MODEL_10M,MODEL_DIR_MODEL,FEATURE_FILE)
    for p in required:
        if not p.exists(): raise RuntimeError(f"V5 model missing: {p.name}. Train model first.")

    df=get_live_frame(token)
    if len(df)<60: raise RuntimeError(f"Need at least 60 candles. Found {len(df)}.")
    x,labels=_build_training_compatible_features(df)
    cols=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    missing=[c for c in cols if c not in x.columns]
    if missing: raise RuntimeError(f"Live/training feature mismatch: {missing[:10]}")
    row=clean_X(x,cols).iloc[[-1]]
    m5=joblib.load(MODEL_5M); m10=joblib.load(MODEL_10M); clf=joblib.load(MODEL_DIR_MODEL)
    for model_name, model in (("5m",m5),("10m",m10),("direction",clf)):
        expected=int(getattr(model,"n_features_in_",len(cols)))
        if expected!=len(cols):
            raise RuntimeError(f"{model_name} model expects {expected} features but saved schema has {len(cols)}. Retrain V5.1.")
    current=float(df["close"].iloc[-1]); p5=float(m5.predict(row)[0]); p10=float(m10.predict(row)[0])
    probs=clf.predict_proba(row)[0]; classes=list(getattr(clf,"classes_",[]))
    prob_map={int(k):float(v) for k,v in zip(classes,probs)}
    up=prob_map.get(2,prob_map.get(1,0.0)); down=prob_map.get(0,0.0); flat=prob_map.get(1,0.0)
    direction="UP" if up>=max(down,flat) else "DOWN" if down>=flat else "FLAT"
    base_conf=max(up,down,flat)

    pattern=pattern_name_from_row(x.iloc[-1])
    pattern_model=joblib.load(PATTERN_MODEL) if PATTERN_MODEL.exists() else None
    pattern_stats=json.loads(PATTERN_STATS_FILE.read_text(encoding="utf-8")) if PATTERN_STATS_FILE.exists() else {}
    pattern_bias=0.0; pattern_conf=None; pattern_dir="NEUTRAL"; pattern_count=0; pattern_hit=None
    if pattern_model is not None and pattern!="NONE":
        pp=pattern_model.predict_proba(row)[0]; pc=list(getattr(pattern_model,"classes_",range(len(pp))))
        pm={int(k):float(v) for k,v in zip(pc,pp)}
        pu,pn,pd=pm.get(2,0.0),pm.get(1,0.0),pm.get(0,0.0)
        pattern_conf=max(pu,pn,pd); pattern_dir="UP" if pu>=max(pn,pd) else "DOWN" if pd>=pn else "NEUTRAL"
        pattern_bias=pu-pd
        st=pattern_stats.get(pattern,{})
        pattern_count=int(st.get("count",0)); pattern_hit=st.get("hit_rate")

    quote_data=_fetch_full_quote(token)
    news=_fetch_news(token)
    tech=_technical_context(df)
    bp,sp,pressure_basis=_pressure(df)

    depth_imb=float(quote_data.get("depth_imbalance",0.0) or 0.0)
    min_move=max(1.0,tech["atr"]*0.15)
    # Live confirmation layer. These are not pretended to be historical ML features.
    trend_bias=1 if tech["trend"] in ("UP","STRONG UP") else -1 if tech["trend"] in ("DOWN","STRONG DOWN") else 0
    rsi_bias=1 if tech["rsi"]>58 else -1 if tech["rsi"]<42 else 0
    depth_bias=float(np.clip(depth_imb*2,-1,1))
    news_sentiment=float(np.clip(news.get("sentiment",0.0),-1,1))
    news_direction=1 if news_sentiment>0.15 else -1 if news_sentiment<-0.15 else 0
    news_strength=min(abs(news_sentiment),1.0)
    news_bias = news_direction * news_strength * (0.78 if abs(p5) >= max(min_move, tech["atr"]*0.1) else 0.52)
    p5_bias=1 if p5>0 else -1 if p5<0 else 0
    p10_bias=1 if p10>0 else -1 if p10<0 else 0
    # 5M and 10M models are blended as a horizon-consensus to reduce noisy single-horizon calls.
    horizon_consensus = 0.65*p5_bias + 0.35*p10_bias
    confluence=(
        0.38*(up-down) +
        0.16*pattern_bias +
        0.12*trend_bias +
        0.08*depth_bias +
        0.14*news_bias +
        0.08*rsi_bias +
        0.12*horizon_consensus
    )
    final_prob=float(np.clip(0.5 + 0.5*confluence,0.02,0.98))
    final_dir="UP" if final_prob>=0.62 else "DOWN" if final_prob<=0.38 else "FLAT"
    confidence = max(final_prob, 1-final_prob) if final_dir != "FLAT" else 0.5
    signal="WAIT"
    score = (final_prob*100) if final_dir=="UP" else ((1-final_prob)*100) if final_dir=="DOWN" else 50
    score += (trend_bias*8)+(depth_bias*7)+(news_bias*10)+(pattern_bias*8)
    score=float(np.clip(score,0,100))
    market="OPEN" if datetime.now(IST).weekday()<5 and dt_time(9,15)<=datetime.now(IST).time()<=dt_time(15,30) else "CLOSED"
    if market=="OPEN" and abs(p5)>=min_move and confidence>=0.80:
        if final_dir=="UP" and p5>0: signal="BUY"
        elif final_dir=="DOWN" and p5<0: signal="SELL"
    levels=_levels(current,tech["atr"],tech["support"],tech["resistance"],p5,final_dir if final_dir!="FLAT" else ("UP" if p5>=0 else "DOWN"),confidence) if signal!="WAIT" else {"entry":None,"stop_loss":None,"target_1":None,"target_2":None,"risk_reward":None}

    next_price=current+p5
    return {
        "model_version":"V5.1",
        "timestamp":str(df["timestamp"].iloc[-1]),
        "current_price":current,
        "market_status":market,
        "next_5m":{"direction":final_dir,"expected_points":round(p5,2),"expected_price":round(next_price,2),
                   "confidence":round(float(confidence),4),"model_probability":round(float(base_conf),4),
                   "confluence_probability":round(float(final_prob),4)},
        "next_10m":{"direction":"UP" if p10>=0 else "DOWN","expected_points":round(p10,2),"expected_price":round(current+p10,2)},
        "signal":signal,"signal_score":round(score,1),
        "pattern":pattern,"pattern_learning":{"enabled":pattern_model is not None,"direction":pattern_dir,
            "confidence":round(pattern_conf,4) if pattern_conf is not None else None,
            "historical_hit_rate":round(float(pattern_hit),4) if pattern_hit is not None else None,
            "training_samples":pattern_count,"weight":0.15+0.20*min(1,pattern_count/100)},
        "trend":tech["trend"],"rsi":round(tech["rsi"],2),"vwap":round(tech["vwap"],2),
        "support":round(tech["support"],2),"resistance":round(tech["resistance"],2),
        "buyer_pressure":round(bp,1),"seller_pressure":round(sp,1),"pressure_basis":pressure_basis,
        "market_depth":{"available":bool(quote_data.get("depth_supported",False)),
            "buyers":round(float(quote_data.get("bid5_quantity",0)),0),
            "sellers":round(float(quote_data.get("ask5_quantity",0)),0),
            "imbalance":round(depth_imb,4),
            "total_buy_quantity":round(float(quote_data.get("total_buy_quantity",0)),0),
            "total_sell_quantity":round(float(quote_data.get("total_sell_quantity",0)),0)},
        "news":{"sentiment":round(float(news.get("sentiment",0.0)),4),"items":news.get("items",[])[:8],"updated_at":news.get("ts")},
        "levels":levels,"entry":levels["entry"],"stop_loss":levels["stop_loss"],"target_1":levels["target_1"],"target_2":levels["target_2"],
        "reasons":[
            f"ML base direction={direction}, probability={base_conf:.2f}",
            f"Pattern={pattern}, learned bias={pattern_bias:.2f}",
            f"Trend={tech['trend']}, RSI={tech['rsi']:.1f}",
            f"Buyer/Seller pressure={bp:.1f}/{sp:.1f}",
            f"Depth imbalance={depth_imb:+.2f}" if quote_data.get("depth_supported") else "Order-book depth unavailable for this instrument",
            f"News sentiment={news.get('sentiment',0):+.2f}",
        ],
        "note":"Live news/order-book data are used as a confirmation layer. Historical ML features remain causal and do not fabricate historical news/depth. Research/paper-trading only."
    }

def get_chart_history(token, rng="1D"):
    from upstox_api import fetch_range
    days={"1D":1,"5D":5,"1M":30,"3M":90}.get(str(rng).upper(),1)
    rows=fetch_range(token,days)
    if not rows: return {"range":str(rng).upper(),"candles":[],"count":0}
    out=[]
    for r in rows:
        if len(r)<5: continue
        try:
            ts=pd.to_datetime(r[0],errors="coerce")
            if pd.isna(ts): continue
            out.append({"timestamp":ts.isoformat(),"open":float(r[1]),"high":float(r[2]),"low":float(r[3]),"close":float(r[4]),
                        "volume":float(r[5] or 0) if len(r)>5 else 0,"open_interest":float(r[6] or 0) if len(r)>6 else 0})
        except (TypeError,ValueError): pass
    return {"range":str(rng).upper(),"candles":out,"count":len(out)}
