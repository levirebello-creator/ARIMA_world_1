# ─────────────────────────────────────────────────────────────────────────────
#  App 1 · ARIMA Forecaster  |  IndiaForecast Suite  |  Blue #38bdf8
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, io, warnings
from statsmodels.tsa.arima.model import ARIMA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Forecaster · IndiaForecast",
                   page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ACCENT="#38bdf8"; BG="#080d18"; CARD="#0d1424"; BORDER="#1a2640"; GRID="#111a2c"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.stApp{{background:{BG};color:#e2e8f0;}}
[data-testid="stSidebar"]{{background:{CARD};border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{{color:#8896b0!important;font-size:.82rem!important;}}
[data-testid="metric-container"]{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:.8rem 1rem;}}
[data-testid="metric-container"] label{{color:#4a6080!important;font-size:.65rem!important;letter-spacing:.08em;text-transform:uppercase;}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{color:#f0f6ff!important;font-size:1.15rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{{font-size:.72rem!important;font-family:'JetBrains Mono',monospace!important;}}
.hero{{background:linear-gradient(135deg,#0a1930 0%,#071224 60%,#0a1930 100%);border:1px solid #1a3060;border-radius:14px;padding:1.6rem 2rem;margin-bottom:1.4rem;}}
.hero-ticker{{font-size:1.8rem;font-weight:700;color:{ACCENT};letter-spacing:-0.02em;margin:0;}}
.hero-name{{font-size:.9rem;color:#94a3b8;margin-top:.2rem;}}
.hero-meta{{font-size:.7rem;color:#3d5070;margin-top:.3rem;font-family:'JetBrains Mono',monospace;}}
.badge{{display:inline-block;background:rgba(56,189,248,.1);color:{ACCENT};border:1px solid rgba(56,189,248,.25);border-radius:20px;padding:.12rem .6rem;font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-right:.35rem;}}
.sec-label{{color:{ACCENT};font-size:.65rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid {BORDER};padding-bottom:.35rem;margin-bottom:.8rem;}}
.fc-row{{display:flex;justify-content:space-between;align-items:center;padding:.45rem .8rem;border-radius:7px;margin-bottom:.2rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;}}
.fc-row:nth-child(odd){{background:{CARD};}} .fc-row:nth-child(even){{background:#0a101c;}}
.fc-dt{{color:#6b84a0;}} .fc-px{{color:#e2e8f0;font-weight:600;}}
.fc-up{{color:#34d399;}} .fc-dn{{color:#f87171;}}
.info-box{{background:#0a1624;border-left:3px solid {ACCENT};border-radius:0 8px 8px 0;padding:.75rem .9rem;font-size:.76rem;color:#6b84a0;line-height:1.75;margin:.5rem 0;}}
.fund-card{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:.9rem 1.1rem;}}
.fund-row{{display:flex;justify-content:space-between;padding:.28rem 0;border-bottom:1px solid {BORDER};font-size:.78rem;}}
.fund-row:last-child{{border-bottom:none;}}
.fund-key{{color:#4a6080;}} .fund-val{{color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-weight:600;}}
.trade-card{{border-radius:12px;padding:1rem 1.2rem;margin:.4rem 0;}}
.trade-entry{{background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.25);}}
.trade-sl{{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.25);}}
.trade-t1{{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.25);}}
.trade-t2{{background:rgba(168,85,247,.07);border:1px solid rgba(168,85,247,.25);}}
.sent-box{{padding:.65rem .9rem;border-radius:8px;font-size:.76rem;margin-top:.4rem;}}
.search-result{{padding:.4rem .8rem;background:{CARD};border:1px solid {BORDER};border-radius:6px;margin:.2rem 0;cursor:pointer;font-size:.8rem;}}
#MainMenu,footer,header{{visibility:hidden;}}
hr{{border-color:{BORDER};margin:1rem 0;}}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EXCHANGES = {
    "🇮🇳 NSE (India)":".NS","🇮🇳 BSE (India)":".BO",
    "🇺🇸 NYSE / NASDAQ":"","🇬🇧 LSE (UK)":".L",
    "🇩🇪 Xetra (Germany)":".DE","🇫🇷 Euronext (France)":".PA",
    "🇯🇵 TSE (Japan)":".T","🇭🇰 HKEX (Hong Kong)":".HK",
    "🇨🇳 Shanghai":".SS","🇦🇺 ASX (Australia)":".AX",
    "🇨🇦 TSX (Canada)":".TO","🇸🇬 SGX (Singapore)":".SI",
    "🇰🇷 KRX (South Korea)":".KS","🇧🇷 B3 (Brazil)":".SA",
}
INDEXES = {
    "NIFTY 50":"^NSEI","NIFTY BANK":"^NSEBANK","NIFTY IT":"^CNXIT",
    "NIFTY PHARMA":"^CNXPHARMA","NIFTY AUTO":"^CNXAUTO","NIFTY FMCG":"^CNXFMCG",
    "NIFTY MIDCAP100":"^NSEMDCP100","NIFTY SMALLCAP100":"^CNXSC",
    "SENSEX":"^BSESN","INDIA VIX":"^INDIAVIX",
    "S&P 500":"^GSPC","NASDAQ 100":"^NDX","DOW JONES":"^DJI",
    "FTSE 100":"^FTSE","DAX":"^GDAXI","NIKKEI 225":"^N225",
    "HANG SENG":"^HSI","ASX 200":"^AXJO",
    "Gold":"GC=F","Crude Oil (WTI)":"CL=F","Bitcoin USD":"BTC-USD",
}
TIMELINES={"1 Month":21,"3 Months":63,"6 Months":126,"1 Year":252,"2 Years":504,"5 Years":1260}

# Comprehensive name→symbol map
NAME_MAP = {
    "reliance industries":"RELIANCE.NS","reliance":"RELIANCE.NS",
    "tata consultancy services":"TCS.NS","tcs":"TCS.NS",
    "infosys":"INFY.NS","infy":"INFY.NS",
    "hdfc bank":"HDFCBANK.NS","hdfcbank":"HDFCBANK.NS",
    "icici bank":"ICICIBANK.NS","icicibank":"ICICIBANK.NS",
    "hindustan unilever":"HINDUNILVR.NS","hul":"HINDUNILVR.NS",
    "state bank of india":"SBIN.NS","sbi":"SBIN.NS",
    "bajaj finance":"BAJFINANCE.NS",
    "bharti airtel":"BHARTIARTL.NS","airtel":"BHARTIARTL.NS",
    "kotak mahindra bank":"KOTAKBANK.NS","kotak":"KOTAKBANK.NS",
    "larsen toubro":"LT.NS","l&t":"LT.NS","lt":"LT.NS",
    "hcl technologies":"HCLTECH.NS","hcltech":"HCLTECH.NS",
    "axis bank":"AXISBANK.NS","wipro":"WIPRO.NS",
    "asian paints":"ASIANPAINT.NS",
    "maruti suzuki":"MARUTI.NS","maruti":"MARUTI.NS",
    "titan":"TITAN.NS","titan company":"TITAN.NS",
    "sun pharma":"SUNPHARMA.NS","sun pharmaceutical":"SUNPHARMA.NS",
    "ultratech cement":"ULTRACEMCO.NS","ongc":"ONGC.NS",
    "nestle india":"NESTLEIND.NS","nestle":"NESTLEIND.NS",
    "power grid":"POWERGRID.NS","ntpc":"NTPC.NS",
    "adani enterprises":"ADANIENT.NS","adani ports":"ADANIPORTS.NS",
    "tata motors":"TATAMOTORS.NS","tata steel":"TATASTEEL.NS",
    "tech mahindra":"TECHM.NS","hindalco":"HINDALCO.NS",
    "jsw steel":"JSWSTEEL.NS","grasim":"GRASIM.NS",
    "coal india":"COALINDIA.NS","divis labs":"DIVISLAB.NS",
    "cipla":"CIPLA.NS","dr reddy":"DRREDDY.NS","dr. reddy":"DRREDDY.NS",
    "eicher motors":"EICHERMOT.NS","hero motocorp":"HEROMOTOCO.NS","hero":"HEROMOTOCO.NS",
    "bpcl":"BPCL.NS","sbi life":"SBILIFE.NS","hdfc life":"HDFCLIFE.NS",
    "indusind bank":"INDUSINDBK.NS","mahindra":"M&M.NS","m&m":"M&M.NS",
    "bajaj finserv":"BAJAJFINSV.NS","apollo hospitals":"APOLLOHOSP.NS",
    "britannia":"BRITANNIA.NS","tata consumer":"TATACONSUM.NS",
    "itc":"ITC.NS","bel":"BEL.NS","bharat electronics":"BEL.NS",
    "hal":"HAL.NS","hindustan aeronautics":"HAL.NS","irfc":"IRFC.NS",
    "zomato":"ZOMATO.NS","paytm":"PAYTM.NS","nykaa":"NYKAA.NS",
    "dmart":"DMART.NS","avenue supermarts":"DMART.NS","pidilite":"PIDILITIND.NS",
    "havells":"HAVELLS.NS","trent":"TRENT.NS","vedanta":"VEDL.NS",
    "pfc":"PFC.NS","power finance":"PFC.NS","rec":"RECLTD.NS",
    "shriram finance":"SHRIRAMFIN.NS",
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","tesla":"TSLA","nvidia":"NVDA","meta":"META",
    "jpmorgan":"JPM","netflix":"NFLX","visa":"V","mastercard":"MA",
    "samsung":"005930.KS","toyota":"7203.T","alibaba":"9988.HK",
    "tencent":"0700.HK","hsbc":"HSBA.L","bp":"BP.L","shell":"SHEL.L",
    "volkswagen":"VOW3.DE","siemens":"SIE.DE","lvmh":"MC.PA",
}

# Exchange suffix detector
SUFFIX_EXCHANGE_MAP = {
    ".NS":"🇮🇳 NSE (India)",".BO":"🇮🇳 BSE (India)",
    ".L":"🇬🇧 LSE (UK)",".DE":"🇩🇪 Xetra (Germany)",
    ".PA":"🇫🇷 Euronext (France)",".T":"🇯🇵 TSE (Japan)",
    ".HK":"🇭🇰 HKEX (Hong Kong)",".SS":"🇨🇳 Shanghai",
    ".AX":"🇦🇺 ASX (Australia)",".TO":"🇨🇦 TSX (Canada)",
    ".SI":"🇸🇬 SGX (Singapore)",".KS":"🇰🇷 KRX (South Korea)",".SA":"🇧🇷 B3 (Brazil)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_symbol(raw_input, exchange_suffix):
    """
    Smart resolver: tries Yahoo Finance search API first → falls back to local
    name map → falls back to symbol+suffix. Returns (ticker_symbol, detected_exchange_label)
    """
    raw = raw_input.strip()
    low = raw.lower()

    # 1. Try Yahoo Finance search API first (works on Streamlit Cloud)
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": raw, "quotesCount": 5, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            quotes = r.json().get("quotes", [])
            if quotes:
                best = quotes[0]
                sym  = best.get("symbol","")
                if sym:
                    for sfx, exch in SUFFIX_EXCHANGE_MAP.items():
                        if sym.endswith(sfx): return sym, exch
                    return sym, "🇺🇸 NYSE / NASDAQ"
    except: pass

    # 2. Fall back to local name map
    if low in NAME_MAP:
        sym = NAME_MAP[low]
        for sfx, exch in SUFFIX_EXCHANGE_MAP.items():
            if sym.endswith(sfx): return sym, exch
        return sym, "🇺🇸 NYSE / NASDAQ"

    # 3. Treat as direct symbol + chosen exchange suffix
    sym = raw.upper() + exchange_suffix
    for sfx, exch in SUFFIX_EXCHANGE_MAP.items():
        if sym.endswith(sfx): return sym, exch
    return sym, "🇺🇸 NYSE / NASDAQ"

def search_suggestions(query):
    """Return name-map suggestions for partial match"""
    q = query.lower().strip()
    if len(q) < 2: return []
    return [(k.title(), v) for k, v in NAME_MAP.items() if q in k][:6]

@st.cache_data(ttl=86400,show_spinner=False)
def fetch_nse_list():
    try:
        s=requests.Session()
        s.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://www.nseindia.com/"})
        s.get("https://www.nseindia.com",timeout=8)
        r=s.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",timeout=15)
        if r.status_code!=200: return {}
        df=pd.read_csv(io.BytesIO(r.content)); df.columns=df.columns.str.strip()
        sc=[c for c in df.columns if "SYMBOL" in c.upper()][0]
        nc=[c for c in df.columns if "NAME"   in c.upper()][0]
        src=[c for c in df.columns if "SERIES" in c.upper()][0]
        eq=df[df[src].str.strip().isin(["EQ","N"])]
        return {f"{row[sc].strip()} — {str(row[nc]).strip().title()}":f"{row[sc].strip()}.NS"
                for _,row in eq.iterrows() if str(row[sc]).strip()}
    except: return {}

@st.cache_data(ttl=3600,show_spinner=False)
def load_price(sym):
    df=yf.download(sym,start=datetime.today()-timedelta(days=5*365+30),
                   end=datetime.today(),progress=False,auto_adjust=True)
    if df.empty: return pd.Series(dtype=float)
    return df["Close"].squeeze().dropna()

@st.cache_data(ttl=3600,show_spinner=False)
def load_info(sym):
    try: return yf.Ticker(sym).info
    except: return {}

@st.cache_data(ttl=1800,show_spinner=False)
def load_news(sym):
    try:
        t=yf.Ticker(sym); return t.news[:10] if t.news else []
    except: return []

def fmt_price(v):
    try:
        v=float(v)
        if v>=1_00_00_000: return f"₹{v/1_00_00_000:.1f}Cr"
        if v>=1_00_000:    return f"₹{v/1_00_000:.1f}L"
        if v>=10_000:      return f"₹{v:,.0f}"
        if v>=1_000:       return f"₹{v:,.1f}"
        return f"₹{v:,.2f}"
    except: return "N/A"

def fmt_chg(chg,pct):
    sym="▲" if chg>=0 else "▼"
    return f"{sym} {abs(chg):.1f} ({pct:+.1f}%)"

def fmt_large(v):
    try:
        v=float(v)
        if v>=1e12: return f"${v/1e12:.2f}T"
        if v>=1e9:  return f"${v/1e9:.2f}B"
        if v>=1e6:  return f"${v/1e6:.2f}M"
        return f"{v:,.2f}"
    except: return "N/A"

def calc_indicators(close):
    d=close.diff()
    gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan))
    e12=close.ewm(span=12).mean(); e26=close.ewm(span=26).mean()
    macd=e12-e26; sig=macd.ewm(span=9).mean(); hist=macd-sig
    bb=close.rolling(20).mean(); bbs=close.rolling(20).std()
    vol=close.pct_change().rolling(30).std()*np.sqrt(252)*100
    return rsi,macd,sig,hist,bb,bb+2*bbs,bb-2*bbs,vol

def trading_levels(close, fc_s):
    hi=close.rolling(14).max(); lo=close.rolling(14).min()
    atr=float((hi-lo).rolling(14).mean().iloc[-1])
    cur=float(close.iloc[-1])
    d=close.diff()
    gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    rsi_now=float((100-100/(1+gain/loss.replace(0,np.nan))).dropna().iloc[-1])
    sma20=float(close.rolling(20).mean().iloc[-1])
    if rsi_now<65:
        entry=cur; note="Current market price (RSI neutral — good entry)"
    elif rsi_now<75:
        entry=round(sma20*1.001,2); note=f"Near SMA20 ₹{sma20:,.2f} (RSI elevated — wait for minor dip)"
    else:
        entry=round(sma20*0.99,2); note=f"Below SMA20 (RSI overbought — wait for pullback)"
    sl=round(entry-1.5*atr,2); risk=entry-sl
    t1=round(entry+1.5*risk,2); t2=round(float(fc_s.iloc[-1]),2); t3=round(entry+3*risk,2)
    return {"entry":entry,"note":note,"sl":sl,"sl_pct":(sl-entry)/entry*100,
            "t1":t1,"t1_pct":(t1-entry)/entry*100,
            "t2":t2,"t2_pct":(t2-entry)/entry*100,
            "t3":t3,"t3_pct":(t3-entry)/entry*100,
            "rr":round(abs((t1-entry)/risk),2),"rsi":rsi_now,"atr":atr}

def arima_price_forecast(close,p,q,fc_days):
    log_ret=np.log(close/close.shift(1)).dropna()
    model=ARIMA(log_ret,order=(p,0,q)).fit()
    fc_res=model.get_forecast(steps=fc_days)
    fc_lr=fc_res.predicted_mean.values
    ci=fc_res.conf_int()
    last=float(close.iloc[-1])
    fc_px=last*np.exp(np.cumsum(fc_lr))
    ci_lo=last*np.exp(np.cumsum(ci.iloc[:,0].values))
    ci_hi=last*np.exp(np.cumsum(ci.iloc[:,1].values))
    band=last*0.40
    ci_lo=np.maximum(ci_lo,fc_px-band); ci_hi=np.minimum(ci_hi,fc_px+band)
    fut=pd.bdate_range(start=close.index[-1],periods=fc_days+1)[1:]
    n=min(len(fut),len(fc_px))
    return (pd.Series(fc_px[:n],index=fut[:n]),
            pd.Series(ci_lo[:n],index=fut[:n]),
            pd.Series(ci_hi[:n],index=fut[:n]),model)

def confidence_label(model,ci_lo,ci_hi,cur):
    """Translate AIC/BIC fit quality + forecast band width into plain English."""
    try:
        band_pct=float((ci_hi.iloc[-1]-ci_lo.iloc[-1])/cur*100)
        aic_bic_gap=abs(float(model.aic)-float(model.bic))
        if band_pct<25 and aic_bic_gap<10:
            return "High","#34d399","Tight forecast band and a well-fit model (AIC/BIC close together)."
        elif band_pct<50:
            return "Medium","#f59e0b","Moderate forecast uncertainty — treat the target as a range, not a point."
        else:
            return "Low","#f87171","Wide forecast band — long horizon or volatile stock reduces precision."
    except Exception:
        return "Medium","#f59e0b","Could not fully assess model fit — treat the forecast with caution."

def backtest_arima(close,p,q,test_days=60):
    """Train on all data except the last `test_days`, forecast that window, compare to actual."""
    if len(close)<test_days*2:
        return None
    train=close.iloc[:-test_days]; actual=close.iloc[-test_days:]
    try:
        log_ret=np.log(train/train.shift(1)).dropna()
        bt_model=ARIMA(log_ret,order=(p,0,q)).fit()
        fc_res=bt_model.get_forecast(steps=test_days)
        fc_lr=fc_res.predicted_mean.values
        last=float(train.iloc[-1])
        pred=last*np.exp(np.cumsum(fc_lr))
        n=min(len(pred),len(actual))
        pred_s=pd.Series(pred[:n],index=actual.index[:n])
        actual_s=actual.iloc[:n]
        mape=float(np.mean(np.abs((actual_s.values-pred_s.values)/actual_s.values))*100)
        return pred_s,actual_s,mape
    except Exception:
        return None

def get_sentiment(news):
    if not news: return 0.0,[]
    sia=SentimentIntensityAnalyzer(); scores,hl=[],[]
    for n in news:
        t=n.get("title","")
        if t:
            sc=sia.polarity_scores(t)["compound"]; scores.append(sc); hl.append((t,sc))
    return (float(np.mean(scores)) if scores else 0.0),hl

def trend_strength(close):
    """One-line trend label from price vs SMA20/50/200 alignment."""
    if len(close) < 200:
        return "Insufficient Data", "#94a3b8"
    px   = float(close.iloc[-1])
    s20  = float(close.rolling(20).mean().iloc[-1])
    s50  = float(close.rolling(50).mean().iloc[-1])
    s200 = float(close.rolling(200).mean().iloc[-1])
    if px > s20 > s50 > s200:  return "Strong Uptrend", "#10b981"
    if px < s20 < s50 < s200:  return "Strong Downtrend", "#dc2626"
    if px > s50 and px > s200: return "Uptrend", "#34d399"
    if px < s50 and px < s200: return "Downtrend", "#f87171"
    return "Sideways", "#f59e0b"

def compute_signal(cur, fc_end, rsi_now, trend_label):
    """Combine forecast direction, trend alignment and RSI momentum into BUY/SELL/WAIT."""
    score = 0
    chg_pct = (fc_end - cur) / cur * 100
    if chg_pct > 3:    score += 1
    elif chg_pct < -3: score -= 1
    if trend_label in ("Strong Uptrend", "Uptrend"):     score += 1
    elif trend_label in ("Strong Downtrend", "Downtrend"): score -= 1
    if rsi_now < 35:  score += 1
    elif rsi_now > 70: score -= 1
    if score >= 2:
        return ("BUY", "#10b981",
                "Forecast trend, price trend and momentum are aligned to the upside.")
    if score <= -2:
        return ("SELL", "#ef4444",
                "Forecast trend, price trend and momentum are aligned to the downside.")
    return ("WAIT", "#f59e0b",
            "Signals are mixed — wait for clearer confirmation before entering a position.")

def position_sizing(capital, entry, sl, t1, t2):
    """₹ capital → max shares, capital at risk, and potential reward at T1/T2."""
    if capital is None or capital <= 0 or entry is None or entry <= 0:
        return None
    risk_per_share = entry - sl
    max_shares = int(capital // entry)
    return {
        "max_shares":      max_shares,
        "total_cost":      max_shares * entry,
        "capital_at_risk": max_shares * risk_per_share if risk_per_share > 0 else 0.0,
        "reward_t1":       max_shares * (t1 - entry),
        "reward_t2":       max_shares * (t2 - entry),
    }

def rr_bar_chart(lvl, accent, bg, grid):
    """Horizontal Entry/SL/Target1-3 risk:reward bar chart."""
    labels = ["Target 3", "Target 2", "Target 1", "Entry", "Stop Loss"]
    values = [lvl["t3"], lvl["t2"], lvl["t1"], lvl["entry"], lvl["sl"]]
    pcts   = [lvl["t3_pct"], lvl["t2_pct"], lvl["t1_pct"], 0.0, lvl["sl_pct"]]
    colors = ["#a855f7", "#c084fc", "#34d399", accent, "#f87171"]
    text   = [f"{fmt_price(v)}  ({p:+.2f}%)" for v, p in zip(values, pcts)]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
        marker_color=colors, text=text, textposition="outside",
        textfont=dict(color="#e2e8f0", size=11)))
    fig.update_layout(paper_bgcolor=bg, plot_bgcolor=bg, height=260,
        margin=dict(l=0, r=90, t=10, b=10), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=grid, tickprefix="₹", tickformat=",.0f", color="#6b84a0"),
        yaxis=dict(color="#e2e8f0", tickfont=dict(size=12)),
        font=dict(family="Inter", color="#6b84a0", size=11))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:.5rem 0 .15rem 0;'>
      <span style='color:{ACCENT};font-size:1.05rem;font-weight:700;'>📈 ARIMA Forecaster</span><br>
      <span style='color:#2d4060;font-size:.65rem;'>IndiaForecast Suite · App 1 of 3</span>
    </div><hr style='border-color:{BORDER};margin:.4rem 0 .8rem 0;'>""",unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>🔍 One-Shot Search</div>",unsafe_allow_html=True)
    quick_search=st.text_input("Search stock or symbol",
        placeholder="e.g. Reliance, INFY, Apple, AAPL — press Enter",
        label_visibility="collapsed",key="quick_search").strip()
    st.caption("Type a name or symbol and hit Enter — everything below loads automatically. No mode to pick.")

    ticker_sym=""; short_name=""; long_name=""

    if quick_search:
        suggs=search_suggestions(quick_search)
        if suggs:
            st.markdown(f"<div style='color:{ACCENT};font-size:.65rem;margin:.3rem 0;'>Suggestions:</div>",
                        unsafe_allow_html=True)
            for name_s,sym_s in suggs[:4]:
                if st.button(f"{name_s}  ({sym_s})",key=f"sugg_{sym_s}",use_container_width=True):
                    quick_search=sym_s

        exch_default_idx=0
        exch=st.selectbox("Exchange (override if symbol is ambiguous)",
                          list(EXCHANGES.keys()),index=exch_default_idx)
        exch_suffix=EXCHANGES[exch]

        ticker_sym,_=resolve_symbol(quick_search,exch_suffix)
        # Override suffix with user-chosen exchange if they changed it
        for sfx in SUFFIX_EXCHANGE_MAP:
            if ticker_sym.endswith(sfx):
                ticker_sym=ticker_sym.replace(sfx,exch_suffix) if exch_suffix else ticker_sym.split(".")[0]
                break
        parts=ticker_sym.split("."); short_name=parts[0]; long_name=quick_search.title()

    st.markdown("<hr style='border-color:#1a2640;margin:.6rem 0;'>",unsafe_allow_html=True)
    with st.expander("📋 Or browse NSE List / Indexes",expanded=not bool(quick_search)):
        browse_mode=st.radio("Browse",["📋 NSE Live List","📊 Major Indexes"],
                             label_visibility="collapsed",key="browse_mode")
        if browse_mode=="📊 Major Indexes":
            idx=st.selectbox("Index / Asset",list(INDEXES.keys()),key="idx_sel")
            if not quick_search:
                ticker_sym=INDEXES[idx]; short_name=idx; long_name=idx
        else:
            with st.spinner("Loading NSE list…"):
                nse=fetch_nse_list()
            if nse:
                q2=st.text_input("🔍 Filter",placeholder="INFY, Reliance…",key="nse_filter")
                opts={k:v for k,v in nse.items() if q2.upper() in k.upper()} if q2.strip() else nse
                opts=opts or nse
                lbl=st.selectbox("Stock",list(opts.keys()),label_visibility="collapsed",key="nse_sel")
                if not quick_search:
                    ticker_sym=opts[lbl]
                    p2=lbl.split(" — "); short_name=p2[0].strip()
                    long_name=p2[1].strip() if len(p2)>1 else p2[0]
            else:
                st.warning("Could not load NSE list. Use the search box above instead.")
                if not quick_search:
                    ticker_sym="RELIANCE.NS"; short_name="RELIANCE"; long_name="Reliance Industries"

    st.markdown("<hr style='border-color:#1a2640;margin:.5rem 0;'>",unsafe_allow_html=True)
    tl_lbl=st.select_slider("Forecast Horizon",list(TIMELINES.keys()),value="6 Months")
    fc_days=TIMELINES[tl_lbl]

    st.markdown(f"<div class='sec-label' style='margin-top:.7rem;'>ARIMA Parameters</div>",
                unsafe_allow_html=True)
    auto_mode=st.toggle("Auto (p,q)",value=True)
    p_val=st.slider("p — AR",0,5,2) if not auto_mode else 2
    q_val=st.slider("q — MA",0,5,2) if not auto_mode else 2

    st.markdown(f"<div class='sec-label' style='margin-top:.7rem;'>Technical Indicators</div>",
                unsafe_allow_html=True)
    show_bb    =st.toggle("Bollinger Bands",     value=True)
    show_ci    =st.toggle("Confidence Band",     value=True)
    show_rsi   =st.toggle("RSI (14)",            value=True)
    show_macd  =st.toggle("MACD",                value=True)
    show_vol   =st.toggle("Volatility Chart",    value=True)
    show_ret   =st.toggle("Returns Distribution",value=False)
    show_sent  =st.toggle("News Sentiment",      value=True)
    show_trade =st.toggle("Trade Levels (Entry/SL/Target)", value=True)

    st.markdown(f"<div class='sec-label' style='margin-top:.7rem;'>Position Sizing</div>",
                unsafe_allow_html=True)
    capital=st.number_input("Total Capital (₹)",min_value=0,value=100000,step=10000,format="%d")

if not ticker_sym:
    st.info("Type a stock name or symbol in the sidebar to begin."); st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker_sym}…"):
    close=load_price(ticker_sym)
if close.empty or len(close)<60:
    st.error(f"No data for **{ticker_sym}**. Check the symbol or try another."); st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="hero">
  <div><span class="badge">ARIMA</span><span class="badge">{tl_lbl} Forecast</span>
  <span class="badge">{ticker_sym}</span></div>
  <p class="hero-ticker" style="margin-top:.6rem;">{short_name}</p>
  <p class="hero-name">{long_name}</p>
  <p class="hero-meta">{ticker_sym} · {len(close):,} trading days · Horizon: {tl_lbl}</p>
</div>""",unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
cur=float(close.iloc[-1]); prev=float(close.iloc[-2])
chg=cur-prev; pct=chg/prev*100
hi52=float(close[-252:].max()) if len(close)>=252 else float(close.max())
lo52=float(close[-252:].min()) if len(close)>=252 else float(close.min())
vol30=float(close.pct_change().rolling(30).std().iloc[-1]*np.sqrt(252)*100)

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Price",      fmt_price(cur),               fmt_chg(chg,pct))
c2.metric("52W High",   fmt_price(hi52))
c3.metric("52W Low",    fmt_price(lo52))
c4.metric("5Y High",    fmt_price(float(close.max())))
c5.metric("Ann. Vol",   f"{vol30:.1f}%")
c6.metric("Data Points",f"{len(close):,}")
st.markdown("<br>",unsafe_allow_html=True)

# ── ARIMA ─────────────────────────────────────────────────────────────────────
with st.spinner("Fitting ARIMA on log returns…"):
    try:
        fc_s,ci_lo,ci_hi,model=arima_price_forecast(close,p_val,q_val,fc_days)
        ok=True
    except Exception as e:
        st.error(f"ARIMA error: {e}"); ok=False
if not ok: st.stop()

rsi,macd,sig,hist_m,bb_mid,bb_up,bb_dn,vol_s=calc_indicators(close)

sent_score,headlines=0.0,[]
if show_sent:
    with st.spinner("Fetching news…"):
        sent_score,headlines=get_sentiment(load_news(ticker_sym))

# ── Signal Card · R:R Bar · Position Sizing ──────────────────────────────────
trend_lbl,trend_color=trend_strength(close)
rsi_clean=rsi.dropna()
rsi_now=float(rsi_clean.iloc[-1]) if len(rsi_clean) else 50.0
lvl=trading_levels(close,fc_s)
sig_label,sig_color,sig_reason=compute_signal(cur,float(fc_s.iloc[-1]),rsi_now,trend_lbl)

st.markdown(f"""<div style='background:{CARD};border:2px solid {sig_color};border-radius:14px;
  padding:1.2rem 1.6rem;margin-bottom:1rem;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:1rem;'>
  <div>
    <div style='color:{sig_color};font-size:2.1rem;font-weight:800;letter-spacing:.06em;'>{sig_label}</div>
    <div style='color:#94a3b8;font-size:.78rem;margin-top:.3rem;max-width:480px;line-height:1.5;'>{sig_reason}</div>
  </div>
  <div style='text-align:right;'>
    <div style='color:#4a6080;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;'>Trend Strength</div>
    <div style='color:{trend_color};font-size:1.15rem;font-weight:700;'>{trend_lbl}</div>
    <div style='color:#4a6080;font-size:.7rem;margin-top:.25rem;'>RSI {rsi_now:.0f} · {tl_lbl} target {fmt_price(float(fc_s.iloc[-1]))}</div>
  </div>
</div>""",unsafe_allow_html=True)

st.markdown("<div class='sec-label'>Risk : Reward — Entry · Stop Loss · Targets</div>",unsafe_allow_html=True)
st.plotly_chart(rr_bar_chart(lvl,ACCENT,BG,GRID),use_container_width=True)

ps=position_sizing(capital,lvl["entry"],lvl["sl"],lvl["t1"],lvl["t2"])
if ps:
    pc1,pc2,pc3,pc4=st.columns(4)
    pc1.metric("Max Shares",f"{ps['max_shares']:,}",f"Cost {fmt_price(ps['total_cost'])}")
    pc2.metric("Capital at Risk",fmt_price(ps['capital_at_risk']),
               f"{(ps['capital_at_risk']/capital*100):.1f}% of capital" if capital else None)
    pc3.metric("Potential Reward · T1",fmt_price(ps['reward_t1']))
    pc4.metric("Potential Reward · T2",fmt_price(ps['reward_t2']))
else:
    st.caption("Enter your total capital in the sidebar to see position sizing.")
st.markdown("<br>",unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
show_tech_sub=show_rsi or show_macd
rows=1+(1 if show_rsi else 0)+(1 if show_macd else 0)
row_h=[1.0]
if show_rsi:  row_h=[0.6 if show_macd else 0.65]+[0.35 if show_macd else 0.35]
if show_macd: row_h=row_h+[0.25] if show_rsi else [0.7,0.3]
if show_rsi and show_macd: row_h=[0.55,0.225,0.225]

st.markdown("<div class='sec-label'>Price Chart · ARIMA Forecast</div>",unsafe_allow_html=True)
fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,row_heights=row_h,vertical_spacing=0.03)

fig.add_trace(go.Scatter(x=close.index,y=close.values,name="Historical",
    line=dict(color=ACCENT,width=1.5),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"),row=1,col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=bb_up.index,y=bb_up.values,
        line=dict(color="rgba(56,189,248,.3)",width=1,dash="dot"),showlegend=False,name="BB+"),row=1,col=1)
    fig.add_trace(go.Scatter(x=bb_dn.index,y=bb_dn.values,
        line=dict(color="rgba(56,189,248,.3)",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(56,189,248,.04)",showlegend=False,name="BB-"),row=1,col=1)
    fig.add_trace(go.Scatter(x=bb_mid.index,y=bb_mid.values,
        line=dict(color="rgba(56,189,248,.5)",width=1,dash="dash"),showlegend=False,name="BB mid"),row=1,col=1)

if show_ci:
    fig.add_trace(go.Scatter(
        x=list(ci_hi.index)+list(ci_lo.index[::-1]),
        y=list(ci_hi.values)+list(ci_lo.values[::-1]),
        fill="toself",fillcolor="rgba(168,85,247,.12)",
        line=dict(color="rgba(168,85,247,.35)",width=1),
        name="95% CI",hoverinfo="skip"),row=1,col=1)

if show_sent and abs(sent_score)>0.05:
    adj=sent_score*cur*0.02
    fig.add_trace(go.Scatter(x=fc_s.index,y=(fc_s+adj).values,name="Sentiment-Adj",
        line=dict(color="#fbbf24",width=1.8,dash="dot"),
        hovertemplate="Sent-adj ₹%{y:,.2f}<extra></extra>"),row=1,col=1)

fig.add_trace(go.Scatter(x=fc_s.index,y=fc_s.values,name="ARIMA Forecast",
    line=dict(color="#c084fc",width=3,dash="dot"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"),row=1,col=1)

fig.add_vline(x=close.index[-1],line=dict(color="#334155",width=1,dash="dash"),
    annotation_text="Today",annotation_font=dict(color="#4a6080",size=9))

rsi_row=2 if show_rsi else None
macd_row=(3 if show_rsi else 2) if show_macd else None

if show_rsi and rsi_row:
    fig.add_trace(go.Scatter(x=rsi.index,y=rsi.values,name="RSI(14)",
        line=dict(color="#f59e0b",width=1.2)),row=rsi_row,col=1)
    for rsi_lvl,clr in [(70,"rgba(248,113,113,.4)"),(30,"rgba(52,211,153,.4)")]:
        fig.add_hline(y=rsi_lvl,line=dict(color=clr,width=1,dash="dash"),row=rsi_row,col=1)

if show_macd and macd_row:
    fig.add_trace(go.Bar(x=hist_m.index,y=hist_m.values,name="MACD Hist",
        marker_color=["#34d399" if v>=0 else "#f87171" for v in hist_m.values],opacity=0.7),
        row=macd_row,col=1)
    fig.add_trace(go.Scatter(x=macd.index,y=macd.values,name="MACD",
        line=dict(color=ACCENT,width=1.2)),row=macd_row,col=1)
    fig.add_trace(go.Scatter(x=sig.index,y=sig.values,name="Signal",
        line=dict(color="#f59e0b",width=1.2)),row=macd_row,col=1)

ax=dict(showgrid=True,gridcolor=GRID,gridwidth=1,zeroline=False,color="#6b84a0",tickfont=dict(size=10))
ht=560 if (show_rsi and show_macd) else 460 if (show_rsi or show_macd) else 420
fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,
    font=dict(family="Inter",color="#6b84a0",size=11),
    legend=dict(bgcolor=CARD,bordercolor=BORDER,borderwidth=1,font=dict(size=10),x=0.01,y=0.99),
    hovermode="x unified",margin=dict(l=0,r=0,t=10,b=0),height=ht)
fig.update_xaxes(**{k:v for k,v in ax.items() if k not in ["tickprefix"]},tickformat="%b %Y")
fig.update_yaxes(row=1,col=1,tickprefix="₹",tickformat=",.0f",**ax)
if show_rsi and rsi_row: fig.update_yaxes(row=rsi_row,col=1,title_text="RSI",range=[0,100],**ax)
if show_macd and macd_row: fig.update_yaxes(row=macd_row,col=1,title_text="MACD",**ax)
st.plotly_chart(fig,use_container_width=True)
st.markdown("<br>",unsafe_allow_html=True)

# ── Volatility ────────────────────────────────────────────────────────────────
if show_vol:
    with st.expander("📉 Volatility (30-day Rolling Annualised)",expanded=False):
        fv=go.Figure(go.Scatter(x=vol_s.index,y=vol_s.values,fill="tozeroy",
            fillcolor="rgba(56,189,248,.07)",line=dict(color=ACCENT,width=1.4)))
        fv.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,height=190,margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False,
            xaxis=dict(showgrid=True,gridcolor=GRID,tickformat="%b %Y",color="#6b84a0"),
            yaxis=dict(showgrid=True,gridcolor=GRID,ticksuffix="%",color="#6b84a0"),
            font=dict(family="Inter",color="#6b84a0",size=11))
        st.plotly_chart(fv,use_container_width=True)

if show_ret:
    with st.expander("📊 Returns Distribution",expanded=False):
        rets=close.pct_change().dropna()*100
        fr=go.Figure(go.Histogram(x=rets.values,nbinsx=80,marker_color=ACCENT,opacity=0.75))
        fr.add_vline(x=float(rets.mean()),line=dict(color="#f59e0b",width=1.5,dash="dash"),
            annotation_text=f"Mean {rets.mean():.2f}%",annotation_font=dict(color="#f59e0b",size=10))
        fr.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,height=200,margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False,
            xaxis=dict(showgrid=True,gridcolor=GRID,ticksuffix="%",color="#6b84a0"),
            yaxis=dict(showgrid=True,gridcolor=GRID,color="#6b84a0"),
            font=dict(family="Inter",color="#6b84a0",size=11))
        st.plotly_chart(fr,use_container_width=True)
        r1,r2,r3,r4=st.columns(4)
        r1.metric("Mean",f"{rets.mean():.3f}%"); r2.metric("Std",f"{rets.std():.3f}%")
        r3.metric("Skew",f"{float(rets.skew()):.3f}"); r4.metric("Kurt",f"{float(rets.kurt()):.3f}")

st.markdown("<br>",unsafe_allow_html=True)

# ── Trade Levels ──────────────────────────────────────────────────────────────
if show_trade:
    st.markdown("<div class='sec-label'>📍 Trade Levels — Entry · Stop Loss · Targets</div>",unsafe_allow_html=True)
    tc1,tc2,tc3,tc4=st.columns(4)
    tc1.markdown(f"""<div class='trade-card trade-entry'>
      <div style='color:#38bdf8;font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>Entry</div>
      <div style='color:#f0f6ff;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(lvl['entry'])}</div>
      <div style='color:#6b84a0;font-size:.7rem;'>RSI: {lvl['rsi']:.0f} · ATR: {fmt_price(lvl['atr'])}</div>
      <div style='color:#94a3b8;font-size:.68rem;margin-top:.3rem;'>{lvl['note']}</div>
    </div>""",unsafe_allow_html=True)
    tc2.markdown(f"""<div class='trade-card trade-sl'>
      <div style='color:#f87171;font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>Stop Loss</div>
      <div style='color:#f0f6ff;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(lvl['sl'])}</div>
      <div style='color:#f87171;font-size:.75rem;'>{lvl['sl_pct']:.2f}% from entry</div>
      <div style='color:#6b84a0;font-size:.68rem;margin-top:.3rem;'>1.5× ATR below entry</div>
    </div>""",unsafe_allow_html=True)
    tc3.markdown(f"""<div class='trade-card trade-t1'>
      <div style='color:#34d399;font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>Target 1 · 1.5× R:R</div>
      <div style='color:#f0f6ff;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(lvl['t1'])}</div>
      <div style='color:#34d399;font-size:.75rem;'>+{lvl['t1_pct']:.2f}% from entry</div>
      <div style='color:#6b84a0;font-size:.68rem;margin-top:.3rem;'>Conservative profit booking</div>
    </div>""",unsafe_allow_html=True)
    tc4.markdown(f"""<div class='trade-card trade-t2'>
      <div style='color:#c084fc;font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>Target 2 · Forecast</div>
      <div style='color:#f0f6ff;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(lvl['t2'])}</div>
      <div style='color:#c084fc;font-size:.75rem;'>{lvl['t2_pct']:+.2f}% · ARIMA target</div>
      <div style='color:#6b84a0;font-size:.68rem;margin-top:.3rem;'>Model end-of-horizon price</div>
    </div>""",unsafe_allow_html=True)
    st.markdown(f"""<div class='info-box'>
      <b>Risk:Reward</b> = <code>{lvl['rr']}×</code> &nbsp;·&nbsp;
      Risk per share = <code>{fmt_price(lvl['entry']-lvl['sl'])}</code> &nbsp;·&nbsp;
      Reward T1 = <code>{fmt_price(lvl['t1']-lvl['entry'])}</code> &nbsp;·&nbsp;
      Extended Target 3 (3× R:R) = <code>{fmt_price(lvl['t3'])}</code>
      <span style='color:#3d5070;'>&nbsp;({lvl['t3_pct']:+.1f}%)</span><br>
      <span style='color:#3d5070;font-size:.72rem;'>⚠ Trade levels are indicative only. Do your own research. Not financial advice.</span>
    </div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

# ── ARIMA Diagnostics — Confidence · Scenarios · Backtest ───────────────────
st.markdown("<div class='sec-label'>🎯 ARIMA Diagnostics</div>",unsafe_allow_html=True)
conf_lbl,conf_color,conf_note=confidence_label(model,ci_lo,ci_hi,cur)
bull_px=float(ci_hi.iloc[-1]); base_px=float(fc_s.iloc[-1]); bear_px=float(ci_lo.iloc[-1])

dcol1,dcol2,dcol3,dcol4=st.columns(4)
dcol1.markdown(f"""<div class='fund-card' style='text-align:center;'>
  <div style='color:#4a6080;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;'>Forecast Confidence</div>
  <div style='color:{conf_color};font-size:1.4rem;font-weight:800;margin:.3rem 0;'>{conf_lbl}</div>
  <div style='color:#6b84a0;font-size:.68rem;line-height:1.4;'>{conf_note}</div>
</div>""",unsafe_allow_html=True)
dcol2.markdown(f"""<div class='fund-card' style='text-align:center;border-color:rgba(52,211,153,.3);'>
  <div style='color:#34d399;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;'>Bull Scenario</div>
  <div style='color:#f0f6ff;font-size:1.25rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(bull_px)}</div>
  <div style='color:#34d399;font-size:.72rem;'>Upper 95% CI · {((bull_px-cur)/cur*100):+.1f}%</div>
</div>""",unsafe_allow_html=True)
dcol3.markdown(f"""<div class='fund-card' style='text-align:center;border-color:rgba(56,189,248,.3);'>
  <div style='color:{ACCENT};font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;'>Base Scenario</div>
  <div style='color:#f0f6ff;font-size:1.25rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(base_px)}</div>
  <div style='color:{ACCENT};font-size:.72rem;'>Point Forecast · {((base_px-cur)/cur*100):+.1f}%</div>
</div>""",unsafe_allow_html=True)
dcol4.markdown(f"""<div class='fund-card' style='text-align:center;border-color:rgba(248,113,113,.3);'>
  <div style='color:#f87171;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;'>Bear Scenario</div>
  <div style='color:#f0f6ff;font-size:1.25rem;font-weight:700;font-family:JetBrains Mono,monospace;margin:.3rem 0;'>{fmt_price(bear_px)}</div>
  <div style='color:#f87171;font-size:.72rem;'>Lower 95% CI · {((bear_px-cur)/cur*100):+.1f}%</div>
</div>""",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
with st.expander("🧪 60-Day Backtest — Predicted vs Actual",expanded=False):
    bt=backtest_arima(close,p_val,q_val,60)
    if bt:
        pred_s,actual_s,mape=bt
        fbt=go.Figure()
        fbt.add_trace(go.Scatter(x=actual_s.index,y=actual_s.values,name="Actual",
            line=dict(color="#e2e8f0",width=1.6)))
        fbt.add_trace(go.Scatter(x=pred_s.index,y=pred_s.values,name="Predicted",
            line=dict(color=ACCENT,width=1.6,dash="dot")))
        fbt.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,height=260,margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(bgcolor=CARD,bordercolor=BORDER,borderwidth=1,font=dict(size=10)),
            xaxis=dict(showgrid=True,gridcolor=GRID,tickformat="%b %Y",color="#6b84a0"),
            yaxis=dict(showgrid=True,gridcolor=GRID,tickprefix="₹",color="#6b84a0"),
            font=dict(family="Inter",color="#6b84a0",size=11))
        st.plotly_chart(fbt,use_container_width=True)
        bcol1,bcol2=st.columns(2)
        bcol1.metric("MAPE (60-day)",f"{mape:.2f}%")
        acc_lbl="Strong" if mape<5 else "Moderate" if mape<10 else "Weak"
        bcol2.metric("Backtest Read",acc_lbl)
        st.caption("Model trained on all data except the last 60 days, then forecast forward and "
                   "compared against what actually happened.")
    else:
        st.caption("Not enough price history for a 60-day backtest on this symbol.")
st.markdown("<br>",unsafe_allow_html=True)

# ── Monthly forecast + Fundamentals ──────────────────────────────────────────
left,right=st.columns([1.3,1])
with left:
    st.markdown("<div class='sec-label'>Monthly Forecast</div>",unsafe_allow_html=True)
    monthly=fc_s.to_frame("F").resample("ME").last(); monthly.index=monthly.index.strftime("%b %Y")
    pv=cur; rows_h=""
    for lbl,row in monthly.iterrows():
        v=row["F"]; d2=v-pv; pct2=d2/pv*100
        cls="fc-up" if d2>=0 else "fc-dn"; sym2="▲" if d2>=0 else "▼"
        rows_h+=f"<div class='fc-row'><span class='fc-dt'>{lbl}</span><span class='fc-px'>{fmt_price(v)}</span><span class='{cls}'>{sym2} {abs(pct2):.2f}%</span></div>"
        pv=v
    st.markdown(rows_h,unsafe_allow_html=True)

with right:
    st.markdown("<div class='sec-label'>ARIMA Summary</div>",unsafe_allow_html=True)
    fe=float(fc_s.iloc[-1]); up2=(fe-cur)/cur*100
    st.metric(f"{tl_lbl} Target",fmt_price(fe),f"{up2:+.1f}% from today")
    st.metric("Forecast High",  fmt_price(float(fc_s.max())))
    st.metric("Forecast Low",   fmt_price(float(fc_s.min())))
    st.markdown(f"""<div class='info-box'>
      <b>ARIMA({p_val},0,{q_val})</b> on log returns · {len(close):,} days<br>
      AIC <code>{model.aic:.1f}</code> · BIC <code>{model.bic:.1f}</code>
    </div>""",unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:.8rem;'>Fundamentals</div>",unsafe_allow_html=True)
    with st.spinner("Loading…"):
        info=load_info(ticker_sym)
    if info:
        fields=[("Market Cap",fmt_large(info.get("marketCap"))),
                ("P/E Ratio",f"{info.get('trailingPE','N/A')}"),
                ("EPS (TTM)",f"{info.get('trailingEps','N/A')}"),
                ("Div Yield",f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "N/A"),
                ("Beta",f"{info.get('beta','N/A')}"),("Sector",info.get("sector","N/A"))]
        st.markdown("<div class='fund-card'>"+"".join(
            f"<div class='fund-row'><span class='fund-key'>{k}</span><span class='fund-val'>{v}</span></div>"
            for k,v in fields)+"</div>",unsafe_allow_html=True)

# ── Sentiment ─────────────────────────────────────────────────────────────────
if show_sent and headlines:
    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>News Sentiment</div>",unsafe_allow_html=True)
    if sent_score>0.05:    bg="rgba(52,211,153,.08)";  bc="#34d399"; lb=f"BULLISH ({sent_score:+.2f})"
    elif sent_score<-0.05: bg="rgba(248,113,113,.08)"; bc="#f87171"; lb=f"BEARISH ({sent_score:+.2f})"
    else:                  bg="rgba(148,163,184,.08)"; bc="#94a3b8"; lb=f"NEUTRAL ({sent_score:+.2f})"
    st.markdown(f"<div class='sent-box' style='background:{bg};border-left:3px solid {bc};'>"
                f"<b style='color:{bc};'>Overall Sentiment: {lb}</b><br>"
                f"<span style='color:#6b84a0;font-size:.73rem;'>Based on {len(headlines)} recent headlines</span></div>",
                unsafe_allow_html=True)
    with st.expander("View headlines"):
        for h,sc in headlines:
            clr="#34d399" if sc>0.05 else "#f87171" if sc<-0.05 else "#94a3b8"
            st.markdown(f"<span style='color:{clr};font-size:.76rem;'>{'▲' if sc>0.05 else '▼' if sc<-0.05 else '●'} {h} ({sc:+.2f})</span>",unsafe_allow_html=True)

with st.expander("📋 Full Daily Forecast Table"):
    prev_arr=np.concatenate([[cur],fc_s.values[:-1]])
    fc_df=pd.DataFrame({
        "Date":fc_s.index.strftime("%d %b %Y"),
        "Forecast":fc_s.values.round(2),
        "Lower CI":ci_lo.values.round(2),
        "Upper CI":ci_hi.values.round(2),
        "Δ Day (₹)":(fc_s.values-prev_arr).round(2),
        "Δ Day (%)" :((fc_s.values-prev_arr)/prev_arr*100).round(3),
    })
    st.dataframe(fc_df.set_index("Date"),use_container_width=True,height=380)
    st.download_button("⬇ Download CSV",fc_df.to_csv(index=False).encode(),
                       f"ARIMA_{ticker_sym}_{tl_lbl.replace(' ','_')}.csv","text/csv")

st.markdown(f"<div style='text-align:center;color:#1e3050;font-size:.63rem;margin-top:.8rem;'>IndiaForecast · ARIMA · Yahoo Finance · Educational Use Only · Not Financial Advice</div>",unsafe_allow_html=True)
