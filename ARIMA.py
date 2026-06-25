# ─────────────────────────────────────────────────────────────────────────────
#  App 1 · ARIMA Forecaster  |  IndiaForecast Suite  |  Accent: Blue #38bdf8
#  Fix: ARIMA fitted on log returns (stationary) → cumsum back to price
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, io, warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
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
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{{color:#8896b0!important;}}
[data-testid="metric-container"]{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.2rem;}}
[data-testid="metric-container"] label{{color:#4a6080!important;font-size:.7rem!important;letter-spacing:.08em;text-transform:uppercase;}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{color:#f0f6ff!important;font-size:1.45rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{{font-size:.78rem!important;font-family:'JetBrains Mono',monospace!important;}}
.hero{{background:linear-gradient(135deg,#0a1930 0%,#071224 60%,#0a1930 100%);border:1px solid #1a3060;border-radius:14px;padding:1.8rem 2.2rem;margin-bottom:1.6rem;}}
.hero-ticker{{font-size:1.9rem;font-weight:700;color:{ACCENT};letter-spacing:-0.02em;margin:0;}}
.hero-name{{font-size:.95rem;color:#94a3b8;margin-top:.2rem;}}
.hero-meta{{font-size:.72rem;color:#3d5070;margin-top:.35rem;font-family:'JetBrains Mono',monospace;}}
.badge{{display:inline-block;background:rgba(56,189,248,.1);color:{ACCENT};border:1px solid rgba(56,189,248,.25);border-radius:20px;padding:.15rem .65rem;font-size:.67rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-right:.4rem;}}
.sec-label{{color:{ACCENT};font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid {BORDER};padding-bottom:.4rem;margin-bottom:.9rem;}}
.fc-row{{display:flex;justify-content:space-between;align-items:center;padding:.5rem .9rem;border-radius:7px;margin-bottom:.22rem;font-family:'JetBrains Mono',monospace;font-size:.8rem;}}
.fc-row:nth-child(odd){{background:{CARD};}} .fc-row:nth-child(even){{background:#0a101c;}}
.fc-dt{{color:#6b84a0;}} .fc-px{{color:#e2e8f0;font-weight:600;}}
.fc-up{{color:#34d399;}} .fc-dn{{color:#f87171;}}
.info-box{{background:#0a1624;border-left:3px solid {ACCENT};border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.78rem;color:#6b84a0;line-height:1.75;}}
.fund-card{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.2rem;}}
.fund-row{{display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid {BORDER};font-size:.8rem;}}
.fund-row:last-child{{border-bottom:none;}}
.fund-key{{color:#4a6080;}} .fund-val{{color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-weight:600;}}
.sent-box{{padding:.7rem 1rem;border-radius:8px;font-size:.78rem;margin-top:.5rem;}}
#MainMenu,footer,header{{visibility:hidden;}}
hr{{border-color:{BORDER};margin:1.2rem 0;}}
</style>""", unsafe_allow_html=True)

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

# ── Data helpers ──────────────────────────────────────────────────────────────
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

# ── Core forecast: log-return ARIMA → price ────────────────────────────────────
def arima_price_forecast(close, p, q, fc_days):
    """
    Fit ARIMA(p,0,q) on log returns. Log returns are stationary so d=0.
    Cumulative sum of forecast log-returns → price levels.
    This gives a proper trending forecast, not a flat line.
    """
    log_ret = np.log(close / close.shift(1)).dropna()
    model   = ARIMA(log_ret, order=(p, 0, q)).fit()
    fc_res  = model.get_forecast(steps=fc_days)
    fc_lr   = fc_res.predicted_mean.values          # forecast log-returns
    ci      = fc_res.conf_int()
    ci_lo_lr= ci.iloc[:,0].values
    ci_hi_lr= ci.iloc[:,1].values

    last = float(close.iloc[-1])
    # Cumulative log-return → price
    fc_px    = last * np.exp(np.cumsum(fc_lr))
    ci_lo_px = last * np.exp(np.cumsum(ci_lo_lr))
    ci_hi_px = last * np.exp(np.cumsum(ci_hi_lr))

    # Cap CI to ±40% of last price so chart stays readable
    band = last * 0.40
    ci_lo_px = np.maximum(ci_lo_px, fc_px - band)
    ci_hi_px = np.minimum(ci_hi_px, fc_px + band)

    fut = pd.bdate_range(start=close.index[-1], periods=fc_days+1)[1:]
    n   = min(len(fut), len(fc_px))
    return (pd.Series(fc_px[:n],    index=fut[:n]),
            pd.Series(ci_lo_px[:n], index=fut[:n]),
            pd.Series(ci_hi_px[:n], index=fut[:n]),
            model)

def calc_indicators(close):
    d=close.diff()
    gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan))
    e12=close.ewm(span=12).mean(); e26=close.ewm(span=26).mean()
    macd=e12-e26; sig=macd.ewm(span=9).mean(); hist=macd-sig
    bb=close.rolling(20).mean(); bbs=close.rolling(20).std()
    vol=close.pct_change().rolling(30).std()*np.sqrt(252)*100
    return rsi,macd,sig,hist,bb,bb+2*bbs,bb-2*bbs,vol

def get_sentiment(news):
    if not news: return 0.0,[]
    sia=SentimentIntensityAnalyzer(); scores,hl=[],[]
    for n in news:
        t=n.get("title","")
        if t:
            sc=sia.polarity_scores(t)["compound"]; scores.append(sc); hl.append((t,sc))
    return (float(np.mean(scores)) if scores else 0.0),hl

def fmt_large(v):
    try:
        v=float(v)
        if v>=1e12: return f"${v/1e12:.2f}T"
        if v>=1e9:  return f"${v/1e9:.2f}B"
        if v>=1e6:  return f"${v/1e6:.2f}M"
        return f"{v:,.2f}"
    except: return "N/A"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:.6rem 0 .2rem 0;'>
      <span style='color:{ACCENT};font-size:1.1rem;font-weight:700;'>📈 ARIMA Forecaster</span><br>
      <span style='color:#2d4060;font-size:.68rem;'>IndiaForecast Suite · App 1 of 3</span>
    </div><hr style='border-color:{BORDER};margin:.5rem 0 .9rem 0;'>""",unsafe_allow_html=True)

    mode=st.radio("Mode",["🌍 Exchange + Symbol","📋 NSE Live List","📊 Major Indexes"],label_visibility="collapsed")
    ticker_sym=""

    if mode=="📊 Major Indexes":
        n=st.selectbox("Index / Asset",list(INDEXES.keys()))
        ticker_sym=INDEXES[n]; short_name=n; long_name=n
    elif mode=="📋 NSE Live List":
        with st.spinner("Loading NSE list…"):
            nse=fetch_nse_list()
        if nse:
            q=st.text_input("🔍 Search",placeholder="INFY, Reliance…")
            opts={k:v for k,v in nse.items() if q.upper() in k.upper()} if q.strip() else nse
            opts=opts or nse
            lbl=st.selectbox("Stock",list(opts.keys()),label_visibility="collapsed")
            ticker_sym=opts[lbl]
            p2=lbl.split(" — "); short_name=p2[0].strip(); long_name=p2[1].strip() if len(p2)>1 else p2[0]
        else:
            st.warning("Could not load NSE list. Use Exchange mode.")
            ticker_sym="RELIANCE.NS"; short_name="RELIANCE"; long_name="Reliance Industries"
    else:
        exch=st.selectbox("Exchange",list(EXCHANGES.keys()))
        raw=st.text_input("Symbol",placeholder="INFY / AAPL / 7203").upper().strip()
        ticker_sym=raw+EXCHANGES[exch] if raw else ""
        short_name=raw; long_name=raw+" "+exch

    st.markdown("<hr style='border-color:#1a2640;margin:.6rem 0;'>",unsafe_allow_html=True)
    tl_lbl=st.select_slider("Forecast Horizon",list(TIMELINES.keys()),value="6 Months")
    fc_days=TIMELINES[tl_lbl]

    st.markdown(f"<div class='sec-label' style='margin-top:.8rem;'>ARIMA Parameters</div>",unsafe_allow_html=True)
    auto_mode=st.toggle("Auto (p,q)",value=True)
    p_val=st.slider("p — AR order",0,5,2) if not auto_mode else 2
    q_val=st.slider("q — MA order",0,5,2) if not auto_mode else 2

    show_ci  =st.toggle("Confidence band",value=True)
    show_tech=st.toggle("Technical Indicators",value=True)
    show_sent=st.toggle("News Sentiment",value=True)

if not ticker_sym:
    st.info("Enter a symbol in the sidebar to begin."); st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker_sym}…"):
    close=load_price(ticker_sym)
if close.empty or len(close)<60:
    st.error(f"Insufficient data for **{ticker_sym}**. Check symbol or try another."); st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="hero">
  <div><span class="badge">ARIMA · Log-Return</span><span class="badge">{tl_lbl} Forecast</span><span class="badge">{ticker_sym}</span></div>
  <p class="hero-ticker" style="margin-top:.65rem;">{short_name}</p>
  <p class="hero-name">{long_name}</p>
  <p class="hero-meta">{ticker_sym} · {len(close):,} trading days · Forecast horizon: {tl_lbl}</p>
</div>""",unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
cur=float(close.iloc[-1]); prev=float(close.iloc[-2])
chg=cur-prev; pct=chg/prev*100
hi52=float(close[-252:].max()) if len(close)>=252 else float(close.max())
lo52=float(close[-252:].min()) if len(close)>=252 else float(close.min())
vol30=float(close.pct_change().rolling(30).std().iloc[-1]*np.sqrt(252)*100)

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Price",      f"₹{cur:,.2f}",f"{chg:+.2f} ({pct:+.2f}%)")
c2.metric("52W High",   f"₹{hi52:,.2f}")
c3.metric("52W Low",    f"₹{lo52:,.2f}")
c4.metric("5Y High",    f"₹{float(close.max()):,.2f}")
c5.metric("Ann. Vol",   f"{vol30:.1f}%")
c6.metric("Data Points",f"{len(close):,}")
st.markdown("<br>",unsafe_allow_html=True)

# ── ARIMA forecast (log-return → price) ───────────────────────────────────────
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

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Price Chart · ARIMA Forecast · Bollinger Bands</div>",unsafe_allow_html=True)
rows=3 if show_tech else 1
row_h=[0.55,0.225,0.225] if show_tech else [1.0]
fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,row_heights=row_h,vertical_spacing=0.03)

# Historical
fig.add_trace(go.Scatter(x=close.index,y=close.values,name="Historical",
    line=dict(color=ACCENT,width=1.5),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"),row=1,col=1)

# Bollinger
if show_tech:
    fig.add_trace(go.Scatter(x=bb_up.index,y=bb_up.values,name="BB Upper",
        line=dict(color="rgba(56,189,248,.3)",width=1,dash="dot"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=bb_dn.index,y=bb_dn.values,name="BB Lower",
        line=dict(color="rgba(56,189,248,.3)",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(56,189,248,.04)",showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=bb_mid.index,y=bb_mid.values,name="BB Mid",
        line=dict(color="rgba(56,189,248,.5)",width=1,dash="dash"),showlegend=False),row=1,col=1)

# CI — drawn before forecast so forecast stays on top
if show_ci:
    fig.add_trace(go.Scatter(
        x=list(ci_hi.index)+list(ci_lo.index[::-1]),
        y=list(ci_hi.values)+list(ci_lo.values[::-1]),
        fill="toself",fillcolor="rgba(168,85,247,.13)",
        line=dict(color="rgba(168,85,247,.4)",width=1),
        name="95% CI",hoverinfo="skip"),row=1,col=1)

# Sentiment-adjusted line
if show_sent and abs(sent_score)>0.05:
    adj=sent_score*cur*0.02
    fig.add_trace(go.Scatter(x=fc_s.index,y=(fc_s+adj).values,name="Sentiment-Adj",
        line=dict(color="#fbbf24",width=1.8,dash="dot"),
        hovertemplate="Sent-adj ₹%{y:,.2f}<extra></extra>"),row=1,col=1)

# ARIMA forecast — drawn LAST so always on top
fig.add_trace(go.Scatter(x=fc_s.index,y=fc_s.values,name="ARIMA Forecast",
    line=dict(color="#c084fc",width=3,dash="dot"),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast ₹%{y:,.2f}<extra></extra>"),row=1,col=1)

fig.add_vline(x=close.index[-1],line=dict(color="#334155",width=1,dash="dash"),
    annotation_text="Today",annotation_font=dict(color="#4a6080",size=10))

# RSI
if show_tech:
    fig.add_trace(go.Scatter(x=rsi.index,y=rsi.values,name="RSI(14)",
        line=dict(color="#f59e0b",width=1.2)),row=2,col=1)
    for lvl,clr in [(70,"rgba(248,113,113,.4)"),(30,"rgba(52,211,153,.4)")]:
        fig.add_hline(y=lvl,line=dict(color=clr,width=1,dash="dash"),row=2,col=1)
    # MACD
    fig.add_trace(go.Bar(x=hist_m.index,y=hist_m.values,name="MACD Hist",
        marker_color=["#34d399" if v>=0 else "#f87171" for v in hist_m.values],opacity=0.7),row=3,col=1)
    fig.add_trace(go.Scatter(x=macd.index,y=macd.values,name="MACD",
        line=dict(color=ACCENT,width=1.2)),row=3,col=1)
    fig.add_trace(go.Scatter(x=sig.index,y=sig.values,name="Signal",
        line=dict(color="#f59e0b",width=1.2)),row=3,col=1)

ax=dict(showgrid=True,gridcolor=GRID,gridwidth=1,zeroline=False,color="#6b84a0",tickfont=dict(size=10))
fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,
    font=dict(family="Inter",color="#6b84a0",size=11),
    legend=dict(bgcolor=CARD,bordercolor=BORDER,borderwidth=1,font=dict(size=10),x=0.01,y=0.99),
    hovermode="x unified",margin=dict(l=0,r=0,t=10,b=0),height=580 if show_tech else 450)
fig.update_xaxes(**{k:v for k,v in ax.items() if k!="tickprefix"},tickformat="%b %Y")
fig.update_yaxes(row=1,col=1,tickprefix="₹",tickformat=",.0f",**ax)
if show_tech:
    fig.update_yaxes(row=2,col=1,title_text="RSI",range=[0,100],**ax)
    fig.update_yaxes(row=3,col=1,title_text="MACD",**ax)
st.plotly_chart(fig,use_container_width=True)
st.markdown("<br>",unsafe_allow_html=True)

# ── Volatility ────────────────────────────────────────────────────────────────
with st.expander("📉 Volatility (30-day Rolling Annualised)"):
    fv=go.Figure(go.Scatter(x=vol_s.index,y=vol_s.values,fill="tozeroy",
        fillcolor="rgba(56,189,248,.07)",line=dict(color=ACCENT,width=1.4),
        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra></extra>"))
    fv.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,height=200,
        margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
        xaxis=dict(showgrid=True,gridcolor=GRID,tickformat="%b %Y",color="#6b84a0"),
        yaxis=dict(showgrid=True,gridcolor=GRID,ticksuffix="%",color="#6b84a0"),
        font=dict(family="Inter",color="#6b84a0",size=11))
    st.plotly_chart(fv,use_container_width=True)

# ── Returns dist ──────────────────────────────────────────────────────────────
with st.expander("📊 Returns Distribution"):
    rets=close.pct_change().dropna()*100
    fr=go.Figure(go.Histogram(x=rets.values,nbinsx=80,marker_color=ACCENT,opacity=0.75))
    fr.add_vline(x=float(rets.mean()),line=dict(color="#f59e0b",width=1.5,dash="dash"),
        annotation_text=f"Mean {rets.mean():.2f}%",annotation_font=dict(color="#f59e0b",size=10))
    fr.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,height=210,
        margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
        xaxis=dict(showgrid=True,gridcolor=GRID,ticksuffix="%",color="#6b84a0"),
        yaxis=dict(showgrid=True,gridcolor=GRID,color="#6b84a0"),
        font=dict(family="Inter",color="#6b84a0",size=11))
    st.plotly_chart(fr,use_container_width=True)
    r1,r2,r3,r4=st.columns(4)
    r1.metric("Mean Daily",f"{rets.mean():.3f}%")
    r2.metric("Std Dev",   f"{rets.std():.3f}%")
    r3.metric("Skewness",  f"{float(rets.skew()):.3f}")
    r4.metric("Kurtosis",  f"{float(rets.kurt()):.3f}")

st.markdown("<br>",unsafe_allow_html=True)

# ── Monthly forecast table + fundamentals ─────────────────────────────────────
left,right=st.columns([1.3,1])

with left:
    st.markdown("<div class='sec-label'>Monthly Forecast</div>",unsafe_allow_html=True)
    monthly=fc_s.to_frame("F").resample("ME").last()
    monthly.index=monthly.index.strftime("%b %Y")
    pv=cur; rows_h=""
    for lbl,row in monthly.iterrows():
        v=row["F"]; d2=v-pv; pct2=d2/pv*100
        cls="fc-up" if d2>=0 else "fc-dn"; sym2="▲" if d2>=0 else "▼"
        rows_h+=f"<div class='fc-row'><span class='fc-dt'>{lbl}</span><span class='fc-px'>₹{v:,.2f}</span><span class='{cls}'>{sym2} {abs(pct2):.2f}%</span></div>"
        pv=v
    st.markdown(rows_h,unsafe_allow_html=True)

with right:
    st.markdown("<div class='sec-label'>ARIMA Summary</div>",unsafe_allow_html=True)
    fe=float(fc_s.iloc[-1]); up2=(fe-cur)/cur*100
    st.metric(f"{tl_lbl} Target",f"₹{fe:,.2f}",f"{up2:+.1f}% from today")
    st.metric("Forecast High",   f"₹{float(fc_s.max()):,.2f}")
    st.metric("Forecast Low",    f"₹{float(fc_s.min()):,.2f}")
    st.markdown(f"""<div class='info-box'>
      <b>ARIMA({p_val},0,{q_val})</b> on log returns · {len(close):,} days<br>
      AIC <code>{model.aic:.1f}</code> · BIC <code>{model.bic:.1f}</code><br>
      Horizon: <b>{tl_lbl}</b> ({fc_days} business days)
    </div>""",unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:1rem;'>Fundamentals</div>",unsafe_allow_html=True)
    with st.spinner("Loading…"):
        info=load_info(ticker_sym)
    if info:
        fields=[("Market Cap",fmt_large(info.get("marketCap"))),
                ("P/E Ratio",f"{info.get('trailingPE','N/A')}"),
                ("EPS (TTM)",f"{info.get('trailingEps','N/A')}"),
                ("Div Yield",f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "N/A"),
                ("52W High", f"₹{info.get('fiftyTwoWeekHigh','N/A')}"),
                ("52W Low",  f"₹{info.get('fiftyTwoWeekLow','N/A')}"),
                ("Beta",     f"{info.get('beta','N/A')}"),
                ("Sector",   info.get("sector","N/A"))]
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
                f"<b style='color:{bc};'>Sentiment: {lb}</b><br>"
                f"<span style='color:#6b84a0;font-size:.75rem;'>Based on {len(headlines)} headlines</span></div>",
                unsafe_allow_html=True)
    with st.expander("View headlines"):
        for h,sc in headlines:
            clr="#34d399" if sc>0.05 else "#f87171" if sc<-0.05 else "#94a3b8"
            st.markdown(f"<span style='color:{clr};font-size:.78rem;'>{'▲' if sc>0.05 else '▼' if sc<-0.05 else '●'} {h} ({sc:+.2f})</span>",unsafe_allow_html=True)

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("📋 Full Daily Forecast Table"):
    prev_arr=np.concatenate([[cur],fc_s.values[:-1]])
    fc_df=pd.DataFrame({
        "Date":         fc_s.index.strftime("%d %b %Y"),
        "Forecast (₹)": fc_s.values.round(2),
        "Lower CI (₹)": ci_lo.values.round(2),
        "Upper CI (₹)": ci_hi.values.round(2),
        "Δ Day (₹)":    (fc_s.values-prev_arr).round(2),
        "Δ Day (%)":    ((fc_s.values-prev_arr)/prev_arr*100).round(3),
    })
    st.dataframe(fc_df.set_index("Date"),use_container_width=True,height=400)
    st.download_button("⬇ Download CSV",fc_df.to_csv(index=False).encode(),
                       f"ARIMA_{ticker_sym}_{tl_lbl.replace(' ','_')}.csv","text/csv")

st.markdown(f"<div style='text-align:center;color:#1e3050;font-size:.65rem;margin-top:1rem;'>IndiaForecast · ARIMA App · Yahoo Finance · Educational Use Only</div>",unsafe_allow_html=True)
