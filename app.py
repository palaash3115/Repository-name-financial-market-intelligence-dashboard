import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, time as dtime
import pytz
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantEdge Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS  — Premium Dark Fintech Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #030810;
    color: #C9D1D9;
}
.stApp { background-color: #030810; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080E18 0%, #060C16 100%);
    border-right: 1px solid #0F1F33;
}
section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

/* ── Metric Cards ───────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #0A1020 0%, #0E1830 100%);
    border: 1px solid #1A2840;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #00D4FF44;
    box-shadow: 0 8px 32px rgba(0,212,255,0.08), 0 0 0 1px rgba(0,212,255,0.1);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D4FF, #0066FF);
    border-radius: 14px 14px 0 0;
}
.metric-card.green::before  { background: linear-gradient(90deg, #00FF88, #00C853); }
.metric-card.green:hover    { border-color: #00FF8844; box-shadow: 0 8px 32px rgba(0,255,136,0.08), 0 0 0 1px rgba(0,255,136,0.1); }
.metric-card.red::before    { background: linear-gradient(90deg, #FF4560, #FF0040); }
.metric-card.red:hover      { border-color: #FF456044; box-shadow: 0 8px 32px rgba(255,69,96,0.08); }
.metric-card.gold::before   { background: linear-gradient(90deg, #FFD700, #FF8C00); }
.metric-card.gold:hover     { border-color: #FFD70044; box-shadow: 0 8px 32px rgba(255,215,0,0.08); }
.metric-card.purple::before { background: linear-gradient(90deg, #BF5AF2, #7B2FFF); }
.metric-card.purple:hover   { border-color: #BF5AF244; box-shadow: 0 8px 32px rgba(191,90,242,0.08); }

/* Glow variant for key metrics */
.metric-card.glow-blue {
    border-color: #00D4FF33;
    box-shadow: 0 0 20px rgba(0,212,255,0.06), inset 0 0 40px rgba(0,212,255,0.02);
}
.metric-card.glow-green {
    border-color: #00FF8833;
    box-shadow: 0 0 20px rgba(0,255,136,0.06), inset 0 0 40px rgba(0,255,136,0.02);
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4A6080;
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: #E6EDF3;
    line-height: 1.1;
}
.metric-sub {
    font-size: 11px;
    color: #4A6080;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.3px;
}

/* ── Section Headers ────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 20px 0;
    padding-bottom: 14px;
    border-bottom: 1px solid #0F1F33;
}
.section-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0;
    letter-spacing: 0.3px;
}
.section-tag {
    background: linear-gradient(135deg, #0A1628, #0D2040);
    border: 1px solid #1A3050;
    border-radius: 5px;
    padding: 3px 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #00D4FF;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Top Banner ─────────────────────────────────────────────── */
.top-banner {
    background: linear-gradient(135deg, #080E1C 0%, #0A1428 40%, #070D1A 100%);
    border: 1px solid #1A2840;
    border-radius: 16px;
    padding: 26px 34px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.top-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.top-banner::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00D4FF33, transparent);
}
.banner-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #E6EDF3;
    letter-spacing: 0.3px;
}
.banner-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #00D4FF;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 5px;
    opacity: 0.8;
}

/* ── Decision Badge ─────────────────────────────────────────── */
.decision-badge {
    display: inline-block;
    padding: 10px 32px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    transition: box-shadow 0.3s ease;
}
.badge-buy  {
    background: rgba(0,255,136,0.08);
    border: 1.5px solid #00FF88;
    color: #00FF88;
    box-shadow: 0 0 24px rgba(0,255,136,0.15), inset 0 0 20px rgba(0,255,136,0.04);
}
.badge-sell {
    background: rgba(255,69,96,0.08);
    border: 1.5px solid #FF4560;
    color: #FF4560;
    box-shadow: 0 0 24px rgba(255,69,96,0.15), inset 0 0 20px rgba(255,69,96,0.04);
}
.badge-hold {
    background: rgba(255,215,0,0.08);
    border: 1.5px solid #FFD700;
    color: #FFD700;
    box-shadow: 0 0 24px rgba(255,215,0,0.15), inset 0 0 20px rgba(255,215,0,0.04);
}

/* ── Info Box ───────────────────────────────────────────────── */
.info-box {
    background: linear-gradient(135deg, #080E18, #0A1220);
    border: 1px solid #1A2840;
    border-left: 3px solid #00D4FF;
    border-radius: 10px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #5A7A9A;
    line-height: 1.9;
}

/* ── NIFTY 50 Table ─────────────────────────────────────────── */
.nifty-table-wrapper {
    background: linear-gradient(145deg, #070D18 0%, #090F1E 100%);
    border: 1px solid #1A2840;
    border-radius: 14px;
    overflow: hidden;
}
.nifty-header-row {
    display: grid;
    grid-template-columns: 2fr 1.2fr 1.2fr 1fr 1.2fr;
    padding: 10px 18px;
    background: #0A1525;
    border-bottom: 1px solid #1A2840;
}
.nifty-header-cell {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4A6080;
}
.nifty-row {
    display: grid;
    grid-template-columns: 2fr 1.2fr 1.2fr 1fr 1.2fr;
    padding: 9px 18px;
    border-bottom: 1px solid #0D1A28;
    transition: background 0.15s ease;
    align-items: center;
}
.nifty-row:hover { background: rgba(0,212,255,0.03); }
.nifty-row:last-child { border-bottom: none; }
.nifty-name {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #C9D1D9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.nifty-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4A6080;
    margin-top: 2px;
}
.nifty-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: #C9D1D9;
}
.nifty-change-pos {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #00FF88;
    background: rgba(0,255,136,0.08);
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
}
.nifty-change-neg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #FF4560;
    background: rgba(255,69,96,0.08);
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
}
.nifty-vol {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4A6080;
}
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px; height: 20px;
    border-radius: 50%;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    margin-right: 8px;
    flex-shrink: 0;
}
.rank-1  { background: rgba(255,215,0,0.15);  color: #FFD700; border: 1px solid #FFD70044; }
.rank-2  { background: rgba(192,192,192,0.12); color: #C0C0C0; border: 1px solid #C0C0C044; }
.rank-3  { background: rgba(205,127,50,0.12);  color: #CD7F32; border: 1px solid #CD7F3244; }
.rank-other { background: rgba(74,96,128,0.12); color: #4A6080; border: 1px solid #1A2840; }

/* ── Market Status Badge ────────────────────────────────────── */
.market-status-open {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(0,255,136,0.08);
    border: 1px solid #00FF8833;
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #00FF88;
    letter-spacing: 1px;
}
.market-status-closed {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,69,96,0.08);
    border: 1px solid #FF456033;
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #FF4560;
    letter-spacing: 1px;
}
.market-status-pre {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,215,0,0.08);
    border: 1px solid #FFD70033;
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #FFD700;
    letter-spacing: 1px;
}
.pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: pulse 1.8s infinite;
}
.pulse-green { background: #00FF88; box-shadow: 0 0 6px #00FF88; }
.pulse-red   { background: #FF4560; box-shadow: 0 0 6px #FF4560; }
.pulse-gold  { background: #FFD700; box-shadow: 0 0 6px #FFD700; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.85); }
}

/* ── Auto-refresh bar ───────────────────────────────────────── */
.refresh-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #070D18;
    border: 1px solid #1A2840;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 10px;
}
.refresh-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4A6080;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.refresh-time {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #00D4FF;
}

/* ── Input Widgets ──────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #080E18 !important;
    border-color: #1A2840 !important;
    color: #C9D1D9 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] { background-color: #080E18 !important; }
div[data-baseweb="select"] * { background-color: #080E18 !important; color: #C9D1D9 !important; }
.stButton > button {
    background: linear-gradient(135deg, #0A1628, #0D2040) !important;
    border: 1px solid #1A3050 !important;
    color: #00D4FF !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #00D4FF66 !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ── Plotly container ───────────────────────────────────────── */
.js-plotly-plot {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #1A2840;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070D18; }
::-webkit-scrollbar-thumb { background: #1A2840; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3850; }

/* ── Dataframe ──────────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
RISK_FREE_RATE = 0.065
MARKET_TICKER  = "^NSEI"
TRADING_DAYS   = 252
AUTO_REFRESH_S = 60          # auto-refresh interval in seconds

# NIFTY 50 – Top 20 most liquid stocks
NIFTY50_STOCKS: dict[str, str] = {
    "RELIANCE.NS":    "Reliance Industries",
    "TCS.NS":         "Tata Consultancy Svcs",
    "HDFCBANK.NS":    "HDFC Bank",
    "ICICIBANK.NS":   "ICICI Bank",
    "INFY.NS":        "Infosys",
    "HINDUNILVR.NS":  "Hindustan Unilever",
    "ITC.NS":         "ITC Limited",
    "SBIN.NS":        "State Bank of India",
    "BHARTIARTL.NS":  "Bharti Airtel",
    "KOTAKBANK.NS":   "Kotak Mahindra Bank",
    "LT.NS":          "Larsen & Toubro",
    "AXISBANK.NS":    "Axis Bank",
    "MARUTI.NS":      "Maruti Suzuki",
    "TITAN.NS":       "Titan Company",
    "BAJFINANCE.NS":  "Bajaj Finance",
    "WIPRO.NS":       "Wipro Limited",
    "ULTRACEMCO.NS":  "UltraTech Cement",
    "ASIANPAINT.NS":  "Asian Paints",
    "TATAMOTORS.NS":  "Tata Motors",
    "NESTLEIND.NS":   "Nestle India",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#060C18",
    font=dict(family="IBM Plex Mono, monospace", color="#4A6080", size=11),
    xaxis=dict(gridcolor="#0F1F33", showgrid=True, zeroline=False,
               tickfont=dict(color="#4A6080"), showspikes=True,
               spikecolor="#00D4FF", spikethickness=1),
    yaxis=dict(gridcolor="#0F1F33", showgrid=True, zeroline=False,
               tickfont=dict(color="#4A6080")),
    legend=dict(bgcolor="rgba(7,13,24,0.9)", bordercolor="#1A2840",
                borderwidth=1, font=dict(color="#4A6080")),
    margin=dict(l=16, r=16, t=44, b=16),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#080E18", bordercolor="#1A2840",
                    font=dict(color="#C9D1D9", family="IBM Plex Mono")),
)

POPULAR_STOCKS: dict[str, str] = {
    "RELIANCE.NS":   "Reliance Industries",
    "TCS.NS":        "Tata Consultancy",
    "HDFCBANK.NS":   "HDFC Bank",
    "INFY.NS":       "Infosys",
    "ICICIBANK.NS":  "ICICI Bank",
    "WIPRO.NS":      "Wipro",
    "SBIN.NS":       "State Bank of India",
    "TATAMOTORS.NS": "Tata Motors",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS":   "Adani Enterprises",
}

# ══════════════════════════════════════════════════════════════
#  AUTO-REFRESH  (session-state timer)
# ══════════════════════════════════════════════════════════════
if "last_auto_refresh" not in st.session_state:
    st.session_state.last_auto_refresh = time.time()

_elapsed = time.time() - st.session_state.last_auto_refresh
if _elapsed >= AUTO_REFRESH_S:
    st.session_state.last_auto_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

_next_refresh_in = max(0, int(AUTO_REFRESH_S - _elapsed))

# ══════════════════════════════════════════════════════════════
#  MARKET STATUS  (Indian IST hours)
# ══════════════════════════════════════════════════════════════
def get_market_status() -> tuple[str, str, str]:
    """
    Returns (status_label, css_class, dot_class) based on IST time.
    NSE hours: Pre-open 09:00–09:15, Market 09:15–15:30, Mon–Fri
    """
    ist   = pytz.timezone("Asia/Kolkata")
    now   = datetime.now(ist)
    wd    = now.weekday()          # 0=Mon … 4=Fri
    t     = now.time()
    pre_s = dtime(9,  0)
    mkt_s = dtime(9, 15)
    mkt_e = dtime(15, 30)

    if wd >= 5:                    # Weekend
        return "Market Closed", "market-status-closed", "pulse-red"
    if t < pre_s or t >= mkt_e:
        return "Market Closed", "market-status-closed", "pulse-red"
    if pre_s <= t < mkt_s:
        return "Pre-Open", "market-status-pre", "pulse-gold"
    return "Market Open", "market-status-open", "pulse-green"

# ══════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """Two-stage fetch: yf.download → Ticker.history fallback."""
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        return df[keep].dropna(how="all").copy()
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, actions=False)
        df = _clean(df)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        df = _clean(df)
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(period: str) -> pd.DataFrame:
    return fetch_data(MARKET_TICKER, period)

@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_nifty50_snapshot() -> pd.DataFrame:
    """
    Fetch 5-day OHLCV for all NIFTY 50 watchlist stocks,
    compute daily % change, sort by % change descending.
    """
    rows = []
    tickers_str = " ".join(NIFTY50_STOCKS.keys())
    try:
        raw = yf.download(
            tickers_str, period="5d",
            auto_adjust=True, progress=False, actions=False,
            group_by="ticker",
        )
    except Exception:
        raw = pd.DataFrame()

    for sym, name in NIFTY50_STOCKS.items():
        try:
            if raw.empty:
                raise ValueError("bulk download failed")
            if isinstance(raw.columns, pd.MultiIndex):
                df_s = raw[sym].dropna(how="all") if sym in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                df_s = raw.dropna(how="all")
            if df_s.empty or len(df_s) < 2:
                raise ValueError("empty")
            curr = float(df_s["Close"].iloc[-1])
            prev = float(df_s["Close"].iloc[-2])
            vol  = float(df_s["Volume"].iloc[-1])
        except Exception:
            # per-ticker fallback
            try:
                df_s = yf.Ticker(sym).history(period="5d", auto_adjust=True)
                if df_s.empty or len(df_s) < 2:
                    continue
                curr = float(df_s["Close"].iloc[-1])
                prev = float(df_s["Close"].iloc[-2])
                vol  = float(df_s["Volume"].iloc[-1])
            except Exception:
                continue

        chg_pct = (curr - prev) / prev * 100 if prev != 0 else 0.0
        rows.append({
            "Name":    name,
            "Ticker":  sym,
            "Price":   curr,
            "Change%": chg_pct,
            "Volume":  vol,
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.sort_values("Change%", ascending=False, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out

# ══════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    c    = df["Close"].squeeze()
    high = df["High"].squeeze()
    low  = df["Low"].squeeze()

    df["Returns"]   = c.pct_change()
    df["MA20"]      = c.rolling(20).mean()
    df["MA50"]      = c.rolling(50).mean()
    df["MA200"]     = c.rolling(200).mean()
    df["Volatility"]= df["Returns"].rolling(20).std() * np.sqrt(TRADING_DAYS)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    sma          = c.rolling(20).mean()
    std          = c.rolling(20).std()
    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std

    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df

# ══════════════════════════════════════════════════════════════
#  CAPM & RISK METRICS
# ══════════════════════════════════════════════════════════════
def compute_capm_metrics(stock_ret: pd.Series, market_ret: pd.Series):
    aligned = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if len(aligned) < 30:
        return None, None, None, None, None, None, None
    aligned.columns = ["stock", "market"]
    cov      = aligned.cov()
    beta     = cov.loc["stock","market"] / cov.loc["market","market"]
    mkt_ann  = aligned["market"].mean() * TRADING_DAYS
    act_ann  = aligned["stock"].mean()  * TRADING_DAYS
    vol      = aligned["stock"].std()   * np.sqrt(TRADING_DAYS)
    exp_ret  = RISK_FREE_RATE + beta * (mkt_ann - RISK_FREE_RATE)
    sharpe   = (act_ann - RISK_FREE_RATE) / vol if vol > 0 else 0
    treynor  = (act_ann - RISK_FREE_RATE) / beta if beta != 0 else 0
    alpha    = act_ann - exp_ret
    return float(beta), float(exp_ret), float(act_ann), float(vol), float(sharpe), float(treynor), float(alpha)

# ══════════════════════════════════════════════════════════════
#  ML MODEL
# ══════════════════════════════════════════════════════════════
def build_ml_model(df: pd.DataFrame):
    features = ["Close","Volume","MA20","MA50","RSI","Volatility","MACD","ATR"]
    dff = df[features].copy().dropna()
    if len(dff) < 100:
        return None, None, None, None, None, None, None, None
    c = dff["Close"].squeeze()
    dff = dff.copy()
    dff["Target"] = c.shift(-1)
    dff.dropna(inplace=True)
    X, y = dff[features].values, dff["Target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    mdl = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    mdl.fit(X_tr, y_tr)
    y_pred   = mdl.predict(X_te)
    r2       = float(r2_score(y_te, y_pred))
    mae      = float(mean_absolute_error(y_te, y_pred))
    rmse     = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    nxt      = float(mdl.predict(sc.transform(dff[features].iloc[-1:].values))[0])
    return mdl, sc, nxt, r2, mae, rmse, features, mdl.feature_importances_

# ══════════════════════════════════════════════════════════════
#  DECISION ENGINE
# ══════════════════════════════════════════════════════════════
def get_decision(cur, pred, sharpe, rsi, vol):
    score   = 0
    upside  = (pred - cur) / cur * 100
    reasons = []

    if upside > 1.5:    score += 2; reasons.append(f"Strong upside: +{upside:.1f}%")
    elif upside > 0.5:  score += 1; reasons.append(f"Moderate upside: +{upside:.1f}%")
    elif upside < -1.5: score -= 2; reasons.append(f"Strong downside: {upside:.1f}%")
    else:               score -= 1; reasons.append(f"Weak upside: {upside:.1f}%")

    if sharpe is not None:
        if sharpe > 1.5:    score += 2; reasons.append(f"High Sharpe: {sharpe:.2f}")
        elif sharpe > 0.5:  score += 1; reasons.append(f"Acceptable Sharpe: {sharpe:.2f}")
        elif sharpe < 0:    score -= 2; reasons.append(f"Negative Sharpe: {sharpe:.2f}")
        else:               score -= 1; reasons.append(f"Low Sharpe: {sharpe:.2f}")

    if rsi is not None:
        if rsi < 30:  score += 2; reasons.append(f"RSI oversold: {rsi:.0f}")
        elif rsi > 70: score -= 2; reasons.append(f"RSI overbought: {rsi:.0f}")

    if vol is not None and vol > 0.40:
        score -= 1; reasons.append(f"High volatility: {vol*100:.0f}%")

    if score >= 3:    return "BUY",  "badge-buy",  score, reasons
    elif score <= -2: return "SELL", "badge-sell", score, reasons
    else:             return "HOLD", "badge-hold", score, reasons

# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════
def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.73, 0.27], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=df["Close"].squeeze(), name="OHLC",
        increasing_fillcolor="#00FF88", increasing_line_color="#00FF88",
        decreasing_fillcolor="#FF4560", decreasing_line_color="#FF4560",
        line=dict(width=1),
    ), row=1, col=1)
    for col, color, name in [("MA20","#00D4FF","MA 20"),("MA50","#FFD700","MA 50"),("MA200","#BF5AF2","MA 200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col].squeeze(), name=name,
                                     line=dict(color=color, width=1.5, dash="dot"), opacity=0.85), row=1, col=1)
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"].squeeze(), name="BB Upper",
                                  line=dict(color="#2A3850", width=1, dash="dash"), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"].squeeze(), name="BB Lower",
                                  line=dict(color="#2A3850", width=1, dash="dash"), opacity=0.7,
                                  fill="tonexty", fillcolor="rgba(42,56,80,0.07)"), row=1, col=1)
    colors = ["#00FF88" if r >= 0 else "#FF4560" for r in df["Returns"].squeeze().fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(), name="Volume",
                          marker_color=colors, opacity=0.65), row=2, col=1)
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text=f"<b>{ticker}</b>  —  Price & Volume",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=560, showlegend=True,
        xaxis2=dict(gridcolor="#0F1F33", showgrid=True, rangeslider=dict(visible=False)),
        yaxis =dict(gridcolor="#0F1F33", showgrid=True,
                    title=dict(text="Price", font=dict(color="#4A6080", size=10))),
        yaxis2=dict(gridcolor="#0F1F33", showgrid=True,
                    title=dict(text="Volume", font=dict(color="#4A6080", size=10))),
    )
    fig.update_layout(**layout)
    return fig

def rsi_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        subplot_titles=["RSI (14)", "MACD (12/26/9)"])
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI",
                              line=dict(color="#00D4FF", width=2),
                              fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"), row=1, col=1)
    for lvl, color in [(70,"#FF4560"),(30,"#00FF88"),(50,"#1A2840")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=color, line_width=1, opacity=0.5, row=1, col=1)
    macd_hist = (df["MACD"] - df["Signal"]).squeeze().fillna(0)
    fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram",
                          marker_color=["#00FF88" if v>=0 else "#FF4560" for v in macd_hist],
                          opacity=0.55), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(), name="MACD",
                              line=dict(color="#00D4FF", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"].squeeze(), name="Signal",
                              line=dict(color="#FFD700", width=1.5)), row=2, col=1)
    layout = PLOTLY_LAYOUT.copy()
    layout.update(height=430, showlegend=True,
                  xaxis2=dict(gridcolor="#0F1F33", showgrid=True, rangeslider=dict(visible=False)))
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color="#4A6080", size=10, family="IBM Plex Mono"))
    return fig

def returns_distribution_chart(returns: pd.Series) -> go.Figure:
    r = returns.dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, name="Daily Returns",
                                marker_color="#00D4FF", opacity=0.7,
                                marker_line=dict(color="#060C18", width=0.4)))
    fig.add_vline(x=r.mean(), line_dash="dash", line_color="#FFD700", line_width=1.5,
                  annotation_text=f"μ={r.mean():.2f}%",
                  annotation_font=dict(color="#FFD700", size=10))
    fig.add_vline(x=0, line_color="#2A3850", line_width=1, opacity=0.6)
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Daily Returns Distribution</b>",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=330, xaxis_title="Return (%)", yaxis_title="Frequency",
    )
    fig.update_layout(**layout)
    return fig

def rolling_sharpe_chart(returns: pd.Series, window: int = 60) -> go.Figure:
    roll_ret  = returns.rolling(window).mean() * TRADING_DAYS
    roll_std  = returns.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    rs        = (roll_ret - RISK_FREE_RATE) / roll_std
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name=f"Rolling Sharpe ({window}d)",
                              line=dict(color="#BF5AF2", width=2),
                              fill="tozeroy", fillcolor="rgba(191,90,242,0.06)"))
    fig.add_hline(y=1, line_dash="dash", line_color="#00FF88", line_width=1, opacity=0.5,
                  annotation_text="Sharpe=1", annotation_font=dict(color="#00FF88", size=9))
    fig.add_hline(y=0, line_color="#FF4560", line_width=1, opacity=0.3)
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text=f"<b>Rolling {window}-Day Sharpe Ratio</b>",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=310,
    )
    fig.update_layout(**layout)
    return fig

def capm_scatter(stock_ret: pd.Series, mkt_ret: pd.Series, beta: float, ticker: str) -> go.Figure:
    aligned = pd.concat([stock_ret, mkt_ret], axis=1).dropna()
    aligned.columns = ["stock","market"]
    x = aligned["market"] * 100
    y = aligned["stock"]  * 100
    xl = np.linspace(x.min(), x.max(), 100)
    yl = RISK_FREE_RATE / TRADING_DAYS * 100 + beta * xl
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Daily Returns",
                              marker=dict(color="#00D4FF", size=4, opacity=0.4,
                                          line=dict(color="#060C18", width=0.3))))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode="lines", name=f"SCL (β={beta:.2f})",
                              line=dict(color="#FFD700", width=2, dash="dash")))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text=f"<b>{ticker} vs Market</b>  —  Security Characteristic Line",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=370, xaxis_title="Market Return (%)", yaxis_title=f"{ticker} Return (%)",
    )
    fig.update_layout(**layout)
    return fig

def feature_importance_chart(features: list, importances: np.ndarray) -> go.Figure:
    idx   = np.argsort(importances)[::-1]
    feat  = [features[i] for i in idx]
    imp   = importances[idx]
    alpha = [0.4 + 0.6 * v / imp.max() for v in imp]
    colors= [f"rgba(0,212,255,{a:.2f})" for a in alpha]
    fig   = go.Figure(go.Bar(x=imp, y=feat, orientation="h",
                              marker_color=colors, marker_line=dict(color="#060C18", width=0.4)))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Feature Importance</b>  —  Random Forest",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=310,
        yaxis=dict(gridcolor="#0F1F33", categoryorder="total ascending", tickfont=dict(color="#4A6080")),
    )
    fig.update_layout(**layout)
    return fig

def prediction_vs_actual_chart(df: pd.DataFrame, model, scaler, features: list) -> go.Figure:
    dff   = df[features].copy().dropna()
    close = df["Close"].squeeze()
    dff["Target"] = close.shift(-1)
    dff.dropna(inplace=True)
    preds = model.predict(scaler.transform(dff[features].values))
    tail  = min(120, len(dff))
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=dff.index[-tail:], y=dff["Target"].values[-tail:],
                              name="Actual", line=dict(color="#C9D1D9", width=1.5)))
    fig.add_trace(go.Scatter(x=dff.index[-tail:], y=preds[-tail:],
                              name="Predicted", line=dict(color="#00D4FF", width=1.5, dash="dot")))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Actual vs Predicted</b>  —  Close Price (Last 120 Days)",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=340,
    )
    fig.update_layout(**layout)
    return fig

def portfolio_performance_chart(norm_df: pd.DataFrame) -> go.Figure:
    palette = ["#00D4FF","#00FF88","#FFD700","#BF5AF2","#FF4560","#FF8C00","#00C853","#4A9EFF"]
    fig = go.Figure()
    for i, col in enumerate(norm_df.columns):
        fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col,
                                  mode="lines", line=dict(width=2, color=palette[i % len(palette)])))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Portfolio Performance</b>  —  Normalised (Base 100)",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=370,
    )
    fig.update_layout(**layout)
    return fig

def volatility_heatmap(ret_df: pd.DataFrame) -> go.Figure:
    corr = ret_df.corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#FF4560"],[0.5,"#060C18"],[1,"#00FF88"]],
        zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color="#4A6080"), bgcolor="#070D18",
                      bordercolor="#1A2840"),
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Correlation Matrix</b>",
                   font=dict(color="#C9D1D9", size=13, family="Space Grotesk"), x=0.01),
        height=370,
        xaxis=dict(gridcolor="#0F1F33", tickfont=dict(color="#4A6080")),
        yaxis=dict(gridcolor="#0F1F33", tickfont=dict(color="#4A6080")),
    )
    fig.update_layout(**layout)
    return fig

# ══════════════════════════════════════════════════════════════
#  UI HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════
def metric_card(label: str, value: str, sub: str = "", color: str = "default", glow: bool = False):
    extra = f" glow-{color}" if glow and color in ("blue","green") else ""
    card_cls = f"metric-card {color}{extra}"
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def section_header(icon: str, title: str, tag: str = ""):
    tag_html = f'<span class="section-tag">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon}&nbsp; {title}</h2>{tag_html}
    </div>""", unsafe_allow_html=True)

def fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{'+'if val>=0 else ''}{val:.{decimals}f}%"

def fmt_num(val: float, prefix: str = "") -> str:
    av = abs(val)
    if av >= 1e7: return f"{prefix}{val/1e7:.2f}Cr"
    if av >= 1e5: return f"{prefix}{val/1e5:.2f}L"
    return f"{prefix}{val:,.0f}"

# ══════════════════════════════════════════════════════════════
#  NIFTY 50 TABLE RENDERER
# ══════════════════════════════════════════════════════════════
def render_nifty50_panel(df_n: pd.DataFrame):
    """Render the styled NIFTY 50 top-20 table in HTML."""
    if df_n.empty:
        st.warning("NIFTY 50 data unavailable — Yahoo Finance may be rate-limiting. Auto-retrying.")
        return

    header = """
    <div class="nifty-table-wrapper">
    <div class="nifty-header-row">
        <div class="nifty-header-cell"># &nbsp; Stock</div>
        <div class="nifty-header-cell">Price (₹)</div>
        <div class="nifty-header-cell">Change %</div>
        <div class="nifty-header-cell">Volume</div>
        <div class="nifty-header-cell">Ticker</div>
    </div>"""

    rows_html = ""
    for rank, (_, row) in enumerate(df_n.iterrows(), 1):
        chg_cls  = "nifty-change-pos" if row["Change%"] >= 0 else "nifty-change-neg"
        chg_sign = "▲" if row["Change%"] >= 0 else "▼"
        chg_str  = f"{chg_sign} {abs(row['Change%']):.2f}%"
        rnk_cls  = {1:"rank-1", 2:"rank-2", 3:"rank-3"}.get(rank, "rank-other")

        rows_html += f"""
        <div class="nifty-row">
            <div style="display:flex; align-items:center;">
                <span class="rank-badge {rnk_cls}">{rank}</span>
                <div>
                    <div class="nifty-name">{row['Name']}</div>
                </div>
            </div>
            <div class="nifty-price">₹{row['Price']:,.2f}</div>
            <div><span class="{chg_cls}">{chg_str}</span></div>
            <div class="nifty-vol">{fmt_num(row['Volume'])}</div>
            <div class="nifty-ticker">{row['Ticker'].replace('.NS','')}</div>
        </div>"""

    closing = "</div>"
    st.markdown(header + rows_html + closing, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Branding ──────────────────────────────────────────────
    mkt_status, mkt_css, dot_css = get_market_status()
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))

    st.markdown(f"""
    <div style="padding:18px 0 20px 0; border-bottom:1px solid #0F1F33; margin-bottom:18px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; color:#E6EDF3; letter-spacing:0.3px;">
            QuantEdge <span style="color:#00D4FF;">Pro</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; color:#4A6080;
                    letter-spacing:2px; text-transform:uppercase; margin-top:4px;">
            Quantitative Intelligence
        </div>
        <div style="margin-top:12px;">
            <span class="{mkt_css}">
                <span class="pulse-dot {dot_css}"></span>
                {mkt_status}
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; color:#2A3850;
                    margin-top:6px; letter-spacing:0.5px;">
            IST {ist_now.strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary Ticker ────────────────────────────────────────
    st.markdown('<p style="font-size:9px; color:#4A6080; font-family:IBM Plex Mono; '
                'letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">PRIMARY TICKER</p>',
                unsafe_allow_html=True)

    selected_name = st.selectbox("Stock", list(POPULAR_STOCKS.values()), index=0,
                                  label_visibility="collapsed")
    ticker = [k for k, v in POPULAR_STOCKS.items() if v == selected_name][0]

    custom = st.text_input("Custom ticker", placeholder="e.g. AAPL, MSFT, ^NSEI",
                           label_visibility="collapsed")
    if custom.strip():
        ticker = custom.strip().upper()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Time Period ───────────────────────────────────────────
    st.markdown('<p style="font-size:9px; color:#4A6080; font-family:IBM Plex Mono; '
                'letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">TIME PERIOD</p>',
                unsafe_allow_html=True)
    period_map   = {"3 Months":"3mo","6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}
    period_label = st.selectbox("Period", list(period_map.keys()), index=2,
                                 label_visibility="collapsed")
    period = period_map[period_label]

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Portfolio Tickers ─────────────────────────────────────
    st.markdown('<p style="font-size:9px; color:#4A6080; font-family:IBM Plex Mono; '
                'letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">PORTFOLIO TICKERS</p>',
                unsafe_allow_html=True)

    _PORT_DEFAULT = "RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS"
    if "portfolio_raw" not in st.session_state:
        st.session_state["portfolio_raw"] = _PORT_DEFAULT

    portfolio_raw = st.text_area(
    "Portfolio",
    value=_PORT_DEFAULT,
    height=100,
    label_visibility="collapsed",
    help="One ticker per line",
    key="portfolio_raw"
    )
    portfolio_tickers = [t.strip().upper() for t in portfolio_raw.splitlines() if t.strip()]
    if not portfolio_tickers:
        portfolio_tickers = _PORT_DEFAULT.splitlines()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Refresh ───────────────────────────────────────────────
    refresh = st.button("⟳  Refresh Data", use_container_width=True)
    if refresh:
        st.session_state.last_auto_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

    # ── Footer ────────────────────────────────────────────────
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="border-top:1px solid #0F1F33; padding-top:14px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                    color:#2A3850; text-align:center; line-height:2;">
            Risk-Free Rate: 6.5% p.a.<br>
            Market Index: NIFTY 50<br>
            Data: Yahoo Finance<br>
            Auto-refresh: {AUTO_REFRESH_S}s
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD PRIMARY DATA
# ══════════════════════════════════════════════════════════════
with st.spinner("Loading market data…"):
    df_raw    = fetch_data(ticker, period)
    df_market = fetch_market_data(period)
    info      = get_ticker_info(ticker)

if df_raw.empty:
    st.markdown(f"""
    <div style="background:rgba(255,69,96,0.07); border:1px solid #FF456033;
                border-left:4px solid #FF4560; border-radius:12px; padding:20px 24px; margin:16px 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:#FF4560; font-weight:600; margin-bottom:10px;">
            ⚠ No data returned for <code style="background:rgba(255,69,96,0.12); padding:2px 8px;
            border-radius:4px;">{ticker}</code>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#4A6080; line-height:2;">
            &nbsp;• NSE tickers need <code>.NS</code> suffix (e.g. <code>TATAMOTORS.NS</code>)<br>
            &nbsp;• Yahoo Finance may be rate-limiting — wait 30s, then click <strong style="color:#00D4FF;">Refresh Data</strong><br>
            &nbsp;• Try a shorter time period (e.g. 6 Months)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = compute_indicators(df_raw)
df.dropna(subset=["MA20","MA50","RSI"], inplace=True)

close_s       = df["Close"].squeeze()
current_price = float(close_s.iloc[-1])
prev_price    = float(close_s.iloc[-2]) if len(close_s) > 1 else current_price
day_change    = current_price - prev_price
day_chg_pct   = (day_change / prev_price * 100) if prev_price != 0 else 0
volume_today  = float(df["Volume"].squeeze().iloc[-1])
n             = min(252, len(close_s))
high_52w      = float(close_s.rolling(n).max().iloc[-1])
low_52w       = float(close_s.rolling(n).min().iloc[-1])
rsi_now       = float(df["RSI"].squeeze().iloc[-1])
atr_now       = float(df["ATR"].squeeze().iloc[-1]) if "ATR" in df.columns else 0
vol_now       = float(df["Volatility"].squeeze().iloc[-1]) if "Volatility" in df.columns else 0

# CAPM
capm_result = None
beta_val = exp_ret = actual_ret = vol_ann = sharpe = treynor = alpha = None
if not df_market.empty:
    df_mkt = compute_indicators(df_market)
    try:
        capm_result = compute_capm_metrics(
            df["Returns"].squeeze().dropna(),
            df_mkt["Returns"].squeeze().dropna(),
        )
        if capm_result and capm_result[0] is not None:
            beta_val, exp_ret, actual_ret, vol_ann, sharpe, treynor, alpha = capm_result
    except Exception:
        pass

# ML
ml_result = build_ml_model(df)
model = scaler = next_price = r2 = mae = rmse = feat_names = feat_imp = None
if ml_result and ml_result[0] is not None:
    model, scaler, next_price, r2, mae, rmse, feat_names, feat_imp = ml_result

# Decision
decision_label = decision_class = decision_score = decision_reasons = None
if next_price is not None:
    decision_label, decision_class, decision_score, decision_reasons = get_decision(
        current_price, next_price, sharpe, rsi_now, vol_now
    )

# Meta
name_display = info.get("longName") or info.get("shortName") or ticker
sector_str   = info.get("sector",   "")
exchange_str = info.get("exchange", "")
currency_str = info.get("currency", "INR")
chg_color    = "#00FF88" if day_change >= 0 else "#FF4560"
chg_arrow    = "▲" if day_change >= 0 else "▼"
mkt_status, mkt_css, dot_css = get_market_status()

# ══════════════════════════════════════════════════════════════
#  AUTO-REFRESH STATUS BAR
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="refresh-bar">
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <span class="{mkt_css}">
            <span class="pulse-dot {dot_css}"></span>
            {mkt_status}
        </span>
        <span class="refresh-label">IST &nbsp;{ist_now.strftime('%d %b %Y  %H:%M:%S')}</span>
    </div>
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <span class="refresh-label">Next auto-refresh in</span>
        <span class="refresh-time">{_next_refresh_in}s</span>
        <span class="refresh-label" style="color:#1A2840;">|</span>
        <span class="refresh-label">Data: Yahoo Finance</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TOP BANNER  — Selected Stock
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="top-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
            <div class="banner-title">{name_display}</div>
            <div class="banner-sub">{ticker} &nbsp;&nbsp;|&nbsp;&nbsp; {exchange_str} &nbsp;&nbsp;|&nbsp;&nbsp; {sector_str}</div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:700; color:#E6EDF3; letter-spacing:0.5px;">
                {currency_str} {current_price:,.2f}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:{chg_color}; margin-top:5px; letter-spacing:0.5px;">
                {chg_arrow} {abs(day_change):,.2f} &nbsp;&nbsp; ({fmt_pct(day_chg_pct)})
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  §0  NIFTY 50 TOP 20 LIVE SCANNER
# ══════════════════════════════════════════════════════════════
section_header("📊", "NIFTY 50 — Live Market Scanner", "TOP 20 BY GAIN")

with st.spinner("Scanning NIFTY 50…"):
    df_nifty = fetch_nifty50_snapshot()

# Summary stats row
if not df_nifty.empty:
    gainers  = (df_nifty["Change%"] > 0).sum()
    losers   = (df_nifty["Change%"] < 0).sum()
    best_chg = df_nifty["Change%"].max()
    worst_chg= df_nifty["Change%"].min()
    best_nm  = df_nifty.loc[df_nifty["Change%"].idxmax(), "Name"]
    worst_nm = df_nifty.loc[df_nifty["Change%"].idxmin(), "Name"]
    avg_chg  = df_nifty["Change%"].mean()
    avg_color= "#00FF88" if avg_chg >= 0 else "#FF4560"

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        metric_card("Gainers / Losers",
                    f'<span style="color:#00FF88">{gainers}</span> / <span style="color:#FF4560">{losers}</span>',
                    "Of 20 tracked stocks", "default")
    with sc2:
        metric_card("Best Performer", fmt_pct(best_chg),
                    best_nm[:22], "green", glow=True)
    with sc3:
        metric_card("Worst Performer", fmt_pct(worst_chg),
                    worst_nm[:22], "red")
    with sc4:
        metric_card("Avg Move",
                    f'<span style="color:{avg_color}">{fmt_pct(avg_chg)}</span>',
                    "Equal-weight basket", "default")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Split table into two columns of 10 for better readability
col_l, col_r = st.columns(2)
if not df_nifty.empty:
    with col_l:
        render_nifty50_panel(df_nifty.iloc[:10].reset_index(drop=True))
    with col_r:
        render_nifty50_panel(df_nifty.iloc[10:20].reset_index(drop=True)
                             if len(df_nifty) > 10 else df_nifty.iloc[:0])
else:
    st.info("NIFTY 50 scanner data unavailable. Auto-refreshing shortly.")

# ══════════════════════════════════════════════════════════════
#  §1  MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════
section_header("📋", "Market Overview", "LIVE SNAPSHOT")

c1, c2, c3, c4, c5, c6 = st.columns(6)
rsi_label = "Oversold" if rsi_now < 30 else "Overbought" if rsi_now > 70 else "Neutral"
rsi_color = "green"    if rsi_now < 30 else "red"        if rsi_now > 70 else "default"

with c1: metric_card("Current Price", f"₹{current_price:,.2f}",
                     f"{chg_arrow} {fmt_pct(day_chg_pct)} today",
                     "green" if day_change >= 0 else "red", glow=True)
with c2: metric_card("Volume", fmt_num(volume_today), "Shares traded")
with c3: metric_card("52W High", f"₹{high_52w:,.2f}",
                     f"Dist: {(current_price/high_52w-1)*100:.1f}%")
with c4: metric_card("52W Low",  f"₹{low_52w:,.2f}",
                     f"Dist: {(current_price/low_52w-1)*100:.1f}%")
with c5: metric_card("RSI (14)", f"{rsi_now:.1f}", rsi_label, rsi_color)
with c6: metric_card("ATR (14)", f"₹{atr_now:.2f}", "Average True Range", "purple")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  §2  TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
section_header("📈", "Technical Analysis", "CHARTS")

st.plotly_chart(candlestick_chart(df, ticker), use_container_width=True,
                config={"displayModeBar": False})

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(rsi_macd_chart(df), use_container_width=True,
                    config={"displayModeBar": False})
with col_r:
    st.plotly_chart(returns_distribution_chart(df["Returns"].squeeze()),
                    use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════
#  §3  PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════
section_header("🤖", "Predictive Analytics", "ML · RANDOM FOREST")

if model is not None:
    pred_chg  = (next_price - current_price) / current_price * 100
    pred_color= "green" if next_price >= current_price else "red"

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Next-Day Forecast", f"₹{next_price:,.2f}",
                         f"Δ {fmt_pct(pred_chg)}", pred_color, glow=True)
    with m2: metric_card("R² Score", f"{r2:.4f}", "Model fit quality",
                         "green" if r2>0.85 else "gold" if r2>0.70 else "red")
    with m3: metric_card("MAE",  f"₹{mae:.2f}",  "Mean Absolute Error")
    with m4: metric_card("RMSE", f"₹{rmse:.2f}", "Root Mean Sq. Error")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    p1, p2 = st.columns([3, 2])
    with p1:
        st.plotly_chart(prediction_vs_actual_chart(df, model, scaler, feat_names),
                        use_container_width=True, config={"displayModeBar": False})
    with p2:
        st.plotly_chart(feature_importance_chart(feat_names, feat_imp),
                        use_container_width=True, config={"displayModeBar": False})

    # Decision signal
    if decision_label:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        d1, d2 = st.columns([1, 3])
        with d1:
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:center; padding:20px 10px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; color:#4A6080;
                            letter-spacing:2px; text-transform:uppercase; margin-bottom:14px;">
                    DECISION SIGNAL
                </div>
                <div class="decision-badge {decision_class}">{decision_label}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#4A6080;
                            margin-top:12px; text-align:center; line-height:1.7;">
                    ₹{next_price:,.2f}<br>{fmt_pct(pred_chg)}<br>
                    <span style="color:#2A3850;">Score: {decision_score:+d}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with d2:
            sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
            vol_str    = f"{vol_now*100:.1f}%" if vol_now else "N/A"
            reasons_html = "<br>".join(f"&nbsp;&nbsp;→ {r}" for r in (decision_reasons or []))
            st.markdown(f"""
            <div class="info-box">
                <strong style="color:#C9D1D9; font-size:13px;">Decision Logic</strong>
                <br><br>
                📌 Current Price &nbsp;&nbsp;&nbsp;: ₹{current_price:,.2f}<br>
                🎯 Predicted Next-Day : ₹{next_price:,.2f} ({fmt_pct(pred_chg)})<br>
                📊 RSI (14) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {rsi_now:.1f} ({rsi_label})<br>
                📉 Annual Volatility &nbsp;: {vol_str}<br>
                ⚡ Sharpe Ratio &nbsp;&nbsp;&nbsp;&nbsp;: {sharpe_str}
                <br><br>
                <span style="color:#2A3850;">Scoring factors:</span><br>
                {reasons_html}
                <br><br>
                <span style="color:#2A3850; font-size:10px;">
                Not financial advice. For research purposes only.
                </span>
            </div>""", unsafe_allow_html=True)
else:
    st.warning("Insufficient data for ML model. Try a longer time period (2Y+).")

# ══════════════════════════════════════════════════════════════
#  §4  RISK & RETURN ANALYSIS
# ══════════════════════════════════════════════════════════════
section_header("📉", "Risk & Return Analysis", "CAPM · BETA · SHARPE · ALPHA")

if capm_result and beta_val is not None:
    r1, r2c, r3, r4, r5 = st.columns(5)
    with r1: metric_card("Beta (β)", f"{beta_val:.3f}", "vs NIFTY 50",
                         "red" if abs(beta_val)>1.5 else "gold" if abs(beta_val)>1 else "green")
    with r2c: metric_card("Sharpe Ratio", f"{sharpe:.3f}", "Risk-adj. return",
                          "green" if sharpe>1 else "gold" if sharpe>0 else "red", glow=(sharpe>1))
    with r3:  metric_card("CAPM Exp. Ret", fmt_pct(exp_ret*100), "Annualised")
    with r4:  metric_card("Actual Return", fmt_pct(actual_ret*100), "Annualised",
                          "green" if actual_ret>=exp_ret else "red")
    with r5:  metric_card("Jensen's Alpha", fmt_pct(alpha*100), "Excess vs CAPM",
                          "green" if alpha>=0 else "red")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(
            capm_scatter(df["Returns"].squeeze(), df_mkt["Returns"].squeeze(), beta_val, ticker),
            use_container_width=True, config={"displayModeBar": False},
        )
    with ch2:
        st.plotly_chart(rolling_sharpe_chart(df["Returns"].squeeze()),
                        use_container_width=True, config={"displayModeBar": False})

    capm_formula = (f"E(R) = {RISK_FREE_RATE*100:.1f}% + {beta_val:.2f} × "
                    f"({actual_ret*100:.1f}% − {RISK_FREE_RATE*100:.1f}%) = {exp_ret*100:.2f}%")
    beta_desc = ("Aggressive — amplifies market" if beta_val > 1.2 else
                 "Defensive — less volatile"      if beta_val < 0.8 else
                 "Neutral — tracks market")
    st.markdown(f"""
    <div class="info-box">
        <strong style="color:#C9D1D9; font-size:13px;">CAPM Breakdown</strong><br><br>
        🔢 Formula &nbsp;&nbsp;&nbsp;: {capm_formula}<br>
        📌 Beta &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {beta_val:.3f} &nbsp; ({beta_desc})<br>
        ⚡ Alpha &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {fmt_pct(alpha*100)} &nbsp;
            {'✓ Outperforming CAPM' if alpha > 0 else '✗ Underperforming CAPM'}<br>
        📊 Treynor &nbsp;&nbsp;&nbsp;: {treynor:.4f} &nbsp; (return per unit systematic risk)<br>
        💹 Ann. Vol &nbsp;&nbsp;: {vol_ann*100:.2f}%
    </div>""", unsafe_allow_html=True)
else:
    st.info("CAPM metrics unavailable — NIFTY 50 index data could not be retrieved.")

# ══════════════════════════════════════════════════════════════
#  §5  PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════
section_header("💼", "Portfolio Analysis", "MULTI-ASSET")

valid_tickers = [t for t in portfolio_tickers if t]
if len(valid_tickers) >= 2:
    with st.spinner("Loading portfolio data…"):
        port_prices = {}
        port_rets   = {}
        port_names  = {}
        for t in valid_tickers:
            try:
                praw = fetch_data(t, period)
                if not praw.empty:
                    praw          = compute_indicators(praw)
                    pclose        = praw["Close"].squeeze()
                    port_prices[t]= pclose
                    port_rets[t]  = praw["Returns"].squeeze().dropna()
                    pi            = get_ticker_info(t)
                    port_names[t] = pi.get("shortName") or pi.get("longName") or t
            except Exception:
                continue

    if len(port_prices) >= 2:
        price_df = pd.DataFrame(port_prices).dropna()
        ret_df   = pd.DataFrame(port_rets).dropna()
        norm_df  = price_df / price_df.iloc[0] * 100
        norm_df.columns = [port_names.get(c, c) for c in norm_df.columns]

        st.plotly_chart(portfolio_performance_chart(norm_df),
                        use_container_width=True, config={"displayModeBar": False})

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(volatility_heatmap(ret_df),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            rows = []
            for t in ret_df.columns:
                r       = ret_df[t].dropna()
                ann_r   = r.mean() * TRADING_DAYS * 100
                ann_v   = r.std()  * np.sqrt(TRADING_DAYS) * 100
                sr      = (ann_r/100 - RISK_FREE_RATE) / (ann_v/100) if ann_v > 0 else 0
                mdd     = (price_df[t] / price_df[t].cummax() - 1).min() * 100
                rows.append({"Ticker":t, "Name":port_names.get(t,t),
                              "Ann.Ret":fmt_pct(ann_r), "Volatility":f"{ann_v:.1f}%",
                              "Sharpe":f"{sr:.2f}", "MaxDD":fmt_pct(mdd)})

            stats_df = pd.DataFrame(rows).set_index("Ticker")
            st.markdown('<p style="font-family:IBM Plex Mono,monospace; font-size:9px; '
                        'color:#4A6080; letter-spacing:1.5px; text-transform:uppercase; '
                        'margin:12px 0 8px 0;">PORTFOLIO STATISTICS</p>', unsafe_allow_html=True)
            st.dataframe(stats_df, use_container_width=True, height=250)

        # Equal-weight portfolio summary
        ew_ret = ret_df.mean(axis=1)
        ew_ann = ew_ret.mean() * TRADING_DAYS * 100
        ew_vol = ew_ret.std()  * np.sqrt(TRADING_DAYS) * 100
        ew_sr  = (ew_ann/100 - RISK_FREE_RATE) / (ew_vol/100) if ew_vol > 0 else 0
        ew_dd  = ((1+ew_ret).cumprod() / (1+ew_ret).cumprod().cummax() - 1).min() * 100

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        pw1, pw2, pw3, pw4 = st.columns(4)
        with pw1: metric_card("EW Portfolio Return", fmt_pct(ew_ann), "Annualised",
                              "green" if ew_ann>0 else "red")
        with pw2: metric_card("Portfolio Volatility", f"{ew_vol:.2f}%", "Annualised")
        with pw3: metric_card("Portfolio Sharpe", f"{ew_sr:.3f}", "Equal-weighted",
                              "green" if ew_sr>1 else "gold" if ew_sr>0 else "red")
        with pw4: metric_card("Max Drawdown", fmt_pct(ew_dd), "Equal-weighted",
                              "red" if ew_dd<-20 else "gold")
    else:
        st.warning("Could not fetch sufficient data for portfolio analysis.")
else:
    st.info("Add at least 2 tickers in the Portfolio section (sidebar) to enable portfolio analytics.")

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; border-top:1px solid #0F1F33; padding-top:20px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                color:#1A2840; line-height:2.2; letter-spacing:0.5px;">
        QuantEdge Pro &nbsp;·&nbsp; Powered by Yahoo Finance &nbsp;·&nbsp;
        NIFTY 50 Live Scanner &nbsp;·&nbsp; CAPM &amp; ML Analytics
        <br>Not financial advice &nbsp;·&nbsp; For research &amp; educational purposes only
    </div>
</div>
""", unsafe_allow_html=True)
