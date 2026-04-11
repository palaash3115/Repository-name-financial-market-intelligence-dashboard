import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import datetime
from models.model import predict_price

@st.cache_data(ttl=5)
def load_stock_data(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

st.set_page_config(page_title="Financial Market Intelligence", layout="wide")

st_autorefresh(interval=10 * 1000, key="datarefresh")

st.title("📊 Financial Market Intelligence Dashboard")
st.markdown("🟢 **LIVE MODE ACTIVE** (Auto-refresh every 10s)")
st.caption("Interactive AI-powered dashboard for stock analysis, forecasting, and portfolio tracking.")

# ---------------------------
# Sidebar Controls
# ---------------------------

st.sidebar.header("Controls")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Nestle India": "NESTLEIND.NS",
    "Tata Steel": "TATASTEEL.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "JSW Steel": "JSWSTEEL.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected_stock]
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "5y"])
currency = st.sidebar.selectbox("Display Currency", ["INR", "USD"])
show_ma = st.sidebar.checkbox("Show Moving Averages", True)
compare_stocks = st.sidebar.multiselect("Compare Stocks", list(stocks.values()))

# ---------------------------
# Load Data
# ---------------------------

data = load_stock_data(ticker, period)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    st.warning("No data returned for this stock. Try another time period.")
    st.stop()

# ---------------------------
# Indicators
# ---------------------------

data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# ---------------------------
# Metrics
# ---------------------------

usd_to_inr = 83

if len(data) < 2:
    st.error("Not enough valid data to display metrics")
    st.stop()

latest_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])
volume = int(data["Volume"].iloc[-1])
change = latest_price - prev_price
percent_change = (change / prev_price) * 100 if prev_price != 0 else 0

if currency == "USD":
    latest_price /= usd_to_inr
    change /= usd_to_inr
    currency_symbol = "$"
else:
    currency_symbol = "₹"

# ---------------------------
# Market Snapshot
# ---------------------------

st.subheader("📌 Market Snapshot")
st.caption(f"Last Updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
st.caption("This section shows the latest market statistics for the selected stock.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"{currency_symbol}{round(latest_price, 2)}")
col2.metric("Price Change", f"{currency_symbol}{round(change, 2)}")
col3.metric("Percent Change", f"{round(percent_change, 2)}%")
col4.metric("Volume", f"{volume:,}")

st.divider()

# ---------------------------
# Candlestick Chart
# ---------------------------

chart_col, gauge_col = st.columns([3, 1])

with chart_col:
    st.subheader("📈 Stock Price Chart")
    st.caption("Candlestick charts show the opening, highest, lowest, and closing price for each trading period.")

    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="MA200"))

    st.plotly_chart(fig, use_container_width=True)

with gauge_col:
    st.subheader("Momentum")
    momentum = max(min(percent_change * 10 + 50, 100), 0)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=momentum,
        title={"text": "Momentum"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

st.divider()

# ---------------------------
# Volume + Trend
# ---------------------------

c1, c2 = st.columns(2)

with c1:
    st.subheader("📦 Trading Volume")
    st.caption("Volume represents how many shares were traded during the period.")
    vol_chart = px.bar(data, x=data.index, y="Volume")
    st.plotly_chart(vol_chart, use_container_width=True)

with c2:
    st.subheader("📉 Price Trend")
    st.caption("This chart shows how the closing price has changed over time.")
    price_chart = px.line(data, x=data.index, y="Close")
    st.plotly_chart(price_chart, use_container_width=True)

st.divider()

# ---------------------------
# RSI
# ---------------------------

st.subheader("📊 RSI Indicator")
st.caption("RSI measures whether a stock may be overbought (>70) or oversold (<30).")

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
rsi_fig.add_hline(y=70, line_dash="dash")
rsi_fig.add_hline(y=30, line_dash="dash")
st.plotly_chart(rsi_fig, use_container_width=True)

st.divider()

# ---------------------------
# Stock Comparison
# ---------------------------

if compare_stocks:
    comp_df = pd.DataFrame()
    for stock in compare_stocks:
        try:
            temp = load_stock_data(stock, period)
            if isinstance(temp.columns, pd.MultiIndex):
                temp.columns = temp.columns.get_level_values(0)
            if not temp.empty:
                comp_df[stock] = temp["Close"]
        except Exception:
            continue
    comp_df.dropna(inplace=True)
    if not comp_df.empty:
        comp_fig = px.line(comp_df)
        st.plotly_chart(comp_fig, use_container_width=True)
        st.divider()

# ---------------------------
# AI Prediction (Next Day Only)
# ---------------------------

st.subheader("🤖 AI Price Prediction")
st.caption("Predictions are generated using a Random Forest machine learning model trained on historical market data.")

try:
    predicted_price, df_model, _, r2, mae, rmse = predict_price(ticker)

    if df_model is None or df_model.empty:
        st.error("Model failed due to insufficient data")
        st.stop()

    df_model = df_model.dropna()

    if len(df_model) == 0:
        st.error("Prediction data invalid")
        st.stop()

    current_price = float(df_model["Close"].iloc[-1])

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"₹{round(current_price, 2)}")
    col2.metric("Predicted Next Price", f"₹{round(predicted_price, 2)}")
    # ---------------------------
# 💡 BUY / SELL RECOMMENDATION
# ---------------------------
    st.subheader("💡 Recommendation")

    if predicted_price > current_price:
        st.success("🟢 BUY — Model predicts price will increase")
    elif predicted_price < current_price:
        st.error("🔴 SELL — Model predicts price will decrease")
    else:
        st.info("⚖️ HOLD — No significant change expected")

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=df_model.index,
        y=df_model["Close"],
        name="Historical Price"
    ))
    pred_fig.add_trace(go.Scatter(
        x=[df_model.index[-1]],
        y=[predicted_price],
        mode="markers",
        marker=dict(size=12),
        name="Predicted Price"
    ))
    st.plotly_chart(pred_fig, use_container_width=True)

    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", round(r2, 3))
    c2.metric("MAE", round(mae, 2))
    c3.metric("RMSE", round(rmse, 2))

except Exception as e:
    st.error(f"AI prediction failed: {e}")

st.divider()

# ---------------------------
# Portfolio Tracker
# ---------------------------

st.subheader("💼 Portfolio Tracker")

with st.form("portfolio_form"):
    portfolio_stock = st.selectbox("Select Stock", list(stocks.keys()))
    quantity = st.number_input("Quantity", min_value=1, step=1)
    buy_price = st.number_input("Buy Price", min_value=0.0)
    submit = st.form_submit_button("Calculate Portfolio Value")

if submit:
    try:
        symbol = stocks[portfolio_stock]
        port_data = load_stock_data(symbol, "1d")

        if isinstance(port_data.columns, pd.MultiIndex):
            port_data.columns = port_data.columns.get_level_values(0)

        if port_data.empty:
            st.error("Could not fetch current price. Please try again.")
        else:
            current_price = float(port_data["Close"].iloc[-1])
            investment_value = buy_price * quantity
            current_value = current_price * quantity
            profit = current_value - investment_value

            col1, col2, col3 = st.columns(3)
            col1.metric("Investment Value", f"₹{round(investment_value, 2)}")
            col2.metric("Current Value", f"₹{round(current_value, 2)}")
            col3.metric("Profit / Loss", f"₹{round(profit, 2)}")
    except Exception as e:
        st.error(f"Portfolio calculation failed: {e}")
