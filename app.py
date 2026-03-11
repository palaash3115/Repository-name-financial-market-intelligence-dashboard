import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from models.model import predict_price
@st.cache_data(ttl=600)
def load_stock_data(symbol, period):
    df = yf.download(symbol, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df
st.set_page_config(page_title="Financial Market Intelligence", layout="wide")

st.title("📊 Financial Market Intelligence Dashboard")
st.caption(
"Interactive AI-powered dashboard for stock analysis, forecasting, and portfolio tracking."
)

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

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

ticker = stocks[selected_stock]

period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo","3mo","6mo","1y","5y"]
)

currency = st.sidebar.selectbox(
    "Display Currency",
    ["INR","USD"]
)

show_ma = st.sidebar.checkbox("Show Moving Averages", True)

compare_stocks = st.sidebar.multiselect(
    "Compare Stocks",
    list(stocks.values())
)

# ---------------------------
# Download Data
# ---------------------------

data = load_stock_data(ticker, period)

# Fix MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Handle empty data
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

data["RSI"] = 100 - (100/(1+rs))

# ---------------------------
# Metrics
# ---------------------------

usd_to_inr = 83

latest_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])
volume = int(data["Volume"].iloc[-1])

change = latest_price - prev_price
percent_change = (change/prev_price)*100

if currency == "USD":
    latest_price /= usd_to_inr
    change /= usd_to_inr
    currency_symbol = "$"
else:
    currency_symbol = "₹"
st.subheader("📌 Market Snapshot")
st.caption("This section shows the latest market statistics for the selected stock.")
col1,col2,col3,col4 = st.columns(4)

col1.metric("Current Price", f"{currency_symbol}{round(latest_price,2)}")
col2.metric("Price Change", f"{currency_symbol}{round(change,2)}")
col3.metric("Percent Change", f"{round(percent_change,2)}%")
col4.metric("Volume", f"{volume:,}")

st.divider()

# ---------------------------
# Candlestick Chart
# ---------------------------

chart_col,gauge_col = st.columns([3,1])

with chart_col:

    st.subheader("📈 Stock Price Chart")
    st.caption("Candlestick charts show the opening, highest, lowest, and closing price for each trading period.")

    fig = go.Figure()

    fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_rangeslider_visible=False
)

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    if show_ma:

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="MA50"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA200"],
            name="MA200"
        ))

    st.plotly_chart(fig,use_container_width=True)

with gauge_col:

    st.subheader("Momentum")

    momentum = max(min(percent_change*10+50,100),0)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=momentum,
        title={"text":"Momentum"},
        gauge={"axis":{"range":[0,100]}}
    ))

    st.plotly_chart(gauge,use_container_width=True)

st.divider()

# ---------------------------
# Volume + Trend
# ---------------------------

c1,c2 = st.columns(2)

with c1:

    st.subheader("📦 Trading Volume")
    st.caption("Volume represents how many shares were traded during the period.")

    vol_chart = px.bar(data,x=data.index,y="Volume")

    st.plotly_chart(vol_chart,use_container_width=True)

with c2:

    st.subheader("📉 Price Trend")
    st.caption("This chart shows how the closing price has changed over time.")

    price_chart = px.line(data,x=data.index,y="Close")

    st.plotly_chart(price_chart,use_container_width=True)

st.divider()

# ---------------------------
# RSI
# ---------------------------

st.subheader("📊 RSI Indicator")
st.caption("RSI measures whether a stock may be overbought (>70) or oversold (<30).")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["RSI"],
    name="RSI"
))

rsi_fig.add_hline(y=70,line_dash="dash")
rsi_fig.add_hline(y=30,line_dash="dash")

st.plotly_chart(rsi_fig,use_container_width=True)

st.divider()

# ---------------------------
# Stock Comparison
# ---------------------------

if compare_stocks:

    comp_df = pd.DataFrame()

    for stock in compare_stocks:

        temp = load_stock_data(stock,period)

        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)

        comp_df[stock] = temp["Close"]

    comp_df.dropna(inplace=True)

    fig = px.line(comp_df)

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# ---------------------------
# Recent Market Data
# ---------------------------

st.subheader("Recent Market Data")

st.dataframe(data.tail(10),use_container_width=True)

st.divider()

# ---------------------------
# AI Prediction
# ---------------------------

st.subheader("🤖 AI Price Prediction")
st.caption("Predictions are generated using a Random Forest machine learning model trained on historical market data.")

try:

    predicted_price, df_model, forecast, r2, mae, rmse = predict_price(ticker)

    current_price = float(df_model["Close"].iloc[-1])

    col1,col2 = st.columns(2)

    col1.metric("Current Price", f"₹{round(current_price,2)}")
    col2.metric("Predicted Next Price", f"₹{round(predicted_price,2)}")

    if predicted_price > current_price:
        st.success("📈 BUY Signal")
    else:
        st.error("📉 SELL Signal")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_model.index,
        y=df_model["Close"],
        name="Historical Price"
    ))

    fig.add_trace(go.Scatter(
        x=[df_model.index[-1]],
        y=[predicted_price],
        mode="markers",
        marker=dict(size=12),
        name="Predicted Price"
    ))

    future_dates = pd.date_range(
        start=df_model.index[-1],
        periods=6,
        freq="D"
    )[1:]

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode="lines+markers",
        line=dict(dash="dash"),
        name="5-Day Forecast"
    ))

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("📅 5-Day Forecast")

    forecast_df = pd.DataFrame({
        "Day":["Day 1","Day 2","Day 3","Day 4","Day 5"],
        "Predicted Price":forecast
    })

    st.dataframe(forecast_df,use_container_width=True)

    st.subheader("📊 Model Performance")

    c1,c2,c3 = st.columns(3)

    c1.metric("R² Score",round(r2,3))
    c2.metric("MAE",round(mae,2))
    c3.metric("RMSE",round(rmse,2))

except Exception as e:

    st.error(f"AI prediction failed: {e}")

st.divider()

# ---------------------------
# Best Stock Today Scanner
# ---------------------------

st.subheader("🏆 Best Stock Today (AI Scanner)")

with st.spinner("Scanning stocks with AI..."):

    results = []

    for name, symbol in stocks.items():

        try:

            predicted_price, df_model, forecast, r2, mae, rmse = predict_price(symbol)

            if df_model is None or df_model.empty:
                continue

            current_price = float(df_model["Close"].iloc[-1])

            expected_return = ((predicted_price - current_price) / current_price) * 100

            results.append({
                "Stock": name,
                "Current Price": round(current_price,2),
                "Predicted Price": round(predicted_price,2),
                "Expected Return %": round(expected_return,2)
            })

        except:
            continue


if len(results) > 0:

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values("Expected Return %", ascending=False)

    best_stock = results_df.iloc[0]

    st.success(
        f"Best Opportunity Today: {best_stock['Stock']} "
        f"(Expected Gain: {best_stock['Expected Return %']}%)"
    )

    st.dataframe(
    results_df.style.highlight_max(
        subset=["Expected Return %"],
        color="green"
    ),
    use_container_width=True
)

else:

    st.warning("Scanner could not evaluate stocks. Try refreshing.")
# ---------------------------
# Portfolio Tracker
# ---------------------------

st.subheader("💼 Portfolio Tracker")
with st.form("portfolio_form"):

    portfolio_stock = st.selectbox(
        "Select Stock",
        list(stocks.keys())
    )

    quantity = st.number_input("Quantity", min_value=1, step=1)

    buy_price = st.number_input("Buy Price", min_value=0.0)

    submit = st.form_submit_button("Calculate Portfolio Value")

if submit:

    symbol = stocks[portfolio_stock]

    data = load_stock_data(symbol,"1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    current_price = float(data["Close"].iloc[-1])

    investment_value = buy_price * quantity

    current_value = current_price * quantity

    profit = current_value - investment_value

    col1, col2, col3 = st.columns(3)

    col1.metric("Investment Value", f"₹{round(investment_value,2)}")
    col2.metric("Current Value", f"₹{round(current_value,2)}")

    col3.metric("Profit / Loss", f"₹{round(profit,2)}")
