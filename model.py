import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from functools import lru_cache


@lru_cache(maxsize=64)
def predict_price(ticker):

    df = yf.download(ticker, period="1y", progress=False)

    if df.empty:
        return None, None, None, None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -----------------------------
    # Indicators
    # -----------------------------

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    df["Volatility"] = df["Close"].rolling(10).std()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.dropna()

    # -----------------------------
    # Features
    # -----------------------------

    features = ["Close", "Volume", "MA20", "MA50", "RSI", "Volatility"]

    X = df[features]
    y = df["Close"].shift(-1)

    X = X[:-1]
    y = y[:-1]

    # -----------------------------
    # Model
    # -----------------------------

    model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

    model.fit(X, y)

    # -----------------------------
    # Model Evaluation
    # -----------------------------

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    mae = mean_absolute_error(y, y_pred)

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # -----------------------------
    # Predict Tomorrow
    # -----------------------------

    last_data = df[features].iloc[-1:].values

    predicted_price = model.predict(last_data)[0]

    # -----------------------------
    # 5 Day Forecast
    # -----------------------------

    forecast = []

    temp_data = df.copy()

    for i in range(5):

        last_row = temp_data[features].iloc[-1:].values

        next_price = model.predict(last_row)[0]

        forecast.append(next_price)

        new_row = temp_data.iloc[-1].copy()

        new_row["Close"] = next_price

        temp_data = pd.concat([temp_data, pd.DataFrame([new_row])])

    return predicted_price, df, forecast, r2, mae, rmse
