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

    if len(df) < 60:
        return None, None, None, None, None, None

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
    # Predict Next Day
    # -----------------------------

    last_data = df[features].iloc[-1:].values
    predicted_price = model.predict(last_data)[0]

    return predicted_price, df, None, r2, mae, rmse
