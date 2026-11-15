
from data import fetch_stock_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


def train_technical_model(symbol):

    # Step 1: Fetch historical data
    data = fetch_stock_data(symbol)

    # Step 2: Calculate daily returns
    data["Daily_Return"] = data["Close"].pct_change()

    data["Intraday_Return"] = (
        data["Close"] - data["Open"]
    ) / data["Open"]

    data["High_Low_Range"] = (
        data["High"] - data["Low"]
    ) / data["Close"]

    # Step 3: Calculate moving averages
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()

    data["MA_5_Ratio"] = data["Close"] / data["MA_5"]
    data["MA_10_Ratio"] = data["Close"] / data["MA_10"]

    # Step 4: Calculate RSI
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # Step 5: Calculate MACD
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = (ema_12 - ema_26) / data["Close"]

    macd_signal = (
        (ema_12 - ema_26).ewm(span=9, adjust=False).mean()
    )

    data["MACD_Signal"] = macd_signal / data["Close"]

    # Step 6: Calculate Bollinger Band position
    middle_band = data["Close"].rolling(20).mean()
    std_dev = data["Close"].rolling(20).std()

    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)

    data["BB_Position"] = (
        (data["Close"] - lower_band)
        / (upper_band - lower_band)
    )

    # Step 7: Calculate volatility and volume change
    data["Volatility"] = data["Daily_Return"].rolling(10).std()

    data["Volume_Change"] = data["Volume"].pct_change()

    # Step 8: Predict next day's percentage return
    data["Target"] = (
        data["Close"].shift(-1) / data["Close"]
    ) - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Step 9: Select features
    features = [
        "Daily_Return",
        "Intraday_Return",
        "High_Low_Range",
        "MA_5_Ratio",
        "MA_10_Ratio",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Position",
        "Volatility",
        "Volume_Change"
    ]

    X = data[features]
    y = data["Target"]

    # Step 10: Chronological train-test split
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Step 11: Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    # Step 12: Train model
    model.fit(X_train, y_train)

    # Step 13: Predict percentage returns
    predictions = model.predict(X_test)

    # Step 14: Evaluate return predictions
    model_mae = mean_absolute_error(y_test, predictions)

    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    model_r2 = r2_score(y_test, predictions)

    # Zero-return baseline
    baseline_predictions = np.zeros(len(y_test))

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    # Step 15: Convert returns into closing prices
    actual_prices = (
        data["Close"].iloc[split:].to_numpy()
        * (1 + y_test.to_numpy())
    )

    predicted_prices = (
        data["Close"].iloc[split:].to_numpy()
        * (1 + predictions)
    )

    baseline_prices = data["Close"].iloc[split:].to_numpy()

    price_mae = mean_absolute_error(
        actual_prices, predicted_prices
    )

    baseline_price_mae = mean_absolute_error(
        actual_prices, baseline_prices
    )

    # Step 16: Display results
    print("\nTECHNICAL INDICATORS MODEL")
    print("-----------------------------")

    print("Return MAE:", round(model_mae, 5))
    print("Return RMSE:", round(model_rmse, 5))
    print("Return R2:", round(model_r2, 3))

    print("\nZERO-RETURN BASELINE")
    print("-----------------------------")

    print("Baseline MAE:", round(baseline_mae, 5))
    print("Baseline RMSE:", round(baseline_rmse, 5))

    print("\nCLOSING PRICE COMPARISON")
    print("-----------------------------")

    print("Model Price MAE:", round(price_mae, 2))
    print("Baseline Price MAE:", round(baseline_price_mae, 2))

    return model


if __name__ == "__main__":
    train_technical_model("AAPL")
