
from data import fetch_stock_data
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np


# Step 1: Create a regularized XGBoost model
def create_model():

    return XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.02,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


# Step 2: Generate technical indicators
def create_features(data):

    data = data.copy()

    # Daily returns
    data["Daily_Return"] = data["Close"].pct_change()

    # Previous-day returns (lag features)
    for lag in [1, 2, 3, 5]:
        data[f"Return_Lag_{lag}"] = (
            data["Daily_Return"].shift(lag)
        )

    # Intraday return
    data["Intraday_Return"] = (
        (data["Close"] - data["Open"]) / data["Open"]
    )

    # High-low price range
    data["High_Low_Range"] = (
        (data["High"] - data["Low"]) / data["Close"]
    )

    # Moving averages
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()

    data["MA_5_Ratio"] = data["Close"] / data["MA_5"]
    data["MA_10_Ratio"] = data["Close"] / data["MA_10"]

    # Momentum
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_10"] = data["Close"].pct_change(10)

    # RSI
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = data["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema_26 = data["Close"].ewm(
        span=26, adjust=False
    ).mean()

    macd = ema_12 - ema_26

    data["MACD"] = macd / data["Close"]

    signal = macd.ewm(
        span=9, adjust=False
    ).mean()

    data["MACD_Signal"] = signal / data["Close"]

    # Bollinger Bands
    middle_band = data["Close"].rolling(20).mean()
    std_dev = data["Close"].rolling(20).std()

    upper_band = middle_band + 2 * std_dev
    lower_band = middle_band - 2 * std_dev

    band_width = upper_band - lower_band

    data["BB_Position"] = (
        (data["Close"] - lower_band) / band_width
    )

    # Volatility
    data["Volatility_5"] = (
        data["Daily_Return"].rolling(5).std()
    )

    data["Volatility_10"] = (
        data["Daily_Return"].rolling(10).std()
    )

    # Volume change
    data["Volume_Change"] = data["Volume"].pct_change()

    # Relative volume compared with recent average
    average_volume = data["Volume"].rolling(10).mean()

    data["Volume_Ratio"] = (
        data["Volume"] / average_volume
    )

    return data


# Step 3: Predict stock price
def predict_stock_price(symbol):

    # Fetch historical data
    data = fetch_stock_data(symbol).copy()

    if data.empty:
        raise ValueError("No stock data available")

    # Create features
    data = create_features(data)

    # Select model features
    features = [
        "Daily_Return",
        "Return_Lag_1",
        "Return_Lag_2",
        "Return_Lag_3",
        "Return_Lag_5",
        "Intraday_Return",
        "High_Low_Range",
        "MA_5_Ratio",
        "MA_10_Ratio",
        "Momentum_5",
        "Momentum_10",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Position",
        "Volatility_5",
        "Volatility_10",
        "Volume_Change",
        "Volume_Ratio"
    ]

    # Replace invalid values
    data[features] = data[features].replace(
        [np.inf, -np.inf], np.nan
    )

    # Target: next trading day's return
    data["Target"] = (
        data["Close"].shift(-1) / data["Close"]
    ) - 1

    # Prepare training dataset
    training_data = data.dropna(
        subset=features + ["Target"]
    )

    if len(training_data) < 100:
        raise ValueError("Not enough historical data")

    X = training_data[features]
    y = training_data["Target"]

    # Chronological 80:20 split
    split_index = int(len(training_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_test) < 2:
        raise ValueError("Not enough test data")

    # Train evaluation model
    evaluation_model = create_model()

    evaluation_model.fit(X_train, y_train)

    # Predictions on unseen test data
    test_predictions = evaluation_model.predict(X_test)

    # Model evaluation
    mae = mean_absolute_error(
        y_test, test_predictions
    ) * 100

    rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    ) * 100

    r2 = r2_score(y_test, test_predictions)

    # Zero-return baseline
    baseline_predictions = np.zeros(len(y_test))

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    ) * 100

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test, baseline_predictions
        )
    ) * 100

    # Directional accuracy
    directional_accuracy = np.mean(
        np.sign(y_test) == np.sign(test_predictions)
    ) * 100

    # Train final model on all available labeled data
    model = create_model()
    model.fit(X, y)

    # Get latest available feature row
    latest_data = data.dropna(subset=features)

    if latest_data.empty:
        raise ValueError("Not enough data for prediction")

    latest_features = latest_data[features].iloc[[-1]]

    latest_close = float(
        latest_data["Close"].iloc[-1]
    )

    # Predict next-day return
    predicted_return = float(
        model.predict(latest_features)[0]
    )

    # Calculate predicted closing price
    predicted_price = (
        latest_close * (1 + predicted_return)
    )

    # Return prediction and evaluation results
    return {
        "stock": symbol.upper(),

        "latest_close": round(latest_close, 2),

        "predicted_return_percent": round(
            predicted_return * 100, 3
        ),

        "predicted_next_close": round(
            predicted_price, 2
        ),

        "model_evaluation": {
            "mae_percent": round(float(mae), 4),
            "rmse_percent": round(float(rmse), 4),
            "r2_score": round(float(r2), 4),

            "baseline_mae_percent": round(
                float(baseline_mae), 4
            ),

            "baseline_rmse_percent": round(
                float(baseline_rmse), 4
            ),

            "directional_accuracy_percent": round(
                float(directional_accuracy), 2
            ),

            "training_samples": len(X_train),
            "testing_samples": len(X_test),

            "split": "Chronological 80:20"
        }
    }
