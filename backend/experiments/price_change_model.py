
from data import fetch_stock_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import joblib


def train_price_change_model(symbol):

    # Step 1: Fetch historical stock data
    data = fetch_stock_data(symbol)

    # Step 2: Create features
    data["Previous_Close"] = data["Close"].shift(1)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["Daily_Change"] = data["Close"] - data["Open"]
    data["Daily_Return"] = data["Close"].pct_change()

    # Step 3: Calculate next-day price change
    data["Target"] = data["Close"].shift(-1) - data["Close"]

    # Step 4: Remove missing values
    data = data.dropna()

    # Step 5: Select input features
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Previous_Close",
        "MA_5",
        "MA_10",
        "Daily_Change",
        "Daily_Return"
    ]

    X = data[features]
    y = data["Target"]

    # Step 6: Split data chronologically
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Step 7: Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    # Step 8: Train the model
    model.fit(X_train, y_train)

    # Step 9: Predict price changes
    predicted_changes = model.predict(X_test)

    # Step 10: Convert changes into closing prices
    predicted_prices = X_test["Close"].to_numpy() + predicted_changes

    actual_prices = X_test["Close"].to_numpy() + y_test.to_numpy()

    # Step 11: Evaluate predictions
    mae = mean_absolute_error(actual_prices, predicted_prices)

    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

    r2 = r2_score(actual_prices, predicted_prices)

    # Step 12: Calculate baseline error
    baseline_predictions = X_test["Close"].to_numpy()

    baseline_mae = mean_absolute_error(
        actual_prices,
        baseline_predictions
    )

    print("\nPrice Change Model Results")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

    print("\nBaseline Results")
    print("Baseline MAE:", round(baseline_mae, 2))

    # Step 13: Predict the next trading day's closing price
    latest_data = fetch_stock_data(symbol)

    latest_data["Previous_Close"] = latest_data["Close"].shift(1)
    latest_data["MA_5"] = latest_data["Close"].rolling(5).mean()
    latest_data["MA_10"] = latest_data["Close"].rolling(10).mean()
    latest_data["Daily_Change"] = latest_data["Close"] - latest_data["Open"]
    latest_data["Daily_Return"] = latest_data["Close"].pct_change()

    latest_features = latest_data[features].dropna().iloc[[-1]]

    next_change = model.predict(latest_features)[0]

    current_price = latest_features["Close"].iloc[0]

    next_price = current_price + next_change

    print("\nStock:", symbol)
    print("Current Closing Price:", round(current_price, 2))
    print("Predicted Price Change:", round(next_change, 2))
    print("Predicted Next Closing Price:", round(next_price, 2))

    # Step 14: Save the trained model
    joblib.dump(model, f"{symbol}_model.pkl")

    print("\nModel saved successfully!")

    return model
