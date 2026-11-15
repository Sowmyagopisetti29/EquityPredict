
from data import fetch_stock_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def train_model(symbol):

    # Step 1: Fetch historical stock data
    data = fetch_stock_data(symbol)

    # Step 2: Create additional features
    data["Previous_Close"] = data["Close"].shift(1)

    data["MA_5"] = data["Close"].rolling(window=5).mean()

    data["MA_10"] = data["Close"].rolling(window=10).mean()

    data["Daily_Change"] = data["Close"] - data["Open"]

    data["Daily_Return"] = data["Close"].pct_change()

    # Step 3: Create the target (next day's closing price)
    data["Target"] = data["Close"].shift(-1)

    # Step 4: Remove rows containing missing values
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

    # Step 7: Create the Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Step 8: Train the model
    model.fit(X_train, y_train)

    # Step 9: Predict test data
    predictions = model.predict(X_test)

    # Step 10: Evaluate the model
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    # Step 11: Compare with baseline
    baseline_predictions = X_test["Close"]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print("\nRandom Forest Results")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

    print("\nBaseline Results")
    print("Baseline MAE:", round(baseline_mae, 2))

    # Step 12: Predict the next trading day's closing price
    latest_data = fetch_stock_data(symbol)

    latest_data["Previous_Close"] = latest_data["Close"].shift(1)
    latest_data["MA_5"] = latest_data["Close"].rolling(5).mean()
    latest_data["MA_10"] = latest_data["Close"].rolling(10).mean()
    latest_data["Daily_Change"] = latest_data["Close"] - latest_data["Open"]
    latest_data["Daily_Return"] = latest_data["Close"].pct_change()

    latest_features = latest_data[features].dropna().iloc[[-1]]

    next_day_prediction = model.predict(latest_features)[0]

    print("\nStock:", symbol)
    print("Predicted Next Closing Price:", round(next_day_prediction, 2))

    return model
