
from data import fetch_stock_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# Step 1: Fetch stock data
data = fetch_stock_data("AAPL")


# Step 2: Create features
data["Previous_Close"] = data["Close"].shift(1)
data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()
data["Daily_Change"] = data["Close"] - data["Open"]
data["Daily_Return"] = data["Close"].pct_change()


# Step 3: Create target
data["Target"] = data["Close"].shift(-1)


# Step 4: Remove missing values
data = data.dropna()


# Step 5: Select features
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


# Step 7: Create models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# Step 8: Train and evaluate both models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))


# Step 9: Evaluate baseline
baseline_predictions = X_test["Close"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\nBaseline Results")
print("Baseline MAE:", round(baseline_mae, 2))
