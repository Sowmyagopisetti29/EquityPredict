from data import fetch_stock_data

from sklearn.ensemble import RandomForestRegressor
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


# Step 3: Create target (next day's price change)
data["Target"] = data["Close"].shift(-1) - data["Close"]


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


# Step 7: Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)


# Step 8: Predict price changes
predicted_changes = model.predict(X_test)


# Step 9: Evaluate price changes
change_mae = mean_absolute_error(y_test, predicted_changes)

change_rmse = np.sqrt(
    mean_squared_error(y_test, predicted_changes)
)

change_r2 = r2_score(y_test, predicted_changes)


# Step 10: Zero-change baseline
baseline_changes = np.zeros(len(y_test))

baseline_mae = mean_absolute_error(
    y_test, baseline_changes
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_changes)
)


# Step 11: Display results
print("\nMODEL EVALUATION")
print("-------------------------")

print("\nRandom Forest:")
print("Change MAE:", round(change_mae, 2))
print("Change RMSE:", round(change_rmse, 2))
print("Change R2:", round(change_r2, 2))

print("\nZero-Change Baseline:")
print("Baseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))


# Step 12: Compare closing-price predictions
predicted_prices = (
    X_test["Close"].to_numpy() + predicted_changes
)

actual_prices = (
    X_test["Close"].to_numpy() + y_test.to_numpy()
)

price_baseline = X_test["Close"].to_numpy()

print("\nCLOSING PRICE COMPARISON")
print("-------------------------")

print(
    "Random Forest Price MAE:",
    round(mean_absolute_error(actual_prices, predicted_prices), 2)
)

print(
    "Baseline Price MAE:",
    round(mean_absolute_error(actual_prices, price_baseline), 2)
)