
from data import fetch_stock_data

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import numpy as np


# Step 1: Fetch historical data
data = fetch_stock_data("AAPL")


# Step 2: Create technical indicators
data["Daily_Return"] = data["Close"].pct_change()

data["Intraday_Return"] = (
    data["Close"] - data["Open"]
) / data["Open"]

data["High_Low_Range"] = (
    data["High"] - data["Low"]
) / data["Close"]

data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()

data["MA_5_Ratio"] = data["Close"] / data["MA_5"]
data["MA_10_Ratio"] = data["Close"] / data["MA_10"]

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
ema_26 = data["Close"].ewm(span=26, adjust=False).mean()

macd = ema_12 - ema_26

data["MACD"] = macd / data["Close"]

data["MACD_Signal"] = (
    macd.ewm(span=9, adjust=False).mean()
    / data["Close"]
)

middle_band = data["Close"].rolling(20).mean()
std_dev = data["Close"].rolling(20).std()

upper_band = middle_band + 2 * std_dev
lower_band = middle_band - 2 * std_dev

data["BB_Position"] = (
    (data["Close"] - lower_band)
    / (upper_band - lower_band)
)

data["Volatility"] = data["Daily_Return"].rolling(10).std()

data["Volume_Change"] = data["Volume"].pct_change()


# Step 3: Create next-day return target
data["Target"] = (
    data["Close"].shift(-1) / data["Close"]
) - 1


# Step 4: Clean data
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna().reset_index(drop=True)


# Step 5: Select features
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


# Step 6: Walk-forward validation
initial_train_size = int(len(data) * 0.6)

window_size = 50

model_errors = []
baseline_errors = []

for start in range(
    initial_train_size,
    len(data),
    window_size
):

    end = min(start + window_size, len(data))

    X_train = X.iloc[:start]
    y_train = y.iloc[:start]

    X_test = X.iloc[start:end]
    y_test = y.iloc[start:end]

    # Step 7: Create XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # Step 8: Train and predict
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Step 9: Calculate errors
    model_errors.extend(
        np.abs(y_test.to_numpy() - predictions)
    )

    baseline_errors.extend(
        np.abs(y_test.to_numpy())
    )


# Step 10: Calculate final MAE
model_mae = np.mean(model_errors)
baseline_mae = np.mean(baseline_errors)


# Step 11: Display results
print("\nXGBOOST WALK-FORWARD VALIDATION")
print("--------------------------------")

print("Test predictions:", len(model_errors))

print("XGBoost MAE:", round(model_mae, 5))

print("Baseline MAE:", round(baseline_mae, 5))

if model_mae < baseline_mae:
    print("\nXGBoost performs better than baseline.")
else:
    print("\nBaseline performs better than XGBoost.")
