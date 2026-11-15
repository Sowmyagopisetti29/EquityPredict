
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data import fetch_stock_data


# --------------------------------------------------
# 1. MODEL CONFIGURATIONS
# --------------------------------------------------

MODEL_CONFIGS = {
    "conservative": {
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10
    },

    "balanced": {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5
    },

    "deeper": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 10
    }
}


# --------------------------------------------------
# 2. CREATE XGBOOST MODEL
# --------------------------------------------------

def create_model(config):

    return XGBRegressor(
        **config,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

def create_features(data):

    data = data.copy()

    # Daily return
    data["Daily_Return"] = data["Close"].pct_change()

    # Previous day returns
    for lag in [1, 2, 3, 5]:
        data[f"Return_Lag_{lag}"] = data["Daily_Return"].shift(lag)

    # Intraday return
    data["Intraday_Return"] = (
        data["Close"] - data["Open"]
    ) / data["Open"]

    # High-low price range
    data["High_Low_Range"] = (
        data["High"] - data["Low"]
    ) / data["Close"]

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

    avg_loss = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (ema_12 - ema_26) / data["Close"]

    data["MACD_Signal"] = (
        data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    # Bollinger Band position
    rolling_mean = data["Close"].rolling(20).mean()

    rolling_std = data["Close"].rolling(20).std()

    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

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

    # Volume indicators
    data["Volume_Change"] = data["Volume"].pct_change()

    average_volume = data["Volume"].rolling(10).mean()

    data["Volume_Ratio"] = (
        data["Volume"] / average_volume
    )

    return data


# --------------------------------------------------
# 4. MAIN STOCK PREDICTION FUNCTION
# --------------------------------------------------

def predict_stock_price(symbol):

    # Fetch historical stock data
    data = fetch_stock_data(symbol)

    if data is None or data.empty:
        return {
            "error": "No stock data available"
        }

    data = data.copy()

    # Ensure chronological order
    data = data.sort_index()

    # Remove duplicate dates, if any
    data = data[~data.index.duplicated(keep="last")]

    # Generate technical indicators
    data = create_features(data)

    # Convert infinite values into missing values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Predict next trading day return
    data["Target"] = (
        data["Close"].shift(-1) / data["Close"]
    ) - 1

    # Feature columns
    feature_columns = [
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

    # Keep the latest row for live prediction
    latest_features = data[feature_columns].iloc[[-1]]

    if latest_features.isnull().any().any():
        return {
            "error": "Insufficient data to calculate technical indicators"
        }

    # Historical data with known targets
    training_data = data.dropna(
        subset=feature_columns + ["Target"]
    )

    if len(training_data) < 100:
        return {
            "error": "Insufficient historical data for model training"
        }

    X = training_data[feature_columns]
    y = training_data["Target"]

    # --------------------------------------------------
    # 5. CHRONOLOGICAL 80:20 TRAIN-TEST SPLIT
    # --------------------------------------------------

    split_index = int(len(training_data) * 0.8)

    if split_index < 10 or len(training_data) - split_index < 2:
        return {
            "error": "Insufficient data for train-test evaluation"
        }

    # Remove one observation before the test period.
    # Its target uses the first test-period closing price.

    X_train = X.iloc[:split_index - 1]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index - 1]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------
    # 6. WALK-FORWARD VALIDATION
    # --------------------------------------------------

    time_split = TimeSeriesSplit(
        n_splits=5,
        gap=1
    )

    validation_results = {}

    best_model_name = None
    best_validation_mae = float("inf")

    for model_name, config in MODEL_CONFIGS.items():

        fold_mae = []
        fold_rmse = []

        for train_indices, validation_indices in time_split.split(X_train):

            X_fold_train = X_train.iloc[train_indices]
            y_fold_train = y_train.iloc[train_indices]

            X_fold_validation = X_train.iloc[validation_indices]
            y_fold_validation = y_train.iloc[validation_indices]

            model = create_model(config)

            model.fit(
                X_fold_train,
                y_fold_train
            )

            predictions = model.predict(
                X_fold_validation
            )

            mae = mean_absolute_error(
                y_fold_validation,
                predictions
            ) * 100

            rmse = np.sqrt(
                mean_squared_error(
                    y_fold_validation,
                    predictions
                )
            ) * 100

            fold_mae.append(mae)
            fold_rmse.append(rmse)

        average_mae = float(np.mean(fold_mae))
        average_rmse = float(np.mean(fold_rmse))

        validation_results[model_name] = {
            "average_mae_percent": round(average_mae, 4),
            "average_rmse_percent": round(average_rmse, 4),
            "folds": len(fold_mae)
        }

        if average_mae < best_validation_mae:

            best_validation_mae = average_mae
            best_model_name = model_name

    # --------------------------------------------------
    # 7. FINAL MODEL EVALUATION
    # --------------------------------------------------

    best_config = MODEL_CONFIGS[best_model_name]

    evaluation_model = create_model(best_config)

    evaluation_model.fit(
        X_train,
        y_train
    )

    test_predictions = evaluation_model.predict(
        X_test
    )

    # Model performance
    mae = mean_absolute_error(
        y_test,
        test_predictions
    ) * 100

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    ) * 100

    r2 = r2_score(
        y_test,
        test_predictions
    )

    # Zero-return baseline
    baseline_predictions = np.zeros(len(y_test))

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    ) * 100

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    ) * 100

    # Directional accuracy
    directional_accuracy = np.mean(
        np.sign(test_predictions) == np.sign(y_test)
    ) * 100

    # --------------------------------------------------
    # 8. TRAIN FINAL MODEL USING ALL KNOWN TARGETS
    # --------------------------------------------------

    final_model = create_model(best_config)

    final_model.fit(
        X,
        y
    )

    # Predict next trading day return
    predicted_return = float(
        final_model.predict(latest_features)[0]
    )

    latest_close = float(data["Close"].iloc[-1])

    predicted_next_close = (
        latest_close * (1 + predicted_return)
    )

    # --------------------------------------------------
    # 9. RETURN RESULTS TO FASTAPI
    # --------------------------------------------------

    return {
        "stock": symbol.upper(),

        "latest_close": round(latest_close, 2),

        "predicted_return_percent": round(
            predicted_return * 100, 3
        ),

        "predicted_next_close": round(
            predicted_next_close, 2
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

            "best_model": best_model_name,

            "validation_results": validation_results,

            "training_samples": len(X_train),

            "testing_samples": len(X_test),

            "gap_observations": 1,

            "split": (
                "Chronological 80:20 with "
                "5-fold walk-forward validation "
                "and one-observation gap"
            )
        }
    }
