
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import time
from threading import Lock
from copy import deepcopy

from prediction import predict_stock_price

from database import (
    initialize_database,
    save_prediction,
    get_prediction_history,
    get_stock_prediction_history
)


app = FastAPI(title="EquityPredict")


# Initialize database when FastAPI starts
@app.on_event("startup")
def startup_event():
    initialize_database()


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.0.160:5173",
        "https://equitypredict.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prediction cache configuration
CACHE_DURATION = 60

prediction_cache = {}
cache_lock = Lock()


# Function to get cached prediction
def get_cached_prediction(symbol: str):

    symbol = symbol.strip().upper()
    current_time = time.monotonic()

    with cache_lock:

        cached_data = prediction_cache.get(symbol)

        # Return cached prediction if it is still valid
        if cached_data:
            cached_result, cached_time = cached_data

            if current_time - cached_time < CACHE_DURATION:
                return deepcopy(cached_result)

        # Generate a fresh prediction
        result = predict_stock_price(symbol)

        # Store prediction in cache
        prediction_cache[symbol] = (
            deepcopy(result),
            current_time
        )

        return result


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to EquityPredict!",
        "status": "Backend is running successfully"
    }


# Historical stock data endpoint
@app.get("/stock/{symbol}")
def get_stock_data(symbol: str):

    try:
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period="1mo")

        if data.empty:
            return {"error": "No stock data found"}

        return {
            "stock": symbol.upper(),
            "history": data.reset_index().to_dict(orient="records")
        }

    except Exception as error:
        return {
            "error": "Unable to fetch stock data: " + str(error)
        }


# Single stock price prediction endpoint
@app.get("/predict/{symbol}")
def predict_stock(symbol: str):

    try:
        result = get_cached_prediction(symbol)

        # Save successful prediction in database
        save_prediction(result)

        return result

    except ValueError as error:
        return {"error": str(error)}

    except Exception as error:
        return {
            "error": "Prediction failed: " + str(error)
        }


# Multiple stock prediction endpoint
@app.get("/predict-multiple")
def predict_multiple_stocks(symbols: str):

    stock_symbols = symbols.split(",")

    results = []

    for symbol in stock_symbols:

        symbol = symbol.strip().upper()

        if not symbol:
            continue

        try:
            result = get_cached_prediction(symbol)

            # Save successful prediction in database
            save_prediction(result)

            results.append(result)

        except Exception as error:
            results.append({
                "stock": symbol,
                "error": str(error)
            })

    return {
        "total_stocks": len(results),
        "results": results
    }


# Retrieve all prediction history
@app.get("/predictions/history")
def prediction_history(
    limit: int = Query(default=50, ge=1, le=500)
):

    return {
        "total_records": len(get_prediction_history(limit)),
        "predictions": get_prediction_history(limit)
    }


# Retrieve prediction history for one stock
@app.get("/predictions/history/{symbol}")
def stock_prediction_history(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500)
):

    predictions = get_stock_prediction_history(
        symbol,
        limit
    )

    return {
        "stock": symbol.strip().upper(),
        "total_records": len(predictions),
        "predictions": predictions
    }
