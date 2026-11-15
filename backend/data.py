import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol):

    stock = yf.Ticker(symbol)

    data = stock.history(period="5y")

    if data.empty:
        raise ValueError("No stock data available")

    data = data.dropna()

    data = data.reset_index()

    return data