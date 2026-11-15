from prediction import predict_stock_price

result = predict_stock_price("AAPL")

print("\nSTOCK PREDICTION RESULTS")
print("------------------------")

print("Stock:", result["stock"])
print("Latest Closing Price:", result["latest_close"])
print("Predicted Return:", result["predicted_return_percent"], "%")
print("Predicted Next Closing Price:", result["predicted_next_close"])