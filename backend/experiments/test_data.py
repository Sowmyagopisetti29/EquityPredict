from data import fetch_stock_data

df = fetch_stock_data("AAPL")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nTotal Records:", len(df))