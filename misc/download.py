import yfinance as yf
import pickle

print("import ok")
# Define a list of stock symbols to download
stock_symbols = ["AAPL", "MSFT", "GOOG"]

# Set start and end dates for the historical data
start_date = "2010-01-01"
end_date = "2021-12-31"

# Download the stock data using yfinance
stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

# Save the downloaded data as a pickle file
with open("stock_data.pickle", "wb") as f:
    pickle.dump(stock_data, f)
print("done")

