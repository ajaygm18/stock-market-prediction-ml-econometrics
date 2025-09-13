import yfinance as yf
import pandas as pd
import numpy as np

# 1. Define tickers and date range
tickers = ['^GSPC', 'TSLA']  # Updated to match main data acquisition script
start_date = '2010-01-01'
end_date = '2023-12-31'

# yfinance's end date is exclusive, so we set it to the day after our target end date
# Note: The end date for yfinance is exclusive, so '2024-01-01' correctly includes data up to '2023-12-31'.
# Use auto_adjust=False as specified in requirements
try:
    data_raw = yf.download(tickers, start=start_date, end='2024-01-01', auto_adjust=False)
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Trying alternative approach with individual downloads...")
    data_raw = pd.DataFrame()
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        ticker_data = yf.download(ticker, start=start_date, end='2024-01-01', auto_adjust=False)
        ticker_data['Ticker'] = ticker
        data_raw = pd.concat([data_raw, ticker_data])

# 2. Process the multi-level column index into a tidy format
# We stack at level=1 (the ticker level) and rename the axes before resetting the index.
if isinstance(data_raw.columns, pd.MultiIndex):
    data_processed = data_raw.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
else:
    # Handle case where we downloaded individual tickers
    data_processed = data_raw.reset_index()

# 3. Reorder columns to match a standard format (optional but good practice)
expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
available_columns = [col for col in expected_columns if col in data_processed.columns]
stock_data_df = data_processed[available_columns]

# 4. Save to CSV
stock_data_df.to_csv('stock_data_2010-2023.csv', index=False)

print(f"Data downloaded and saved successfully to 'stock_data_2010-2023.csv'")
print(f"Downloaded {len(stock_data_df)} rows for {len(tickers)} tickers")
print(f"Date range: {stock_data_df['Date'].min()} to {stock_data_df['Date'].max()}")

# Display first few rows to verify
print("\nFirst few rows:")
print(stock_data_df.head())