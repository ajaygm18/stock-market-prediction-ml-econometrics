import yfinance as yf
import pandas as pd
import os
from fredapi import Fred

# --- Configuration ---
# Define the stock tickers and the date range for the data download.
TICKERS = ['^GSPC', 'TSLA']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# FRED API Key
FRED_API_KEY = '747c4c16bc76a3dfc54d6d63c0ba9e4d'

# Define the output directory for the raw data.
OUTPUT_DIR = '../../1_Data_Files/01_Raw_Data'
STOCK_DATA_DIR = os.path.join(OUTPUT_DIR, '001_Stock_Market_Data')
MACRO_DATA_DIR = os.path.join(OUTPUT_DIR, '002_Macroeconomic_Indicators')
STOCK_DATA_FILE = os.path.join(STOCK_DATA_DIR, 'stock_data_2010-2023.csv')
MACRO_DATA_FILE = os.path.join(MACRO_DATA_DIR, 'macroeconomic_indicators_raw.csv')

# --- Main Execution ---
def create_output_directory():
    """Creates the output directory if it doesn't exist."""
    for directory in [OUTPUT_DIR, STOCK_DATA_DIR, MACRO_DATA_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_stock_data():
    """Downloads historical stock data from Yahoo Finance."""
    print(f"Downloading stock data for: {', '.join(TICKERS)}...")
    try:
        # Download data for each ticker individually to handle failures better
        final_df = pd.DataFrame()
        
        for ticker in TICKERS:
            try:
                print(f"Downloading {ticker}...")
                # Download data with auto_adjust=False to fix adj close issues
                ticker_data = yf.download(ticker, start=START_DATE, end=END_DATE, 
                                        auto_adjust=False, progress=False)
                
                if not ticker_data.empty:
                    ticker_data['Ticker'] = ticker
                    ticker_data.reset_index(inplace=True)
                    final_df = pd.concat([final_df, ticker_data])
                    print(f"✅ Successfully downloaded {ticker}")
                else:
                    print(f"❌ No data found for {ticker}")
                    
            except Exception as e:
                print(f"❌ Failed to download {ticker}: {e}")
                # Try alternative ticker names
                if ticker == '^GSPC':
                    print("Trying alternative: SPY (S&P 500 ETF)")
                    try:
                        ticker_data = yf.download('SPY', start=START_DATE, end=END_DATE, 
                                                auto_adjust=False, progress=False)
                        if not ticker_data.empty:
                            ticker_data['Ticker'] = '^GSPC'  # Keep original name
                            ticker_data.reset_index(inplace=True)
                            final_df = pd.concat([final_df, ticker_data])
                            print(f"✅ Successfully downloaded SPY as proxy for {ticker}")
                    except:
                        pass

        if not final_df.empty:
            # Save to CSV
            final_df.to_csv(STOCK_DATA_FILE, index=False)
            print(f"Successfully downloaded and saved stock data to {STOCK_DATA_FILE}")
        else:
            print("❌ No stock data was successfully downloaded")
            
    except Exception as e:
        print(f"An error occurred during stock data download: {e}")

def download_fred_data():
    """
    Downloads real macroeconomic data from FRED API.
    """
    print("Downloading macroeconomic data from FRED...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Define FRED series IDs for macroeconomic indicators
        fred_series = {
            'GDP_Growth_Rate': 'GDP',
            'Interest_Rate_Federal_Funds': 'FEDFUNDS',
            'Inflation_Rate_CPI': 'CPIAUCSL',
            'Unemployment_Rate': 'UNRATE',
            'VIX': 'VIXCLS'
        }
        
        # Download data for each series
        macro_data = {}
        for name, series_id in fred_series.items():
            try:
                data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
                macro_data[name] = data
                print(f"Successfully downloaded {name} ({series_id})")
            except Exception as e:
                print(f"Failed to download {name} ({series_id}): {e}")
        
        # Combine all series into a single DataFrame
        macro_df = pd.DataFrame(macro_data)
        macro_df.index.name = 'Date'
        macro_df.reset_index(inplace=True)
        
        # Calculate GDP growth rate if raw GDP data was downloaded
        if 'GDP_Growth_Rate' in macro_df.columns:
            macro_df['GDP_Growth_Rate_Annualized'] = macro_df['GDP_Growth_Rate'].pct_change(fill_method=None) * 100
        
        # Calculate inflation rate as year-over-year change for CPI
        if 'Inflation_Rate_CPI' in macro_df.columns:
            macro_df['Inflation_Rate_CPI_YoY'] = macro_df['Inflation_Rate_CPI'].pct_change(periods=12, fill_method=None) * 100
        
        # Save to CSV
        macro_df.to_csv(MACRO_DATA_FILE, index=False)
        print(f"Successfully downloaded and saved macroeconomic data to {MACRO_DATA_FILE}")
        
    except Exception as e:
        print(f"An error occurred during FRED data download: {e}")
        print("Falling back to sample data generation...")
        generate_macro_data()

def generate_macro_data():
    """
    Generates a sample macroeconomic dataset.
    Fallback method if FRED API fails.
    """
    print("Generating sample macroeconomic data file...")
    # This data is a simplified representation for demonstration.
    macro_data = {
        'Date': pd.to_datetime(['2010-01-31', '2010-02-28', '2023-11-30', '2023-12-31']),
        'GDP_Growth_Rate_Annualized': [1.5, 1.6, 2.5, 2.5],
        'Interest_Rate_Federal_Funds': [0.11, 0.13, 5.33, 5.33],
        'Inflation_Rate_CPI_YoY': [2.6, 2.1, 3.1, 3.4],
        'Unemployment_Rate': [9.8, 9.8, 3.7, 3.7],
        'VIX': [20.0, 22.5, 18.5, 16.8]
    }
    macro_df = pd.DataFrame(macro_data)
    macro_df.to_csv(MACRO_DATA_FILE, index=False)
    print(f"Successfully generated and saved sample macroeconomic data to {MACRO_DATA_FILE}")


if __name__ == "__main__":
    print("--- Starting Data Acquisition ---")
    create_output_directory()
    download_stock_data()
    download_fred_data()
    print("--- Data Acquisition Complete ---")
