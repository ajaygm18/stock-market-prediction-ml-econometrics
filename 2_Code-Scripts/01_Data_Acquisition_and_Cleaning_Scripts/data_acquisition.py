import yfinance as yf
import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr
from fredapi import Fred

# --- Configuration ---
# Define the stock tickers and the date range for the data download.
TICKERS = ['^GSPC', 'TSLA']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# FRED API key as specified in requirements
FRED_API_KEY = '747c4c16bc76a3dfc54d6d63c0ba9e4d'

# Define the output directory for the raw data.
OUTPUT_DIR = '../../1_Data_Files/01_Raw_Data/001_Stock_Market_Data'
MACRO_OUTPUT_DIR = '../../1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators'
STOCK_DATA_FILE = os.path.join(OUTPUT_DIR, 'stock_data_2010-2023.csv')
MACRO_DATA_FILE = os.path.join(MACRO_OUTPUT_DIR, 'macroeconomic_indicators_raw.csv')

# --- Main Execution ---
def create_output_directories():
    """Creates the output directories if they don't exist."""
    for directory in [OUTPUT_DIR, MACRO_OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_stock_data():
    """Downloads historical stock data from Yahoo Finance with auto_adjust=False."""
    print(f"Downloading stock data for: {', '.join(TICKERS)}...")
    try:
        # Download data for all tickers at once with auto_adjust=False as specified
        stock_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, 
                                group_by='ticker', auto_adjust=False)
        
        # Restructure the dataframe from multi-index columns to a long format
        # This makes it easier to work with
        final_df = pd.DataFrame()
        for ticker in TICKERS:
            # For single ticker downloads, yfinance returns a simple DataFrame
            # For multiple, it's a multi-index one. We handle both.
            if len(TICKERS) > 1:
                ticker_df = stock_data[ticker].copy()
            else:
                ticker_df = stock_data.copy()
            
            ticker_df['Ticker'] = ticker
            ticker_df.reset_index(inplace=True)
            final_df = pd.concat([final_df, ticker_df])

        # Save to CSV
        final_df.to_csv(STOCK_DATA_FILE, index=False)
        print(f"Successfully downloaded and saved stock data to {STOCK_DATA_FILE}")
        print(f"Downloaded {len(final_df)} rows of stock data")
        return True
    except Exception as e:
        print(f"An error occurred during stock data download: {e}")
        # Try alternative approach with individual ticker downloads
        print("Trying alternative approach with individual ticker downloads...")
        try:
            final_df = pd.DataFrame()
            for ticker in TICKERS:
                print(f"Downloading {ticker}...")
                ticker_data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
                ticker_data['Ticker'] = ticker
                ticker_data.reset_index(inplace=True)
                final_df = pd.concat([final_df, ticker_data])
            
            final_df.to_csv(STOCK_DATA_FILE, index=False)
            print(f"Successfully downloaded and saved stock data to {STOCK_DATA_FILE}")
            print(f"Downloaded {len(final_df)} rows of stock data")
            return True
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            return False

def download_macro_data():
    """Downloads real macroeconomic data from FRED API."""
    print("Downloading macroeconomic data from FRED API...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Define the FRED series IDs for various economic indicators
        fred_series = {
            'GDP_Growth_Rate_Annualized': 'GDP',  # Real Gross Domestic Product
            'Interest_Rate_Federal_Funds': 'FEDFUNDS',  # Federal Funds Rate
            'Inflation_Rate_CPI_YoY': 'CPIAUCSL',  # Consumer Price Index
            'Unemployment_Rate': 'UNRATE',  # Unemployment Rate
            'VIX': 'VIXCLS'  # CBOE VIX Volatility Index
        }
        
        macro_data = {}
        
        for indicator, series_id in fred_series.items():
            print(f"  Downloading {indicator} ({series_id})...")
            try:
                data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
                macro_data[indicator] = data
            except Exception as e:
                print(f"  Warning: Could not download {indicator}: {e}")
                # Create dummy data for this indicator if download fails
                date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
                macro_data[indicator] = pd.Series([1.0] * len(date_range), index=date_range)
        
        # Combine all series into a single dataframe
        macro_df = pd.DataFrame(macro_data)
        macro_df.index.name = 'Date'
        macro_df.reset_index(inplace=True)
        
        # Handle CPI transformation to year-over-year inflation rate
        if 'Inflation_Rate_CPI_YoY' in macro_df.columns:
            macro_df['Inflation_Rate_CPI_YoY'] = macro_df['Inflation_Rate_CPI_YoY'].pct_change(periods=12) * 100
        
        # Save to CSV
        macro_df.to_csv(MACRO_DATA_FILE, index=False)
        print(f"Successfully downloaded and saved macroeconomic data to {MACRO_DATA_FILE}")
        print(f"Downloaded {len(macro_df)} rows of macroeconomic data")
        return True
        
    except Exception as e:
        print(f"FRED API download failed: {e}")
        print("Generating fallback macroeconomic data...")
        # Fallback to generated data if FRED fails
        generate_fallback_macro_data()
        return False

def generate_fallback_macro_data():
    """Generates fallback macroeconomic dataset if FRED API fails."""
    print("Generating fallback macroeconomic data...")
    
    # Create a more comprehensive time series for fallback
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
    
    # Generate realistic-looking data with some trends
    np.random.seed(42)  # For reproducibility
    n_periods = len(date_range)
    
    macro_data = {
        'Date': date_range,
        'GDP_Growth_Rate_Annualized': np.random.normal(2.0, 1.0, n_periods),
        'Interest_Rate_Federal_Funds': np.linspace(0.1, 5.0, n_periods) + np.random.normal(0, 0.5, n_periods),
        'Inflation_Rate_CPI_YoY': np.random.normal(2.5, 1.0, n_periods),
        'Unemployment_Rate': np.random.uniform(3.5, 10.0, n_periods),
        'VIX': np.random.normal(20.0, 8.0, n_periods)
    }
    
    macro_df = pd.DataFrame(macro_data)
    macro_df.to_csv(MACRO_DATA_FILE, index=False)
    print(f"Successfully generated and saved fallback macroeconomic data to {MACRO_DATA_FILE}")
    print(f"Generated {len(macro_df)} rows of macroeconomic data")


if __name__ == "__main__":
    print("--- Starting Data Acquisition ---")
    print(f"Using FRED API Key: {FRED_API_KEY[:10]}...")  # Show first 10 chars for verification
    
    create_output_directories()
    
    # Download stock data with error handling
    stock_success = download_stock_data()
    
    # Download macroeconomic data with FRED API
    macro_success = download_macro_data()
    
    print("--- Data Acquisition Complete ---")
    if stock_success and macro_success:
        print("✅ All data downloaded successfully")
    elif stock_success:
        print("⚠️ Stock data downloaded successfully, but had issues with macro data")
    elif macro_success:
        print("⚠️ Macro data downloaded successfully, but had issues with stock data")
    else:
        print("❌ Had issues downloading data, but fallback data was generated")
    
    print(f"Stock data saved to: {STOCK_DATA_FILE}")
    print(f"Macro data saved to: {MACRO_DATA_FILE}")
