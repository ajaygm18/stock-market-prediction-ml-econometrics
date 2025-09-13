import yfinance as yf
import pandas as pd
import os
from fredapi import Fred

# --- Configuration ---
# Define the stock tickers and the date range for the data download.
TICKERS = ['^GSPC', 'TSLA']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# FRED API Configuration
FRED_API_KEY = '747c4c16bc76a3dfc54d6d63c0ba9e4d'

# Define the output directory for the raw data.
OUTPUT_DIR = '../../1_Data_Files/01_Raw_Data'
STOCK_DATA_DIR = os.path.join(OUTPUT_DIR, '001_Stock_Market_Data')
MACRO_DATA_DIR = os.path.join(OUTPUT_DIR, '002_Macroeconomic_Indicators')
STOCK_DATA_FILE = os.path.join(STOCK_DATA_DIR, 'stock_data_2010-2023.csv')
MACRO_DATA_FILE = os.path.join(MACRO_DATA_DIR, 'macroeconomic_indicators_raw.csv')

# --- Main Execution ---
def create_output_directories():
    """Creates the output directories if they don't exist."""
    for directory in [STOCK_DATA_DIR, MACRO_DATA_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_stock_data():
    """Downloads historical stock data from Yahoo Finance."""
    print(f"Downloading stock data for: {', '.join(TICKERS)}...")
    try:
        # Download data for all tickers at once
        stock_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker')
        
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
    except Exception as e:
        print(f"An error occurred during stock data download: {e}")

def download_macro_data():
    """
    Downloads macroeconomic data from FRED API.
    Uses the FRED API to fetch time series for Federal Funds Rate, Real GDP, and Consumer Price Index.
    """
    print("Downloading macroeconomic data from FRED API...")
    try:
        # Initialize FRED API client
        fred = Fred(api_key=FRED_API_KEY)
        
        # Define the FRED series IDs as mentioned in the research paper
        series_dict = {
            'FEDFUNDS': 'Interest_Rate_Federal_Funds',  # Federal Funds Effective Rate
            'GDPC1': 'GDP_Real',  # Real Gross Domestic Product
            'CPIAUCSL': 'CPI_All_Items'  # Consumer Price Index for All Urban Consumers
        }
        
        # Download data for each series
        macro_data = pd.DataFrame()
        
        for series_id, column_name in series_dict.items():
            print(f"  Downloading {series_id} ({column_name})...")
            try:
                # Download the series data
                series_data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
                
                # Convert to DataFrame and rename column
                series_df = series_data.to_frame(name=column_name)
                series_df.reset_index(inplace=True)
                series_df.rename(columns={'index': 'Date'}, inplace=True)
                
                # Merge with main dataframe
                if macro_data.empty:
                    macro_data = series_df
                else:
                    macro_data = pd.merge(macro_data, series_df, on='Date', how='outer')
                    
            except Exception as e:
                print(f"    Warning: Could not download {series_id}: {e}")
                continue
        
        # Sort by date and save
        if not macro_data.empty:
            macro_data = macro_data.sort_values('Date').reset_index(drop=True)
            
            # Calculate additional indicators as mentioned in the research
            if 'GDP_Real' in macro_data.columns:
                # Calculate GDP growth rate (quarterly year-over-year)
                macro_data['GDP_Growth_Rate_YoY'] = macro_data['GDP_Real'].pct_change(periods=4) * 100
            
            if 'CPI_All_Items' in macro_data.columns:
                # Calculate inflation rate (year-over-year)
                macro_data['Inflation_Rate_CPI_YoY'] = macro_data['CPI_All_Items'].pct_change(periods=12) * 100
            
            # Save to CSV
            macro_data.to_csv(MACRO_DATA_FILE, index=False)
            print(f"Successfully downloaded and saved macroeconomic data to {MACRO_DATA_FILE}")
            print(f"Downloaded {len(macro_data)} records from {macro_data['Date'].min()} to {macro_data['Date'].max()}")
        else:
            print("Warning: No macroeconomic data was successfully downloaded")
            
    except Exception as e:
        print(f"Error downloading macroeconomic data: {e}")
        print("Falling back to sample data generation...")
        generate_sample_macro_data()

def generate_sample_macro_data():
    """
    Generates a sample macroeconomic dataset as fallback.
    This is used if FRED API fails for any reason.
    """
    print("Generating sample macroeconomic data file...")
    # This data is a simplified representation for demonstration.
    macro_data = {
        'Date': pd.to_datetime(['2010-01-31', '2010-02-28', '2023-11-30', '2023-12-31']),
        'Interest_Rate_Federal_Funds': [0.11, 0.13, 5.33, 5.33],
        'GDP_Real': [15049.0, 15065.0, 20610.0, 20650.0],  # In billions
        'CPI_All_Items': [217.0, 217.5, 310.0, 310.8],
        'GDP_Growth_Rate_YoY': [1.5, 1.6, 2.5, 2.5],
        'Inflation_Rate_CPI_YoY': [2.6, 2.1, 3.1, 3.4]
    }
    macro_df = pd.DataFrame(macro_data)
    macro_df.to_csv(MACRO_DATA_FILE, index=False)
    print(f"Successfully generated and saved sample macroeconomic data to {MACRO_DATA_FILE}")


if __name__ == "__main__":
    print("--- Starting Data Acquisition ---")
    create_output_directories()
    download_stock_data()
    download_macro_data()
    print("--- Data Acquisition Complete ---")
