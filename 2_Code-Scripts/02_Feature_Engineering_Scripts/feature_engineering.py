import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
import os
import numpy as np

# --- Configuration ---
# Input files from the data processing step.
PROCESSED_DATA_DIR = '../../1_Data_Files/02_Processed_Data'
PROCESSED_INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_stock_data_2010-2023.csv')

# Output file for the feature-engineered data.
FEATURE_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'feature_engineered_data.csv')

# --- Helper Functions ---
def create_output_directory():
    """Creates the output directory for processed data if it doesn't exist."""
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

def load_processed_data():
    """Loads the already processed data."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_INPUT_FILE, parse_dates=['Date'])
    print(f"Loaded {len(df)} records")
    return df

def create_additional_features(df):
    """Creates additional features for machine learning models."""
    print("Creating additional features...")
    
    # Set Date as index temporarily for time-series operations
    df_temp = df.set_index('Date')
    
    # Create lagged features for the normalized close price
    for lag in [1, 3, 5, 10]:
        col_name = f'close_lag_{lag}'
        df_temp[col_name] = df_temp.groupby('Ticker')['Close_normalized'].shift(lag)
    
    # Create price change features
    df_temp['price_change_1d'] = df_temp.groupby('Ticker')['Close_normalized'].diff()
    df_temp['price_change_5d'] = df_temp.groupby('Ticker')['Close_normalized'].diff(5)
    
    # Create volatility measures
    df_temp['return_volatility_10d'] = df_temp.groupby('Ticker')['daily_return_normalized'].transform(
        lambda x: x.rolling(window=10).std()
    )
    
    # Reset index
    df_temp.reset_index(inplace=True)
    
    return df_temp

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Feature Engineering ---")
    
    # 1. Load processed data
    data_df = load_processed_data()
    
    # 2. Create additional features
    data_df = create_additional_features(data_df)
    
    # 3. Final Cleaning
    # Drop rows with NaN values that result from lagged features and rolling window calculations
    data_df.dropna(inplace=True)
    print("Dropped rows with NaN values from feature creation.")

    # 4. Save the final feature-engineered data
    data_df.to_csv(FEATURE_OUTPUT_FILE, index=False)
    print(f"--- Feature Engineering Complete ---")
    print(f"Final dataset with engineered features saved to {FEATURE_OUTPUT_FILE}")
    print(f"Dataset contains {data_df.shape[0]} rows and {data_df.shape[1]} columns.")
