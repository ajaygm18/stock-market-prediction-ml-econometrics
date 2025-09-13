import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
PROCESSED_DATA_DIR = '../../1_Data_Files/02_Processed_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'feature_engineered_data.csv')
RESULTS_DIR = '../../3_Model_Outputs'

# Select a ticker for the case study
TICKER_TO_MODEL = 'TSLA'
TARGET_VARIABLE = 'Close_normalized'

def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Created directory: {RESULTS_DIR}")

def load_processed_data():
    """Loads the final processed and feature-engineered dataset."""
    print(f"Loading processed data for ticker: {TICKER_TO_MODEL}")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'])
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy()
    ticker_df.sort_values('Date', inplace=True)
    ticker_df.set_index('Date', inplace=True)
    print(f"Loaded {len(ticker_df)} records for {TICKER_TO_MODEL}")
    return ticker_df

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and returns performance metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    print(f"{model_name} Performance -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def train_evaluate_arima(train_data, test_data):
    """Trains an ARIMA model."""
    print("\n--- Training ARIMA Model ---")
    model = ARIMA(train_data, order=(5,1,0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return evaluate_model(test_data, predictions, "ARIMA")

def create_sequences(data, n_steps):
    """Converts time-series data into sequences for LSTM."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_evaluate_lstm(train_data, test_data, n_steps=20):  # Reduced steps for faster training
    """Trains a simplified LSTM model."""
    print("\n--- Training LSTM Model ---")
    
    # Prepare sequences
    X_train, y_train = create_sequences(train_data.values, n_steps)
    X_test, y_test = create_sequences(test_data.values, n_steps)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data for LSTM training")
        return {'Model': 'LSTM', 'RMSE': 999, 'MAE': 999, 'MAPE': 999}
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build simplified model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
        LSTM(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train with reduced epochs for speed
    print("Training LSTM... this may take a moment.")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Predict
    predictions = model.predict(X_test, verbose=0)
    
    return evaluate_model(y_test, predictions, "LSTM")

if __name__ == "__main__":
    print("=== Stock Market Prediction Model Training ===")
    
    # Setup
    create_output_directory()
    
    # Load data
    data = load_processed_data()
    
    # Split data (80% train, 20% test)
    split_index = int(0.8 * len(data))
    train_data = data[TARGET_VARIABLE][:split_index]
    test_data = data[TARGET_VARIABLE][split_index:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Train models
    performance_results = []
    
    # ARIMA Model
    try:
        arima_results = train_evaluate_arima(train_data, test_data)
        performance_results.append(arima_results)
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        performance_results.append({'Model': 'ARIMA', 'RMSE': 999, 'MAE': 999, 'MAPE': 999})
    
    # LSTM Model
    try:
        lstm_results = train_evaluate_lstm(train_data, test_data)
        performance_results.append(lstm_results)
    except Exception as e:
        print(f"LSTM training failed: {e}")
        performance_results.append({'Model': 'LSTM', 'RMSE': 999, 'MAE': 999, 'MAPE': 999})
    
    # Save results
    results_df = pd.DataFrame(performance_results)
    results_file = os.path.join(RESULTS_DIR, 'model_performance_results.csv')
    results_df.to_csv(results_file, index=False)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Results saved to: {results_file}")
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))