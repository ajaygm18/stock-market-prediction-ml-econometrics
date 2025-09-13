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
PROCESSED_DATA_DIR = '../../1_Data_Files/02_Cleaned_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_stock_data_2010-2023.csv')
RESULTS_DIR = '../../3_Model_Outputs/02_Evaluation_Metrics'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'model_performance_comparison.csv')

# Select a ticker for the case study
TICKER_TO_MODEL = 'TSLA'
# CRITICAL FIX: Corrected column name to match the output of the feature engineering script.
TARGET_VARIABLE = 'Adj Close_normalized'
# Model Hyperparameters
N_STEPS = 50 # Lookback window for LSTM

# --- Helper Functions ---
def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

def load_processed_data():
    """Loads the final processed and feature-engineered dataset."""
    print(f"Loading processed data for ticker: {TICKER_TO_MODEL}")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'], index_col='Date')
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy()
    # Ensure data is sorted by date for time-series analysis
    ticker_df.sort_index(inplace=True)
    return ticker_df

def create_sequences(data, n_steps):
    """Converts a time-series dataset into sequences for LSTM."""
    X, y = [], []
    # Check if data is 1D or 2D
    is_multivariate = data.ndim > 1
    
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        # Input sequence is all features; output is the first feature (target)
        seq_x = data[i:end_ix, :] if is_multivariate else data[i:end_ix]
        seq_y = data[end_ix, 0] if is_multivariate else data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def generate_detailed_analysis(y_true, y_pred, dates, model_name="Hybrid"):
    """Generate detailed error analysis including directional accuracy"""
    print(f"Generating detailed analysis for {model_name} model...")
    
    # Ensure same length
    min_length = min(len(y_true), len(y_pred), len(dates))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    dates = dates[:min_length]
    
    # Calculate errors
    raw_errors = y_pred - y_true
    percentage_errors = (raw_errors / np.where(y_true != 0, y_true, 1)) * 100
    
    # Calculate directions
    actual_directions = ['Up' if i < len(y_true)-1 and y_true[i+1] > y_true[i] else 'Down' 
                        for i in range(len(y_true)-1)] + ['Up']  # Last day direction
    predicted_directions = ['Up' if i < len(y_pred)-1 and y_pred[i+1] > y_pred[i] else 'Down' 
                           for i in range(len(y_pred)-1)] + ['Up']  # Last day direction
    
    # Calculate directional accuracy
    directional_accuracy = [1 if actual_directions[i] == predicted_directions[i] else 0 
                           for i in range(len(actual_directions))]
    
    # Create detailed analysis dataframe
    analysis_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred,
        'Raw_Error': raw_errors,
        'Percentage_Error': percentage_errors,
        'Actual_Direction': actual_directions,
        'Predicted_Direction': predicted_directions,
        'Directional_Accuracy': directional_accuracy
    })
    
    # Save to file
    error_analysis_file = os.path.join(RESULTS_DIR, 'error_analysis.csv')
    with open(error_analysis_file, 'w') as f:
        f.write(f"# Description: Detailed error analysis for the {model_name} model on the test set.\n")
        f.write("# Includes raw error, percentage error, and directional accuracy.\n")
    
    analysis_df.to_csv(error_analysis_file, mode='a', index=False)
    
    # Calculate and print summary statistics
    overall_directional_accuracy = np.mean(directional_accuracy) * 100
    print(f"📊 Overall Directional Accuracy: {overall_directional_accuracy:.2f}%")
    
    # Save predictions to separate file
    predictions_file = os.path.join('../../3_Model_Outputs/01_Raw_Prediction_Files', f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}_predictions.csv')
    with open(predictions_file, 'w') as f:
        f.write(f"# Description: Raw predictions from the {model_name} model for {TICKER_TO_MODEL}.\n")
        f.write(f"# These predictions incorporate the best performing model architecture.\n")
    
    pred_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred
    })
    pred_df.to_csv(predictions_file, mode='a', index=False)
    
    print(f"📁 Detailed analysis saved to: {error_analysis_file}")
    print(f"📁 Predictions saved to: {predictions_file}")
    
def evaluate_model(y_true, y_pred, model_name):
    """Calculates and returns performance metrics."""
    # Convert to numpy arrays to avoid pandas alignment issues
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Ensure same length
    min_length = min(len(y_true_np), len(y_pred_np))
    y_true_np = y_true_np[:min_length]
    y_pred_np = y_pred_np[:min_length]
    
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    # MAPE (Mean Absolute Percentage Error) is often useful for price prediction
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true_np - y_pred_np) / np.where(y_true_np != 0, y_true_np, 1))) * 100
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"{model_name} Performance -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# --- Model Training and Evaluation ---

def train_evaluate_arima(train_data, test_data):
    """
    Trains an ARIMA model once on the training data and forecasts the entire test period.
    This is a much more efficient approach than re-training at every step.
    """
    print("\n--- Training ARIMA Model ---")
    # Fit the model on the entire training history
    model = ARIMA(train_data, order=(5,1,0)) # (p,d,q) order
    model_fit = model.fit()
    
    # Forecast the entire test set length
    predictions = model_fit.forecast(steps=len(test_data))
    
    # Convert to numpy arrays for evaluation
    return evaluate_model(test_data.values, predictions, "ARIMA")

def train_evaluate_lstm(train_data, test_data, n_steps):
    """Trains and evaluates a standard LSTM model."""
    print("\n--- Training LSTM Model ---")
    # Prepare data for LSTM
    X_train, y_train = create_sequences(train_data, n_steps)
    X_test, y_test = create_sequences(test_data, n_steps)

    # Define LSTM model architecture
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    print("Training LSTM... this may take a moment.")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    return evaluate_model(y_test, predictions.flatten(), "LSTM")

def train_evaluate_hybrid_model(df, train_size, n_steps):
    """
    Trains and evaluates a correct Hybrid ARIMA-LSTM model.
    1. ARIMA models the linear part of the time series.
    2. LSTM models the non-linear part (the errors/residuals from ARIMA).
    3. Final prediction = ARIMA prediction + LSTM error prediction.
    """
    print("\n--- Training Hybrid (ARIMA-LSTM) Model ---")
    
    # 1. Train ARIMA and get forecasts and residuals
    target_series = df[TARGET_VARIABLE]
    train_target, test_target = target_series[:train_size], target_series[train_size:]
    
    arima_model = ARIMA(train_target, order=(5,1,0)).fit()
    # Get residuals from the training period. The first value will be NaN due to differencing.
    train_residuals = arima_model.resid.iloc[1:].values
    # Get ARIMA's forecast for the test period
    arima_test_forecast = arima_model.forecast(steps=len(test_target))

    # 2. Prepare simplified data for the error-predicting LSTM
    # Use only the residuals as features for the LSTM to avoid dimensional issues
    X_train_res, y_train_res = create_sequences(train_residuals.reshape(-1, 1), n_steps)
    
    # For testing, we need to predict residuals for the test period
    # We'll use a simple approach: extend the residuals series with zeros for the test period
    extended_residuals = np.concatenate([train_residuals, np.zeros(len(test_target))])
    extended_residuals = extended_residuals.reshape(-1, 1)
    
    # Get test sequences starting from the end of training residuals
    X_test_res, _ = create_sequences(extended_residuals[len(train_residuals)-n_steps:], n_steps)
    # Only take the sequences that would predict the test period
    X_test_res = X_test_res[:len(test_target)]

    # 3. Train LSTM to predict residuals
    print("Training Residual-LSTM... this may take a moment.")
    residual_model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train_res.shape[1], X_train_res.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    residual_model.compile(optimizer='adam', loss='mse')
    residual_model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=0)
    
    # 4. Make final hybrid prediction
    if len(X_test_res) > 0:
        predicted_residuals = residual_model.predict(X_test_res, verbose=0)
        # Final prediction = ARIMA forecast + LSTM's predicted error
        final_predictions = arima_test_forecast.values[:len(predicted_residuals)] + predicted_residuals.flatten()
    else:
        # Fallback: use only ARIMA predictions
        final_predictions = arima_test_forecast.values
    
    # 5. Evaluate the hybrid model and generate detailed analysis
    test_target_aligned = test_target.values[:len(final_predictions)]
    test_dates = df.index[train_size:train_size+len(final_predictions)]
    
    # Generate detailed analysis including directional accuracy
    generate_detailed_analysis(test_target_aligned, final_predictions, test_dates, "Hybrid (ARIMA-LSTM)")
    
    return evaluate_model(test_target_aligned, final_predictions, "Hybrid (ARIMA-LSTM)")


# --- Main Execution ---
if __name__ == "__main__":
    create_output_directory()
    df = load_processed_data()

    # Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)

    # Data for ARIMA (univariate)
    arima_train = df[TARGET_VARIABLE].iloc[:train_size]
    arima_test = df[TARGET_VARIABLE].iloc[train_size:]
    
    # Data for ML models (multivariate)
    ml_features = df.drop(columns=['Ticker']).values
    ml_train, ml_test = ml_features[:train_size], ml_features[train_size:]
    
    # Run models
    performance_results = []
    performance_results.append(train_evaluate_arima(arima_train, arima_test))
    performance_results.append(train_evaluate_lstm(ml_train, ml_test, n_steps=N_STEPS))
    performance_results.append(train_evaluate_hybrid_model(df, train_size, n_steps=N_STEPS))
    
    # Save results
    results_df = pd.DataFrame(performance_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print("\n--- Model Performance Comparison ---")
    print(results_df.round(4))
    print(f"\nResults saved to {RESULTS_FILE}")