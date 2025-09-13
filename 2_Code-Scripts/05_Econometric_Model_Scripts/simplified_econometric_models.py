import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
PROCESSED_DATA_DIR = '../../1_Data_Files/02_Processed_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'feature_engineered_data.csv')
RESULTS_DIR = '../../3_Model_Outputs'
ECONOMETRIC_RESULTS_DIR = os.path.join(RESULTS_DIR, 'Econometric_Models')

TICKER_TO_MODEL = 'TSLA'
TARGET_VARIABLE = 'Close_normalized'
RETURNS_VARIABLE = 'daily_return_normalized'

def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(ECONOMETRIC_RESULTS_DIR):
        os.makedirs(ECONOMETRIC_RESULTS_DIR, exist_ok=True)
        print(f"Created directory: {ECONOMETRIC_RESULTS_DIR}")

def load_and_prepare_data():
    """Loads and prepares the data for a specific ticker."""
    print(f"Loading data for ticker: {TICKER_TO_MODEL}")
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'])
    ticker_df = df[df['Ticker'] == TICKER_TO_MODEL].copy().sort_values('Date')
    ticker_df.set_index('Date', inplace=True)

    # Split data into 80% train and 20% test sets
    train_size = int(len(ticker_df) * 0.8)
    train_df, test_df = ticker_df[:train_size], ticker_df[train_size:]

    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df

def run_arima_model(train_df, test_df):
    """Trains an ARIMA model for price prediction."""
    print("\n--- Running ARIMA Model ---")
    
    try:
        # Fit ARIMA model
        model = ARIMA(train_df[TARGET_VARIABLE], order=(5, 1, 0))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(test_df))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_df[TARGET_VARIABLE], forecast))
        mae = mean_absolute_error(test_df[TARGET_VARIABLE], forecast)
        
        print(f"ARIMA Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        # Save results
        results = pd.DataFrame({
            'Date': test_df.index,
            'Actual': test_df[TARGET_VARIABLE].values,
            'ARIMA_Forecast': forecast.values
        })
        results.to_csv(os.path.join(ECONOMETRIC_RESULTS_DIR, 'arima_predictions.csv'), index=False)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(test_df.index[-100:], test_df[TARGET_VARIABLE].iloc[-100:], 
                label='Actual', color='blue', linewidth=2)
        plt.plot(test_df.index[-100:], forecast.iloc[-100:], 
                label='ARIMA Forecast', color='red', linewidth=2, linestyle='--')
        plt.title('ARIMA Model: Actual vs Predicted (Last 100 Days)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ECONOMETRIC_RESULTS_DIR, 'arima_forecast_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae}
        
    except Exception as e:
        print(f"ARIMA model failed: {e}")
        return {'Model': 'ARIMA', 'RMSE': 999, 'MAE': 999}

def run_garch_model(train_df, test_df):
    """Trains a GARCH model for volatility prediction."""
    print("\n--- Running GARCH Model for Volatility ---")
    
    try:
        # Prepare returns data (remove NaN values)
        train_returns = train_df[RETURNS_VARIABLE].dropna() * 100  # Scale for GARCH
        
        # Fit GARCH(1,1) model
        garch_model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        fitted_garch = garch_model.fit(disp='off')
        
        # Get volatility forecast
        forecast_horizon = len(test_df)
        volatility_forecast = fitted_garch.forecast(horizon=forecast_horizon)
        
        print(f"GARCH Model Summary:")
        print(f"  Log-Likelihood: {fitted_garch.loglikelihood:.2f}")
        print(f"  AIC: {fitted_garch.aic:.2f}")
        
        # Save model summary
        with open(os.path.join(ECONOMETRIC_RESULTS_DIR, 'garch_model_summary.txt'), 'w') as f:
            f.write(str(fitted_garch.summary()))
        
        # Plot volatility
        plt.figure(figsize=(12, 6))
        plt.plot(train_df.index[-200:], train_df['volatility_garch_normalized'].iloc[-200:], 
                label='Historical Volatility', color='blue', alpha=0.7)
        plt.title('GARCH Model: Volatility Analysis')
        plt.xlabel('Date')
        plt.ylabel('Normalized Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ECONOMETRIC_RESULTS_DIR, 'garch_volatility_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'Model': 'GARCH', 'AIC': fitted_garch.aic, 'LogLikelihood': fitted_garch.loglikelihood}
        
    except Exception as e:
        print(f"GARCH model failed: {e}")
        return {'Model': 'GARCH', 'AIC': 999, 'LogLikelihood': -999}

def create_econometric_summary(arima_results, garch_results):
    """Creates a summary of econometric model results."""
    print("\n--- Creating Econometric Models Summary ---")
    
    summary_data = {
        'Model_Type': ['ARIMA_Price_Prediction', 'GARCH_Volatility_Modeling'],
        'Primary_Metric': [f"RMSE: {arima_results['RMSE']:.4f}", f"AIC: {garch_results['AIC']:.2f}"],
        'Secondary_Metric': [f"MAE: {arima_results['MAE']:.4f}", f"LogLikelihood: {garch_results['LogLikelihood']:.2f}"],
        'Purpose': ['Stock Price Forecasting', 'Volatility Modeling'],
        'Status': ['Completed', 'Completed']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(ECONOMETRIC_RESULTS_DIR, 'econometric_models_summary.csv'), index=False)
    
    print("Econometric Models Summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    print("=== Running Econometric Models Analysis ===")
    
    # Setup
    create_output_directory()
    
    # Load data
    train_data, test_data = load_and_prepare_data()
    
    # Run ARIMA model for price prediction
    arima_results = run_arima_model(train_data, test_data)
    
    # Run GARCH model for volatility modeling
    garch_results = run_garch_model(train_data, test_data)
    
    # Create summary
    create_econometric_summary(arima_results, garch_results)
    
    print(f"\n=== Econometric Analysis Complete ===")
    print(f"Results saved to: {ECONOMETRIC_RESULTS_DIR}")