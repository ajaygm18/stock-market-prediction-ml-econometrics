import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
PROCESSED_DATA_DIR = '../../1_Data_Files/02_Processed_Data'
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'feature_engineered_data.csv')
RESULTS_DIR = '../../3_Model_Outputs'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'model_performance_results.csv')

# Visualization Output Files
PERFORMANCE_PLOT_FILE = os.path.join(RESULTS_DIR, 'model_performance_comparison.png')
STOCK_PRICE_PLOT_FILE = os.path.join(RESULTS_DIR, 'stock_price_trends.png')
CORRELATION_PLOT_FILE = os.path.join(RESULTS_DIR, 'feature_correlation_matrix.png')

TICKER_TO_MODEL = 'TSLA'

def create_output_directory():
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Created directory: {RESULTS_DIR}")

def plot_model_performance():
    """Creates a bar chart comparing model performance."""
    print("Creating model performance comparison plot...")
    
    # Load results
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return
        
    results_df = pd.read_csv(RESULTS_FILE)
    
    # Create performance comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['RMSE', 'MAE', 'MAPE']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, metric in enumerate(metrics):
        axes[i].bar(results_df['Model'], results_df[metric], color=colors[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        for j, v in enumerate(results_df[metric]):
            axes[i].text(j, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PERFORMANCE_PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to: {PERFORMANCE_PLOT_FILE}")

def plot_stock_price_trends():
    """Creates a plot showing stock price trends for both tickers."""
    print("Creating stock price trends plot...")
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot for each ticker
    tickers = df['Ticker'].unique()
    colors = ['blue', 'red']
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        axes[i].plot(ticker_data['Date'], ticker_data['Close_normalized'], 
                    color=colors[i], linewidth=1, alpha=0.8)
        axes[i].set_title(f'{ticker} - Normalized Stock Price Trend')
        axes[i].set_ylabel('Normalized Price')
        axes[i].grid(True, alpha=0.3)
        
        # Add some statistics
        mean_price = ticker_data['Close_normalized'].mean()
        axes[i].axhline(y=mean_price, color=colors[i], linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_price:.3f}')
        axes[i].legend()
    
    axes[1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(STOCK_PRICE_PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stock price trends plot saved to: {STOCK_PRICE_PLOT_FILE}")

def plot_correlation_matrix():
    """Creates a correlation matrix heatmap for key features."""
    print("Creating feature correlation matrix plot...")
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'])
    tesla_data = df[df['Ticker'] == 'TSLA'].copy()
    
    # Select numeric columns for correlation
    numeric_cols = tesla_data.select_dtypes(include=[np.number]).columns
    correlation_features = [col for col in numeric_cols if 'normalized' in col or col in 
                          ['GDP_Growth_Rate_YoY', 'Inflation_Rate_CPI_YoY']]
    
    if len(correlation_features) > 0:
        corr_matrix = tesla_data[correlation_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix (Tesla Data)')
        plt.tight_layout()
        plt.savefig(CORRELATION_PLOT_FILE, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix plot saved to: {CORRELATION_PLOT_FILE}")
    else:
        print("No suitable features found for correlation matrix")

def plot_macroeconomic_indicators():
    """Creates plots for macroeconomic indicators over time."""
    print("Creating macroeconomic indicators plot...")
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=['Date'])
    # Get unique dates and macro data
    macro_data = df[['Date', 'GDP_Growth_Rate_YoY_normalized', 'Interest_Rate_Federal_Funds_normalized', 
                    'Inflation_Rate_CPI_YoY_normalized']].drop_duplicates().sort_values('Date')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # GDP Growth Rate
    axes[0].plot(macro_data['Date'], macro_data['GDP_Growth_Rate_YoY_normalized'], 
                color='green', linewidth=2)
    axes[0].set_title('GDP Growth Rate (Normalized)')
    axes[0].set_ylabel('Normalized Growth Rate')
    axes[0].grid(True, alpha=0.3)
    
    # Interest Rate (normalized)
    axes[1].plot(macro_data['Date'], macro_data['Interest_Rate_Federal_Funds_normalized'], 
                color='blue', linewidth=2)
    axes[1].set_title('Federal Funds Rate (Normalized)')
    axes[1].set_ylabel('Normalized Rate')
    axes[1].grid(True, alpha=0.3)
    
    # Inflation Rate
    axes[2].plot(macro_data['Date'], macro_data['Inflation_Rate_CPI_YoY_normalized'], 
                color='red', linewidth=2)
    axes[2].set_title('Inflation Rate (Normalized)')
    axes[2].set_ylabel('Normalized Inflation Rate')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    macro_plot_file = os.path.join(RESULTS_DIR, 'macroeconomic_indicators.png')
    plt.savefig(macro_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Macroeconomic indicators plot saved to: {macro_plot_file}")

if __name__ == "__main__":
    print("=== Generating Visualizations ===")
    
    # Setup
    create_output_directory()
    
    # Generate plots
    plot_model_performance()
    plot_stock_price_trends()
    plot_correlation_matrix()
    plot_macroeconomic_indicators()
    
    print("\n=== Visualization Generation Complete ===")
    print(f"All plots saved to: {RESULTS_DIR}")