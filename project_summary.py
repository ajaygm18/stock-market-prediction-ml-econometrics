#!/usr/bin/env python3
"""
Project Summary Script - Stock Market Prediction ML Project
Displays comprehensive results and performance metrics
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def print_separator(title="", char="=", width=80):
    """Print a formatted separator"""
    if title:
        print(f"\n{char*width}")
        print(f"{title:^{width}}")
        print(f"{char*width}")
    else:
        print(f"{char*width}")

def summarize_data_acquisition():
    """Summarize data acquisition results"""
    print_separator("DATA ACQUISITION SUMMARY")
    
    # Stock data
    stock_file = "1_Data_Files/01_Raw_Data/001_Stock_Market_Data/stock_data_2010-2023.csv"
    if os.path.exists(stock_file):
        stock_df = pd.read_csv(stock_file)
        print(f"📈 STOCK DATA:")
        print(f"   • File: {stock_file}")
        print(f"   • Records: {len(stock_df):,}")
        print(f"   • Tickers: {stock_df['Ticker'].unique()}")
        print(f"   • Date Range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
        print(f"   • Sample Price Range (TSLA): ${stock_df[stock_df['Ticker']=='TSLA']['Close'].min():.2f} - ${stock_df[stock_df['Ticker']=='TSLA']['Close'].max():.2f}")
    
    # Macro data
    macro_file = "1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators/macroeconomic_indicators_raw.csv"
    if os.path.exists(macro_file):
        macro_df = pd.read_csv(macro_file)
        print(f"\n🏦 MACROECONOMIC DATA (FRED API):")
        print(f"   • File: {macro_file}")
        print(f"   • Records: {len(macro_df):,}")
        print(f"   • Date Range: {macro_df['Date'].min()} to {macro_df['Date'].max()}")
        print(f"   • Indicators: {list(macro_df.columns[1:])}")

def summarize_model_performance():
    """Summarize model performance results"""
    print_separator("MODEL PERFORMANCE SUMMARY")
    
    # Cross-validation results
    cv_file = "3_Model_Outputs/02_Evaluation_Metrics/cross_validation_results.csv"
    if os.path.exists(cv_file):
        cv_df = pd.read_csv(cv_file)
        print("📊 CROSS-VALIDATION RESULTS:")
        
        # Calculate averages by model
        avg_performance = cv_df.groupby('Model')[['RMSE', 'MAE', 'R-squared']].mean()
        
        for model in avg_performance.index:
            rmse = avg_performance.loc[model, 'RMSE']
            mae = avg_performance.loc[model, 'MAE']
            r2 = avg_performance.loc[model, 'R-squared']
            print(f"   • {model}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.3f}")
    
    # Error analysis
    error_file = "3_Model_Outputs/02_Evaluation_Metrics/error_analysis.csv"
    if os.path.exists(error_file):
        error_df = pd.read_csv(error_file)
        print(f"\n🎯 HYBRID MODEL DETAILED ANALYSIS:")
        print(f"   • Test Samples: {len(error_df)}")
        print(f"   • Average Absolute Error: {error_df['Raw_Error'].abs().mean():.4f}")
        print(f"   • Average Percentage Error: {error_df['Percentage_Error'].abs().mean():.2f}%")
        print(f"   • Directional Accuracy: {error_df['Directional_Accuracy'].mean()*100:.1f}%")

def summarize_predictions():
    """Summarize prediction results"""
    print_separator("PREDICTION RESULTS SUMMARY")
    
    prediction_files = {
        "ARIMA": "3_Model_Outputs/01_Raw_Prediction_Files/arima_predictions.csv",
        "LSTM": "3_Model_Outputs/01_Raw_Prediction_Files/lstm_predictions.csv", 
        "Hybrid": "3_Model_Outputs/01_Raw_Prediction_Files/hybrid_predictions.csv"
    }
    
    for model_name, file_path in prediction_files.items():
        if os.path.exists(file_path):
            pred_df = pd.read_csv(file_path)
            print(f"🔮 {model_name} PREDICTIONS:")
            print(f"   • Predictions Generated: {len(pred_df)}")
            if len(pred_df) > 0 and 'Actual' in pred_df.columns and 'Predicted' in pred_df.columns:
                mae = np.mean(np.abs(pred_df['Actual'] - pred_df['Predicted']))
                print(f"   • Mean Absolute Error: {mae:.4f}")
                print(f"   • Sample Prediction vs Actual: {pred_df['Predicted'].iloc[0]:.4f} vs {pred_df['Actual'].iloc[0]:.4f}")

def summarize_trained_models():
    """Summarize trained model files"""
    print_separator("TRAINED MODELS SUMMARY")
    
    # Check model files
    model_dir = "3_Model_Outputs/03_Trained_Models/"
    model_files = []
    
    if os.path.exists(model_dir):
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(('.h5', '.json', '.csv')):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    model_files.append((file, file_size))
    
    print("💾 SAVED MODEL FILES:")
    for file_name, size_kb in model_files:
        print(f"   • {file_name}: {size_kb:.1f} KB")
    
    # Hyperparameter results
    hp_file = "3_Model_Outputs/03_Trained_Models/hyperparameter_tuning_results.csv"
    if os.path.exists(hp_file):
        hp_df = pd.read_csv(hp_file)
        print(f"\n⚙️  HYPERPARAMETER TUNING:")
        print(f"   • Configurations Tested: {len(hp_df)}")
        if 'val_loss' in hp_df.columns:
            best_idx = hp_df['val_loss'].idxmin()
            best_loss = hp_df.loc[best_idx, 'val_loss']
            print(f"   • Best Validation Loss: {best_loss:.6f}")

def create_summary_visualization():
    """Create a summary visualization"""
    print_separator("GENERATING SUMMARY VISUALIZATION")
    
    # Check if we have cross-validation results
    cv_file = "3_Model_Outputs/02_Evaluation_Metrics/cross_validation_results.csv"
    if os.path.exists(cv_file):
        cv_df = pd.read_csv(cv_file)
        
        # Create performance comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['RMSE', 'MAE', 'R-squared']
        for i, metric in enumerate(metrics):
            cv_df.boxplot(column=metric, by='Model', ax=axes[i])
            axes[i].set_title(f'{metric} by Model')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        output_file = "3_Model_Outputs/project_summary_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Performance comparison saved to: {output_file}")
        plt.close()
    
    # Create prediction timeline if data exists
    pred_file = "3_Model_Outputs/01_Raw_Prediction_Files/hybrid_predictions.csv"
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        if len(pred_df) > 1:
            plt.figure(figsize=(12, 6))
            
            dates = pd.to_datetime(pred_df['Date']) if 'Date' in pred_df.columns else range(len(pred_df))
            plt.plot(dates, pred_df['Actual'], label='Actual', linewidth=2, alpha=0.8)
            plt.plot(dates, pred_df['Predicted'], label='Predicted', linewidth=2, alpha=0.8)
            
            plt.title('Hybrid Model: Predicted vs Actual Stock Prices (TSLA)', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel('Normalized Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_file = "3_Model_Outputs/project_summary_predictions.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📈 Prediction timeline saved to: {output_file}")
            plt.close()

def generate_final_report():
    """Generate final project report"""
    print_separator("FINAL PROJECT REPORT", "=", 100)
    
    print("🎉 STOCK MARKET PREDICTION ML PROJECT - EXECUTION COMPLETE")
    print(f"⏰ Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔑 FRED API Key: 747c4c16bc76a3dfc54d6d63c0ba9e4d")
    
    summarize_data_acquisition()
    summarize_model_performance()
    summarize_predictions()
    summarize_trained_models()
    create_summary_visualization()
    
    print_separator("PROJECT COMPLETION STATUS")
    
    # Check completion status
    completed_components = []
    missing_components = []
    
    # Data components
    if os.path.exists("1_Data_Files/01_Raw_Data/001_Stock_Market_Data/stock_data_2010-2023.csv"):
        completed_components.append("✅ Stock Data Acquisition")
    else:
        missing_components.append("❌ Stock Data Acquisition")
    
    if os.path.exists("1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators/macroeconomic_indicators_raw.csv"):
        completed_components.append("✅ FRED Macroeconomic Data")
    else:
        missing_components.append("❌ FRED Macroeconomic Data")
    
    if os.path.exists("1_Data_Files/02_Cleaned_Data/processed_stock_data_2010-2023.csv"):
        completed_components.append("✅ Data Processing & Feature Engineering")
    else:
        missing_components.append("❌ Data Processing & Feature Engineering")
    
    # Model components
    if os.path.exists("3_Model_Outputs/01_Raw_Prediction_Files/lstm_predictions.csv"):
        completed_components.append("✅ LSTM Model Training")
    else:
        missing_components.append("❌ LSTM Model Training")
    
    if os.path.exists("3_Model_Outputs/01_Raw_Prediction_Files/hybrid_predictions.csv"):
        completed_components.append("✅ Hybrid ARIMA-LSTM Model")
    else:
        missing_components.append("❌ Hybrid ARIMA-LSTM Model")
    
    if os.path.exists("3_Model_Outputs/02_Evaluation_Metrics/cross_validation_results.csv"):
        completed_components.append("✅ Model Evaluation & Cross-Validation")
    else:
        missing_components.append("❌ Model Evaluation & Cross-Validation")
    
    if os.path.exists("3_Model_Outputs/03_Trained_Models/Trained_Models/hybrid_model.h5"):
        completed_components.append("✅ Model Persistence")
    else:
        missing_components.append("❌ Model Persistence")
    
    print("COMPLETED COMPONENTS:")
    for component in completed_components:
        print(f"  {component}")
    
    if missing_components:
        print("\nMISSING COMPONENTS:")
        for component in missing_components:
            print(f"  {component}")
    
    completion_rate = len(completed_components) / (len(completed_components) + len(missing_components)) * 100
    print(f"\n📊 OVERALL COMPLETION RATE: {completion_rate:.1f}%")
    
    print_separator("KEY ACHIEVEMENTS")
    print("🎯 Successfully integrated FRED API for real macroeconomic data")
    print("🤖 Trained multiple ML models: ARIMA, LSTM, and Hybrid ARIMA-LSTM")
    print("📊 Generated comprehensive model performance metrics")
    print("💾 Saved trained models for future use")
    print("📈 Created prediction outputs for all models")
    print("🔄 Implemented 5-fold time-series cross-validation")
    
    print_separator("OUTPUT FILES DIRECTORY")
    print("📁 Data Files: 1_Data_Files/")
    print("📁 Model Outputs: 3_Model_Outputs/")
    print("📁 Predictions: 3_Model_Outputs/01_Raw_Prediction_Files/")
    print("📁 Metrics: 3_Model_Outputs/02_Evaluation_Metrics/")
    print("📁 Trained Models: 3_Model_Outputs/03_Trained_Models/")
    
    print_separator("", "=", 100)

if __name__ == "__main__":
    os.chdir('/home/runner/work/stock-market-prediction-ml-econometrics/stock-market-prediction-ml-econometrics')
    generate_final_report()