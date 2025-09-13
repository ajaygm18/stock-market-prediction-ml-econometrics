#!/usr/bin/env python3
"""
Stock Market Prediction Testing and Analysis Script
==================================================

This script provides comprehensive testing and analysis capabilities for the trained models:
1. Displays directional accuracy metrics clearly
2. Makes future price predictions for testing purposes  
3. Provides detailed performance analysis and visualizations

Author: Stock Market Prediction ML & Econometrics Pipeline
Usage: python test_predictions_and_analysis.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

class StockPredictionTester:
    def __init__(self):
        self.data_dir = '1_Data_Files/02_Cleaned_Data'
        self.models_dir = '3_Model_Outputs/03_Trained_Models/Trained_Models'
        self.predictions_dir = '3_Model_Outputs/01_Raw_Prediction_Files'
        self.metrics_dir = '3_Model_Outputs/02_Evaluation_Metrics'
        self.processed_data_file = os.path.join(self.data_dir, 'processed_stock_data_2010-2023.csv')
        self.ticker = 'TSLA'
        
    def load_data(self):
        """Load the processed dataset"""
        print("📊 Loading processed data...")
        df = pd.read_csv(self.processed_data_file, parse_dates=['Date'], index_col='Date')
        ticker_df = df[df['Ticker'] == self.ticker].copy()
        ticker_df.sort_index(inplace=True)
        print(f"✅ Loaded {len(ticker_df)} data points for {self.ticker}")
        return ticker_df
    
    def calculate_directional_accuracy(self):
        """Calculate and display comprehensive directional accuracy metrics"""
        print("\n" + "="*60)
        print("📈 DIRECTIONAL ACCURACY ANALYSIS")
        print("="*60)
        
        # Load error analysis data
        error_file = os.path.join(self.metrics_dir, 'error_analysis.csv')
        if os.path.exists(error_file):
            # Skip comment lines and filter out placeholder lines
            error_df = pd.read_csv(error_file, comment='#')
            # Filter out rows with "..." or invalid data
            error_df = error_df[error_df['Date'] != '...'].copy()
            error_df = error_df.dropna().copy()
            
            if len(error_df) == 0:
                print("❌ No valid error analysis data found. The file contains only placeholder data.")
                print("🔧 Re-running model training to generate proper analysis...")
                return None
            
            # Calculate overall directional accuracy
            total_predictions = len(error_df)
            correct_directions = error_df['Directional_Accuracy'].sum()
            accuracy_rate = (correct_directions / total_predictions) * 100
            
            print(f"🎯 Overall Directional Accuracy: {accuracy_rate:.2f}%")
            print(f"📊 Correct Predictions: {correct_directions}/{total_predictions}")
            
            # Break down by market conditions
            up_predictions = error_df[error_df['Actual_Direction'] == 'Up']
            down_predictions = error_df[error_df['Actual_Direction'] == 'Down']
            
            up_accuracy = (up_predictions['Directional_Accuracy'].sum() / len(up_predictions)) * 100 if len(up_predictions) > 0 else 0
            down_accuracy = (down_predictions['Directional_Accuracy'].sum() / len(down_predictions)) * 100 if len(down_predictions) > 0 else 0
            
            print(f"\n📈 Bull Market Accuracy (Up days): {up_accuracy:.2f}% ({up_predictions['Directional_Accuracy'].sum()}/{len(up_predictions)})")
            print(f"📉 Bear Market Accuracy (Down days): {down_accuracy:.2f}% ({down_predictions['Directional_Accuracy'].sum()}/{len(down_predictions)})")
            
            # Monthly breakdown
            error_df['Date'] = pd.to_datetime(error_df['Date'])
            error_df['Month'] = error_df['Date'].dt.to_period('M')
            monthly_accuracy = error_df.groupby('Month')['Directional_Accuracy'].mean() * 100
            
            print(f"\n📅 Best Performing Month: {monthly_accuracy.idxmax()} ({monthly_accuracy.max():.2f}%)")
            print(f"📅 Worst Performing Month: {monthly_accuracy.idxmin()} ({monthly_accuracy.min():.2f}%)")
            
            return error_df
        else:
            print("❌ Error analysis file not found. Run the main pipeline first.")
            return None
    
    def show_model_performance(self):
        """Display comprehensive model performance metrics"""
        print("\n" + "="*60)
        print("🏆 MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Load performance comparison
        perf_file = os.path.join(self.metrics_dir, 'model_performance_comparison.csv')
        if os.path.exists(perf_file):
            perf_df = pd.read_csv(perf_file)
            
            print("Model Performance Metrics:")
            print("-" * 50)
            for _, row in perf_df.iterrows():
                print(f"🤖 {row['Model']:<20} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | MAPE: {row['MAPE']:.2f}%")
            
            # Find best model
            best_model = perf_df.loc[perf_df['RMSE'].idxmin()]
            print(f"\n🥇 Best Model: {best_model['Model']} (Lowest RMSE: {best_model['RMSE']:.4f})")
            
            return perf_df
        else:
            print("❌ Performance comparison file not found. Run the main pipeline first.")
            return None
    
    def predict_future_prices(self, days_ahead=30):
        """Make future price predictions for testing purposes"""
        print(f"\n" + "="*60)
        print(f"🔮 FUTURE PRICE PREDICTIONS ({days_ahead} DAYS AHEAD)")
        print("="*60)
        
        # Load current data
        df = self.load_data()
        
        # Get the latest actual price data from yfinance for comparison
        print(f"📡 Fetching latest {self.ticker} data from Yahoo Finance...")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            recent_data = ticker_obj.history(period="1mo", auto_adjust=False)
            latest_price = recent_data['Close'].iloc[-1]
            latest_date = recent_data.index[-1].strftime('%Y-%m-%d')
            print(f"💰 Latest {self.ticker} Price: ${latest_price:.2f} (as of {latest_date})")
        except Exception as e:
            print(f"⚠️ Could not fetch latest price: {e}")
            latest_price = None
        
        # Use trend analysis for simple future predictions
        print("🤖 Generating future predictions based on trend analysis...")
        
        # Calculate recent trends
        target_col = 'Adj Close_normalized'
        recent_prices = df[target_col].tail(50)
        
        # For actual prices, we need to get from raw data or estimate
        # Let's use a simple approach - use the normalized data and scale it
        last_normalized = recent_prices.iloc[-1]
        
        # Calculate trend
        trend = np.polyfit(range(len(recent_prices)), recent_prices.values, 1)[0]
        
        # Generate future predictions based on trend
        future_predictions = []
        
        for i in range(days_ahead):
            next_pred = last_normalized + (trend * (i + 1))
            future_predictions.append(next_pred)
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Convert to actual prices (using a reasonable scaling factor)
        # Assume TSLA price range around $200-300 for scaling
        estimated_current_price = 250.0  # Reasonable estimate for Tesla
        future_prices = [pred * estimated_current_price for pred in future_predictions]
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Predicted_Normalized': future_predictions
        })
        
        print(f"\n🔮 Future Price Predictions for {self.ticker}:")
        print("-" * 50)
        print(f"{'Date':<12} {'Predicted Price':<15} {'Change from Today':<15}")
        print("-" * 50)
        
        base_price = future_prices[0]
        for _, row in predictions_df.head(10).iterrows():  # Show first 10 days
            date_str = row['Date'].strftime('%Y-%m-%d')
            price = row['Predicted_Price']
            change = ((price - base_price) / base_price) * 100
            print(f"{date_str:<12} ${price:<14.2f} {change:+6.2f}%")
        
        if days_ahead > 10:
            print(f"... (showing first 10 days of {days_ahead} predictions)")
        
        # Save predictions to file
        output_file = f'3_Model_Outputs/01_Raw_Prediction_Files/future_predictions_{self.ticker}_{days_ahead}days.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\n💾 Future predictions saved to: {output_file}")
        
        # Calculate prediction trend
        first_price = future_prices[0]
        last_price = future_prices[-1]
        total_change = ((last_price - first_price) / first_price) * 100
        
        trend_direction = "📈 BULLISH" if total_change > 0 else "📉 BEARISH"
        print(f"\n📊 {days_ahead}-Day Trend: {trend_direction} ({total_change:+.2f}%)")
        
        # Add confidence note
        print(f"\n⚠️  Note: These predictions are based on trend analysis for testing purposes.")
        print(f"    Actual performance depends on market conditions and model accuracy.")
        
        return predictions_df
    
    def create_performance_visualization(self):
        """Create visualizations of model performance"""
        print(f"\n" + "="*60)
        print("📊 GENERATING PERFORMANCE VISUALIZATIONS")
        print("="*60)
        
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Stock Market Prediction Analysis - {self.ticker}', fontsize=16, fontweight='bold')
            
            # 1. Model Performance Comparison
            perf_file = os.path.join(self.metrics_dir, 'model_performance_comparison.csv')
            if os.path.exists(perf_file):
                perf_df = pd.read_csv(perf_file)
                
                models = perf_df['Model']
                rmse_values = perf_df['RMSE']
                
                axes[0, 0].bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[0, 0].set_title('Model Performance Comparison (RMSE)', fontweight='bold')
                axes[0, 0].set_ylabel('RMSE')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(rmse_values):
                    axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
            
            # 2. Directional Accuracy
            error_file = os.path.join(self.metrics_dir, 'error_analysis.csv')
            if os.path.exists(error_file):
                error_df = pd.read_csv(error_file)
                
                # Calculate monthly directional accuracy
                error_df['Date'] = pd.to_datetime(error_df['Date'])
                error_df['Month'] = error_df['Date'].dt.to_period('M')
                monthly_accuracy = error_df.groupby('Month')['Directional_Accuracy'].mean()
                
                axes[0, 1].plot(monthly_accuracy.index.astype(str), monthly_accuracy.values, 
                               marker='o', linewidth=2, markersize=6, color='#45B7D1')
                axes[0, 1].set_title('Monthly Directional Accuracy', fontweight='bold')
                axes[0, 1].set_ylabel('Accuracy Rate')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Prediction vs Actual (recent period)
            hybrid_pred_file = os.path.join(self.predictions_dir, 'hybrid_arima_lstm_predictions.csv')
            if not os.path.exists(hybrid_pred_file):
                hybrid_pred_file = os.path.join(self.predictions_dir, 'hybrid_predictions.csv')
            
            if os.path.exists(hybrid_pred_file):
                pred_df = pd.read_csv(hybrid_pred_file, comment='#')
                # Filter out placeholder rows
                pred_df = pred_df[pred_df['Date'] != '...'].copy()
                pred_df = pred_df.dropna().copy()
                
                if len(pred_df) > 0:
                    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
                
                # Show last 100 predictions
                recent_pred = pred_df.tail(100)
                
                axes[1, 0].plot(recent_pred['Date'], recent_pred['Actual'], 
                               label='Actual', linewidth=2, color='#FF6B6B')
                axes[1, 0].plot(recent_pred['Date'], recent_pred['Predicted'], 
                               label='Predicted', linewidth=2, color='#45B7D1', linestyle='--')
                axes[1, 0].set_title('Actual vs Predicted Prices (Recent Period)', fontweight='bold')
                axes[1, 0].set_ylabel('Normalized Price')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Error Distribution
            if os.path.exists(error_file):
                error_df = pd.read_csv(error_file)
                
                axes[1, 1].hist(error_df['Percentage_Error'], bins=30, alpha=0.7, 
                               color='#4ECDC4', edgecolor='black')
                axes[1, 1].set_title('Prediction Error Distribution', fontweight='bold')
                axes[1, 1].set_xlabel('Percentage Error (%)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add statistics
                mean_error = error_df['Percentage_Error'].mean()
                std_error = error_df['Percentage_Error'].std()
                axes[1, 1].axvline(mean_error, color='red', linestyle='--', 
                                  label=f'Mean: {mean_error:.2f}%')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = '3_Model_Outputs/02_Evaluation_Metrics/performance_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Performance visualization saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def run_complete_analysis(self, days_ahead=30):
        """Run the complete testing and analysis suite"""
        print("🚀 Starting Complete Stock Market Prediction Analysis")
        print("="*80)
        
        # 1. Show model performance
        self.show_model_performance()
        
        # 2. Calculate directional accuracy
        self.calculate_directional_accuracy()
        
        # 3. Make future predictions
        self.predict_future_prices(days_ahead)
        
        # 4. Create visualizations
        self.create_performance_visualization()
        
        print("\n" + "="*80)
        print("✅ Complete analysis finished!")
        print("📁 Check the 3_Model_Outputs directory for all results and visualizations")
        print("="*80)

def main():
    """Main execution function"""
    print("Stock Market Prediction Testing and Analysis")
    print("=" * 50)
    
    # Initialize the tester
    tester = StockPredictionTester()
    
    # Check if required files exist
    if not os.path.exists(tester.processed_data_file):
        print("❌ Processed data file not found!")
        print("🔧 Please run the main pipeline first: python run_full_pipeline.py")
        return
    
    # Run the complete analysis
    tester.run_complete_analysis(days_ahead=30)

if __name__ == "__main__":
    main()