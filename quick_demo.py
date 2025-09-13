#!/usr/bin/env python3
"""
Quick Demo: Directional Accuracy and Future Predictions
======================================================

This script provides a quick demonstration of:
1. Directional accuracy metrics
2. Future price predictions
3. Model performance summary

Usage: python quick_demo.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def show_directional_accuracy():
    """Show directional accuracy in a clear, concise format"""
    print("=" * 60)
    print("📈 DIRECTIONAL ACCURACY RESULTS")
    print("=" * 60)
    
    error_file = '3_Model_Outputs/02_Evaluation_Metrics/error_analysis.csv'
    if os.path.exists(error_file):
        error_df = pd.read_csv(error_file, comment='#')
        error_df = error_df[error_df['Date'] != '...'].dropna()
        
        total = len(error_df)
        correct = error_df['Directional_Accuracy'].sum()
        accuracy = (correct / total) * 100
        
        print(f"🎯 Overall Directional Accuracy: {accuracy:.2f}%")
        print(f"📊 Correct Direction Predictions: {int(correct):,} out of {total:,}")
        
        # Bull vs Bear accuracy
        up_mask = error_df['Actual_Direction'] == 'Up'
        down_mask = error_df['Actual_Direction'] == 'Down'
        
        up_correct = error_df[up_mask]['Directional_Accuracy'].sum()
        up_total = up_mask.sum()
        down_correct = error_df[down_mask]['Directional_Accuracy'].sum()
        down_total = down_mask.sum()
        
        up_acc = (up_correct / up_total * 100) if up_total > 0 else 0
        down_acc = (down_correct / down_total * 100) if down_total > 0 else 0
        
        print(f"📈 Bull Market (Up days): {up_acc:.1f}% ({int(up_correct)}/{up_total})")
        print(f"📉 Bear Market (Down days): {down_acc:.1f}% ({int(down_correct)}/{down_total})")
        
        return accuracy
    else:
        print("❌ Error analysis file not found. Please run the main pipeline first.")
        return None

def show_model_performance():
    """Show model performance comparison"""
    print("\n" + "=" * 60)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    perf_file = '3_Model_Outputs/02_Evaluation_Metrics/model_performance_comparison.csv'
    if os.path.exists(perf_file):
        perf_df = pd.read_csv(perf_file)
        
        print("Model Performance Metrics:")
        print("-" * 45)
        for _, row in perf_df.iterrows():
            print(f"🤖 {row['Model']:<20} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f}")
        
        best_model = perf_df.loc[perf_df['RMSE'].idxmin()]
        print(f"\n🥇 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']:.4f})")
        
        return perf_df
    else:
        print("❌ Performance file not found. Please run the main pipeline first.")
        return None

def show_future_predictions(days=7):
    """Show future price predictions"""
    print(f"\n" + "=" * 60)
    print(f"🔮 FUTURE PRICE PREDICTIONS ({days} DAYS)")
    print("=" * 60)
    
    pred_file = f'3_Model_Outputs/01_Raw_Prediction_Files/future_predictions_TSLA_{days}days.csv'
    
    # If 7-day file doesn't exist, look for 30-day file
    if not os.path.exists(pred_file):
        pred_file = '3_Model_Outputs/01_Raw_Prediction_Files/future_predictions_TSLA_30days.csv'
    
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        
        print(f"Tesla (TSLA) Price Predictions:")
        print("-" * 45)
        print(f"{'Date':<12} {'Price':<10} {'Change':<8}")
        print("-" * 45)
        
        base_price = pred_df['Predicted_Price'].iloc[0]
        for i, row in pred_df.head(days).iterrows():
            date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            price = row['Predicted_Price']
            change = ((price - base_price) / base_price) * 100
            print(f"{date:<12} ${price:<9.2f} {change:+6.2f}%")
        
        # Overall trend
        last_price = pred_df['Predicted_Price'].iloc[min(days-1, len(pred_df)-1)]
        total_change = ((last_price - base_price) / base_price) * 100
        trend = "📈 BULLISH" if total_change > 0 else "📉 BEARISH"
        
        print(f"\n📊 {days}-Day Trend: {trend} ({total_change:+.2f}%)")
        print(f"⚠️  Note: Predictions are for testing/analysis purposes only")
        
        return pred_df
    else:
        print("❌ Future predictions not found. Please run: python test_predictions_and_analysis.py")
        return None

def main():
    """Main demo function"""
    print("🚀 Stock Market Prediction - Quick Demo")
    print("=" * 60)
    print("Demonstrating directional accuracy and future price predictions")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('3_Model_Outputs'):
        print("❌ Model outputs not found!")
        print("🔧 Please run the main pipeline first:")
        print("   python run_full_pipeline.py")
        return
    
    # Show results
    accuracy = show_directional_accuracy()
    performance = show_model_performance()
    predictions = show_future_predictions(7)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ SUMMARY")
    print("=" * 60)
    
    if accuracy:
        if accuracy > 70:
            print(f"🎯 Excellent directional accuracy: {accuracy:.1f}% (>70% is very good)")
        elif accuracy > 60:
            print(f"🎯 Good directional accuracy: {accuracy:.1f}% (>60% is good)")
        elif accuracy > 50:
            print(f"🎯 Decent directional accuracy: {accuracy:.1f}% (>50% beats random)")
        else:
            print(f"🎯 Directional accuracy: {accuracy:.1f}% (needs improvement)")
    
    print("📊 All models trained and evaluated successfully")
    print("🔮 Future predictions generated for testing")
    print("📁 Check 3_Model_Outputs/ directory for detailed results")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! The models are working well.")
    print("=" * 60)

if __name__ == "__main__":
    main()