#!/usr/bin/env python3
"""
Main Pipeline for Stock Market Prediction ML Econometrics Project
This script runs the complete pipeline from data acquisition to model evaluation.

Required:
- Python 3.9.0
- FRED API key: 747c4c16bc76a3dfc54d6d63c0ba9e4d
- Auto_adjust=False for yfinance to fix adj close missing problem
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Use the virtual environment Python
        python_executable = PROJECT_ROOT / "venv39" / "bin" / "python"
        result = subprocess.run([str(python_executable), script_path], 
                              capture_output=True, text=True, timeout=3600)
        
        elapsed_time = time.time() - start_time
        print(f"Output:\n{result.stdout}")
        
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"❌ FAILED: {description} (Exit code: {result.returncode})")
            return False
        else:
            print(f"✅ SUCCESS: {description} (Completed in {elapsed_time:.1f}s)")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} exceeded 1 hour limit")
        return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {str(e)}")
        return False

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment setup...")
    
    # Check if virtual environment exists
    venv_path = PROJECT_ROOT / "venv39"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    # Check Python version in venv
    python_executable = venv_path / "bin" / "python"
    try:
        result = subprocess.run([str(python_executable), "--version"], 
                              capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"✅ Python version: {version}")
        
        if "3.9" not in version:
            print("⚠️  Warning: Not using Python 3.9 as requested")
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        return False
    
    return True

def main():
    """Run the complete pipeline."""
    print("🚀 Starting Stock Market Prediction ML Pipeline")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"FRED API Key: 747c4c16bc76a3dfc54d6d63c0ba9e4d")
    print("Configuration: auto_adjust=False for yfinance")
    
    if not check_environment():
        print("❌ Environment check failed. Exiting.")
        sys.exit(1)
    
    # Define the pipeline steps
    pipeline_steps = [
        {
            "script": "2_Code-Scripts/01_Data_Acquisition_and_Cleaning_Scripts/data_acquisition.py",
            "description": "Data Acquisition (Stock data + FRED macroeconomic data)"
        },
        {
            "script": "2_Code-Scripts/01_Data_Acquisition_and_Cleaning_Scripts/data_processing.py", 
            "description": "Data Processing and Cleaning"
        },
        {
            "script": "2_Code-Scripts/02_Feature_Engineering_Scripts/feature_engineering.py",
            "description": "Feature Engineering"
        },
        {
            "script": "2_Code-Scripts/03_Model_Training_and_Evaluation_Scripts/model_training_and_evaluation.py",
            "description": "Model Training and Evaluation (ARIMA, LSTM)"
        },
        {
            "script": "2_Code-Scripts/05_Econometric_Model_Scripts/econometric_models.py",
            "description": "Econometric Models (ARIMA, GARCH)"
        },
        {
            "script": "3_Model_Outputs/03_Trained_Models/generate_hybrid_model.py",
            "description": "Hybrid ARIMA-LSTM Model Generation"
        },
        {
            "script": "2_Code-Scripts/04_Visualization_Scripts/visualization.py",
            "description": "Results Visualization"
        }
    ]
    
    # Track successful steps
    successful_steps = 0
    total_steps = len(pipeline_steps)
    
    # Run each step
    for i, step in enumerate(pipeline_steps, 1):
        script_path = step["script"]
        description = step["description"]
        
        print(f"\n🔄 Step {i}/{total_steps}: {description}")
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            continue
        
        # Run the script
        if run_script(script_path, description):
            successful_steps += 1
        else:
            print(f"⚠️  Step {i} failed, but continuing with remaining steps...")
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print('='*60)
    print(f"✅ Successful steps: {successful_steps}/{total_steps}")
    print(f"❌ Failed steps: {total_steps - successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nModel files should be available in:")
        print("- 3_Model_Outputs/03_Trained_Models/")
        print("- 3_Model_Outputs/01_Raw_Predictions/")
        print("- 3_Model_Outputs/02_Evaluation_Metrics/")
    else:
        print("⚠️  Some steps failed. Check the output above for details.")
    
    print(f"\nPipeline completed. Check the output directories for results.")

if __name__ == "__main__":
    main()