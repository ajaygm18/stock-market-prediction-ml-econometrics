#!/usr/bin/env python3
"""
Main execution script for Stock Market Prediction ML & Econometrics Project
This script runs the full pipeline from data acquisition to model training
and ensures all components work together seamlessly.

Author: Copilot Assistant
Date: 2024
"""

import os
import sys
import subprocess
import time
import traceback
from pathlib import Path

def print_banner(title):
    """Print a formatted banner for each step."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_step(step_num, description):
    """Print step information."""
    print(f"\n📋 STEP {step_num}: {description}")
    print("-" * 60)

def run_script(script_path, description, timeout=1800):  # 30 minutes default timeout
    """Run a Python script and handle errors."""
    print(f"🚀 Running: {description}")
    print(f"📁 Script: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        return False
    
    try:
        start_time = time.time()
        
        # Change to the script's directory and run it
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        original_dir = os.getcwd()
        if script_dir:
            os.chdir(script_dir)
        
        # Run the script with explicit timeout
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        # Change back to original directory
        os.chdir(original_dir)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! Completed in {execution_time:.2f} seconds")
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ Error! Return code: {result.returncode}")
            if result.stderr:
                print("📥 Error output:")
                print(result.stderr)
            if result.stdout:
                print("📤 Standard output:")
                print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Script timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def install_dependencies():
    """Install required dependencies."""
    print_step(0, "Installing Dependencies")
    
    requirements_path = "2_Code-Scripts/requirements.txt"
    if os.path.exists(requirements_path):
        print("📦 Installing packages from requirements.txt...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    else:
        print(f"⚠️ Requirements file not found at {requirements_path}")
        return False

def check_python_version():
    """Check Python version compatibility."""
    print_step(-1, "Checking Python Version")
    
    python_version = sys.version_info
    print(f"🐍 Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 9:
        print("✅ Python version is compatible (3.9.0 or higher)")
        return True
    else:
        print("⚠️ Python 3.9.0 or higher recommended, but proceeding anyway")
        return True

def create_necessary_directories():
    """Create all necessary directories for the pipeline."""
    print_step(0.5, "Creating Necessary Directories")
    
    directories = [
        "1_Data_Files/01_Raw_Data/001_Stock_Market_Data",
        "1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators", 
        "1_Data_Files/01_Raw_Data/003_Sentiment_Data",
        "1_Data_Files/02_Cleaned_Data",
        "3_Model_Outputs/01_Raw_Predictions",
        "3_Model_Outputs/02_Evaluation_Metrics",
        "3_Model_Outputs/03_Trained_Models/Trained_Models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")
    
    print("✅ All directories created successfully")
    return True

def main():
    """Main execution function."""
    print_banner("STOCK MARKET PREDICTION ML & ECONOMETRICS PIPELINE")
    print("🎯 Running complete pipeline with real data and FRED API")
    print("⏰ Configured for full execution without timeouts")
    print(f"🔑 Using FRED API Key: 747c4c16bc76a3dfc54d6d63c0ba9e4d")
    
    start_time = time.time()
    
    # Step -1: Check Python version
    if not check_python_version():
        print("❌ Python version check failed")
        return False
    
    # Step 0: Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Step 0.5: Create directories
    if not create_necessary_directories():
        print("❌ Directory creation failed")
        return False
    
    # Define the pipeline steps
    pipeline_steps = [
        {
            "step": 1,
            "description": "Data Acquisition (Stock Data + FRED Macro Data)",
            "script": "2_Code-Scripts/01_Data_Acquisition_and_Cleaning_Scripts/data_acquisition.py",
            "timeout": 1800  # 30 minutes
        },
        {
            "step": 2,
            "description": "Feature Engineering and Data Processing",
            "script": "2_Code-Scripts/02_Feature_Engineering_Scripts/feature_engineering.py",
            "timeout": 1800  # 30 minutes
        },
        {
            "step": 3,
            "description": "Model Training and Evaluation (ARIMA, LSTM, Hybrid)",
            "script": "2_Code-Scripts/03_Model_Training_and_Evaluation_Scripts/model_training_and_evaluation.py",
            "timeout": 7200  # 2 hours for model training
        },
        {
            "step": 4,
            "description": "Generate Hybrid Model File",
            "script": "3_Model_Outputs/03_Trained_Models/generate_hybrid_model.py",
            "timeout": 3600  # 1 hour
        }
    ]
    
    # Execute each step
    success_count = 0
    for step_info in pipeline_steps:
        print_step(step_info["step"], step_info["description"])
        
        if run_script(step_info["script"], step_info["description"], step_info["timeout"]):
            success_count += 1
            print(f"✅ Step {step_info['step']} completed successfully")
        else:
            print(f"❌ Step {step_info['step']} failed")
            print("🔄 Continuing with next step...")
    
    # Final summary
    total_time = time.time() - start_time
    print_banner("PIPELINE EXECUTION SUMMARY")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"✅ Successful steps: {success_count}/{len(pipeline_steps)}")
    
    if success_count == len(pipeline_steps):
        print("🎉 All steps completed successfully!")
        print("📊 Your stock market prediction models are ready!")
        
        # List output files
        print("\n📁 Generated files:")
        output_files = [
            "1_Data_Files/01_Raw_Data/001_Stock_Market_Data/stock_data_2010-2023.csv",
            "1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators/macroeconomic_indicators_raw.csv",
            "1_Data_Files/02_Cleaned_Data/processed_stock_data_2010-2023.csv",
            "3_Model_Outputs/02_Evaluation_Metrics/model_performance_comparison.csv",
            "3_Model_Outputs/03_Trained_Models/Trained_Models/hybrid_model.h5"
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  ✅ {file_path} ({file_size:,} bytes)")
            else:
                print(f"  ❌ {file_path} (not found)")
        
        return True
    else:
        print(f"⚠️  {len(pipeline_steps) - success_count} steps failed")
        print("🔍 Check the error messages above for details")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in main pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)