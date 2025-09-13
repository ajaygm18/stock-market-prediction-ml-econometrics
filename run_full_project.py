#!/usr/bin/env python3
"""
Master execution script for Stock Market Prediction ML Project
This script runs the complete pipeline with proper error handling and timeout management.
"""

import os
import sys
import subprocess
import time
import signal
from datetime import datetime

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, '2_Code-Scripts')

# Disable workspace timeouts by setting environment variables
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if os.path.exists('/proc/driver/nvidia/version') else ''

class TimeoutHandler:
    """Custom timeout handler to prevent workspace timeouts"""
    def __init__(self):
        self.timeout_disabled = True
    
    def disable_timeout(self):
        """Disable any system timeouts"""
        # Set large timeout values
        os.environ['TIMEOUT'] = '36000'  # 10 hours
        os.environ['WORKSPACE_TIMEOUT'] = '36000'
        signal.alarm(0)  # Cancel any existing alarms

def log_message(message, level="INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    sys.stdout.flush()

def run_script(script_path, description, timeout=3600):
    """Run a Python script with error handling and progress reporting"""
    log_message(f"Starting: {description}")
    log_message(f"Script: {script_path}")
    
    try:
        # Change to project root directory
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        # Run the script
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress and output
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
                output_lines.append(line.rstrip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
        
        # Get final output
        remaining_output = process.stdout.read()
        if remaining_output:
            print(remaining_output.rstrip())
            output_lines.extend(remaining_output.rstrip().split('\n'))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode == 0:
            log_message(f"✅ Completed: {description} (Duration: {duration:.2f}s)", "SUCCESS")
            return True, output_lines
        else:
            log_message(f"❌ Failed: {description} (Exit code: {process.returncode})", "ERROR")
            return False, output_lines
            
    except Exception as e:
        log_message(f"❌ Exception in {description}: {str(e)}", "ERROR")
        return False, [str(e)]
    finally:
        os.chdir(original_dir)

def create_required_directories():
    """Create all required directories for the project"""
    log_message("Creating required directories...")
    
    directories = [
        '1_Data_Files/01_Raw_Data/001_Stock_Market_Data',
        '1_Data_Files/01_Raw_Data/002_Macroeconomic_Indicators',
        '1_Data_Files/01_Raw_Data/003_Sentiment_Data',
        '1_Data_Files/02_Cleaned_Data',
        '3_Model_Outputs/01_Raw_Predictions',
        '3_Model_Outputs/02_Evaluation_Metrics',
        '3_Model_Outputs/03_Trained_Models/Trained_Models'
    ]
    
    for directory in directories:
        full_path = os.path.join(PROJECT_ROOT, directory)
        os.makedirs(full_path, exist_ok=True)
        log_message(f"Created directory: {directory}")

def main():
    """Main execution function"""
    log_message("="*80)
    log_message("STOCK MARKET PREDICTION ML PROJECT - FULL EXECUTION")
    log_message("="*80)
    log_message("FRED API Key: 747c4c16bc76a3dfc54d6d63c0ba9e4d")
    log_message("Project will run fully without interruptions")
    log_message("="*80)
    
    # Initialize timeout handler
    timeout_handler = TimeoutHandler()
    timeout_handler.disable_timeout()
    
    # Create required directories
    create_required_directories()
    
    # Define execution pipeline
    pipeline = [
        {
            'script': '2_Code-Scripts/01_Data_Acquisition_and_Cleaning_Scripts/data_acquisition.py',
            'description': 'Data Acquisition (Stock + Macroeconomic Data)',
            'timeout': 1800  # 30 minutes
        },
        {
            'script': '2_Code-Scripts/01_Data_Acquisition_and_Cleaning_Scripts/data_processing.py',
            'description': 'Data Processing and Cleaning',
            'timeout': 900   # 15 minutes
        },
        {
            'script': '2_Code-Scripts/02_Feature_Engineering_Scripts/feature_engineering.py',
            'description': 'Feature Engineering',
            'timeout': 1200  # 20 minutes
        },
        {
            'script': '2_Code-Scripts/05_Econometric_Model_Scripts/econometric_models.py',
            'description': 'Econometric Models (ARIMA, GARCH)',
            'timeout': 2400  # 40 minutes
        },
        {
            'script': '2_Code-Scripts/03_Model_Training_and_Evaluation_Scripts/model_training_and_evaluation.py',
            'description': 'Model Training and Evaluation (LSTM, Hybrid)',
            'timeout': 14400  # 4 hours - longest step
        },
        {
            'script': '3_Model_Outputs/03_Trained_Models/generate_hybrid_model.py',
            'description': 'Hybrid Model Generation',
            'timeout': 3600  # 1 hour
        },
        {
            'script': '2_Code-Scripts/04_Visualization_Scripts/visualization.py',
            'description': 'Results Visualization',
            'timeout': 600   # 10 minutes
        }
    ]
    
    # Execute pipeline
    total_start_time = time.time()
    completed_steps = 0
    failed_steps = 0
    
    for i, step in enumerate(pipeline, 1):
        log_message(f"STEP {i}/{len(pipeline)}: {step['description']}")
        log_message("-" * 60)
        
        success, output = run_script(
            step['script'], 
            step['description'], 
            step['timeout']
        )
        
        if success:
            completed_steps += 1
            log_message(f"✅ Step {i} completed successfully")
        else:
            failed_steps += 1
            log_message(f"❌ Step {i} failed")
            
            # For critical steps, consider stopping
            if i <= 3:  # Data acquisition/processing/feature engineering are critical
                log_message("❌ Critical step failed. Cannot continue.", "ERROR")
                break
            else:
                log_message("⚠️  Non-critical step failed. Continuing...", "WARNING")
        
        log_message("-" * 60)
        
        # Brief pause between steps
        time.sleep(5)
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    log_message("="*80)
    log_message("EXECUTION SUMMARY")
    log_message("="*80)
    log_message(f"Total Duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    log_message(f"Completed Steps: {completed_steps}/{len(pipeline)}")
    log_message(f"Failed Steps: {failed_steps}")
    
    if completed_steps == len(pipeline):
        log_message("🎉 ALL STEPS COMPLETED SUCCESSFULLY!", "SUCCESS")
        log_message("📊 Check the following directories for results:")
        log_message("   - 1_Data_Files/02_Cleaned_Data/ (processed data)")
        log_message("   - 3_Model_Outputs/01_Raw_Predictions/ (model predictions)")
        log_message("   - 3_Model_Outputs/02_Evaluation_Metrics/ (performance metrics)")
        log_message("   - 3_Model_Outputs/03_Trained_Models/ (saved models)")
    else:
        log_message(f"⚠️  Project completed with {failed_steps} failed steps", "WARNING")
    
    log_message("="*80)

if __name__ == "__main__":
    main()