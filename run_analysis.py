#!/usr/bin/env python3
"""
Simple script to run the complete Google Play Store Apps EDA
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'sklearn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_analysis():
    """Run the complete analysis"""
    print("🚀 Starting Google Play Store Apps EDA")
    print("=" * 60)
    
    # Check if data files exist
    data_files = ['data/googleplaystore.csv', 'data/googleplaystore_user_reviews.csv']
    missing_files = []
    
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n📁 Please ensure the CSV files are in the data/ directory")
        return False
    
    print("✅ Data files found!")
    
    # Run the main analysis
    try:
        print("\n📊 Running main analysis...")
        result = subprocess.run([sys.executable, 'main_analysis.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_exploration():
    """Run the data exploration script"""
    print("\n🔍 Running data exploration...")
    try:
        result = subprocess.run([sys.executable, 'notebooks/01_data_exploration.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running exploration: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function"""
    print("Google Play Store Apps - EDA Runner")
    print("=" * 40)
    if not check_dependencies():
        return
    # Always run Complete Analysis
    print("\nRunning Complete Analysis...")
    run_analysis()

if __name__ == "__main__":
    main() 