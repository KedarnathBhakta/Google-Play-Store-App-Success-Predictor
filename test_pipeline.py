#!/usr/bin/env python3
"""
Test Script for Google Play Store Predictive Modeling Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        from predictor import AppSuccessPredictor
        
        predictor = AppSuccessPredictor()
        data = predictor.load_data()
        
        assert len(data) > 0, "Data should not be empty"
        assert 'App' in data.columns, "App column should exist"
        assert 'Rating' in data.columns, "Rating column should exist"
        
        print("✅ Data loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("🧪 Testing feature engineering...")
    
    try:
        from predictor import AppSuccessPredictor
        
        predictor = AppSuccessPredictor()
        predictor.load_data()
        predictor.create_target_variables()
        predictor.engineer_features()
        
        # Check if new features were created
        expected_features = [
            'App_Name_Length', 'App_Name_Word_Count', 'Has_Special_Chars',
            'Category_Count', 'Is_Free', 'Days_Since_Update',
            'Content_Rating_Strictness', 'Android_Version_Numeric',
            'Review_Ratio', 'Rating_Review_Ratio'
        ]
        
        for feature in expected_features:
            assert feature in predictor.data.columns, f"Feature {feature} should exist"
        
        print("✅ Feature engineering test passed")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {str(e)}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("🧪 Testing model training...")
    
    try:
        from predictor import AppSuccessPredictor
        
        predictor = AppSuccessPredictor()
        predictor.load_data()
        predictor.create_target_variables()
        predictor.engineer_features()
        
        # Test training for one target
        results = predictor.train_models('Rating')
        
        assert len(results) > 0, "Should have trained models"
        assert 'XGBoost' in results, "XGBoost should be trained"
        assert 'Random Forest' in results, "Random Forest should be trained"
        
        print("✅ Model training test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {str(e)}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    print("🧪 Testing file structure...")
    
    required_files = [
        'data/apps_cleaned.csv',
        'src/predictor.py',
        'src/explainer.py',
        'run_predictive_modeling.py',
        'predict_new_app.py',
        'deploy_api.py',
        'requirements.txt'
    ]
    
    required_dirs = [
        'data',
        'src',
        'results',
        'reports',
        'models'
    ]
    
    all_good = True
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"❌ {dir_path}/ directory missing")
            all_good = False
    
    return all_good

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Google Play Store Predictive Modeling Pipeline Tests")
    print("=" * 70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run the full pipeline: python run_predictive_modeling.py")
        print("2. Make custom predictions: python predict_new_app.py")
        print("3. Deploy the API: python deploy_api.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 