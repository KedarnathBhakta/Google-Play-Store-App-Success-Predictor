#!/usr/bin/env python3
"""
Custom App Success Predictor

This script allows you to predict the success of new app ideas
using the trained models from the Google Play Store analysis.
"""

import sys
import pandas as pd
import numpy as np
import json
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def load_models():
    """Load trained models and preprocessing objects"""
    models = {}
    preprocessors = {}
    
    # Load models for different targets
    targets = ['Rating', 'Rating_Binary', 'Popularity_Score']
    model_types = ['catboost', 'xgboost']
    
    for target in targets:
        model_loaded = False
        for model_type in model_types:
            model_path = f'models/{target}_{model_type}.joblib'
            if os.path.exists(model_path):
                models[target] = joblib.load(model_path)
                preprocess_path = f'models/{target}_preprocessing.joblib'
                if os.path.exists(preprocess_path):
                    preprocessors[target] = joblib.load(preprocess_path)
                print(f"✅ Loaded {target} model ({model_type})")
                model_loaded = True
                break
        if not model_loaded:
            print(f"⚠️  Model for {target} not found. Run the full pipeline first.")
    
    return models, preprocessors

def get_app_input():
    """Get app details from user input"""
    print("\n🎯 Enter your app details:")
    print("-" * 40)
    
    app = {}
    
    # App name
    app['name'] = input("App name: ").strip()
    
    # Category
    print("\nAvailable categories:")
    categories = [
        'ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE',
        'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION',
        'ENTERTAINMENT', 'EVENTS', 'FAMILY', 'FINANCE', 'FOOD_AND_DRINK',
        'GAME', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME', 'LIBRARIES_AND_DEMO',
        'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'MEDICAL', 'MUSIC_AND_AUDIO',
        'NEWS_AND_MAGAZINES', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY',
        'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS', 'TRAVEL_AND_LOCAL',
        'VIDEO_PLAYERS', 'WEATHER'
    ]
    
    for i, cat in enumerate(categories, 1):
        print(f"{i:2d}. {cat}")
    
    while True:
        try:
            cat_choice = int(input(f"\nSelect category (1-{len(categories)}): "))
            if 1 <= cat_choice <= len(categories):
                app['category'] = categories[cat_choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # App type
    app['type'] = input("App type (Free/Paid): ").strip().title()
    
    # Price
    if app['type'] == 'Paid':
        while True:
            try:
                app['price'] = float(input("Price ($): "))
                break
            except ValueError:
                print("Please enter a valid price.")
    else:
        app['price'] = 0.0
    
    # Size
    while True:
        try:
            app['size_mb'] = float(input("App size (MB): "))
            break
        except ValueError:
            print("Please enter a valid size.")
    
    # Content rating
    print("\nContent rating options:")
    ratings = ['Everyone', 'Everyone 10+', 'Teen', 'Mature 17+']
    for i, rating in enumerate(ratings, 1):
        print(f"{i}. {rating}")
    
    while True:
        try:
            rating_choice = int(input(f"Select content rating (1-{len(ratings)}): "))
            if 1 <= rating_choice <= len(ratings):
                app['content_rating'] = ratings[rating_choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Android version
    while True:
        try:
            app['android_version'] = float(input("Minimum Android version (e.g., 4.4): "))
            break
        except ValueError:
            print("Please enter a valid version number.")
    
    return app

def prepare_features(app, preprocessors):
    """Prepare features for prediction"""
    # Base features
    features = {
        'Reviews': 1000,  # Initial reviews
        'Size_MB': app['size_mb'],
        'Installs_Numeric': 10000,  # Initial installs
        'Price_Numeric': app['price'],
        'App_Name_Length': len(app['name']),
        'App_Name_Word_Count': len(app['name'].split()),
        'Has_Special_Chars': int(any(c in app['name'] for c in '!@#$%^&*()')),
        'Category_Count': 1,
        'Days_Since_Update': 30,
        'Content_Rating_Strictness': {
            'Everyone': 1, 'Everyone 10+': 2, 'Teen': 3, 'Mature 17+': 4
        }.get(app['content_rating'], 1),
        'Android_Version_Numeric': app['android_version'],
        'Review_Ratio': 0.1,
        'Rating_Review_Ratio': 4.0 * 0.1,
        'Is_Free': int(app['type'] == 'Free'),
        'Last_Updated_Year': 2024,
        # Add categorical features
        'Category': app['category'],
        'Type': app['type'],
        'Content Rating': app['content_rating'],
        'Size_Category': pd.cut([app['size_mb']], bins=[0, 10, 50, 100, float('inf')], labels=['Small', 'Medium', 'Large', 'Very Large'])[0],
        'Price_Category': pd.cut([app['price']], bins=[0, 1, 5, float('inf')], labels=['Free/Low', 'Medium', 'High'])[0]
    }
    
    # Get feature names from any available preprocessor
    feature_names = None
    for target in preprocessors:
        if 'feature_names' in preprocessors[target]:
            feature_names = preprocessors[target]['feature_names']
            break
    
    if feature_names is not None:
        # Encode categorical features using label encoders
        for target in preprocessors:
            if 'label_encoders' in preprocessors[target]:
                label_encoders = preprocessors[target]['label_encoders']
                for cat_feat in ['Category', 'Type', 'Content Rating', 'Size_Category', 'Price_Category']:
                    if cat_feat in label_encoders and cat_feat in features:
                        le = label_encoders[cat_feat]
                        try:
                            features[cat_feat] = le.transform([features[cat_feat]])[0]
                        except Exception:
                            # If unseen label, use 0
                            features[cat_feat] = 0
                break
        
        # Add missing features with default values
        for feature in feature_names:
            if feature not in features:
                features[feature] = 0
        
        # Ensure correct order
        feature_vector = [features[feature] for feature in feature_names]
        return pd.DataFrame([feature_vector], columns=feature_names)
    else:
        # Fallback if no preprocessor info available
        feature_vector = list(features.values())
        return pd.DataFrame([feature_vector], columns=list(features.keys()))

def predict_app_success(app, models, preprocessors):
    """Predict success metrics for the app"""
    print(f"\n🔮 Predicting success for: {app['name']}")
    print("-" * 50)
    
    predictions = {}
    
    for target, model in models.items():
        try:
            # Prepare features
            feature_vector = prepare_features(app, preprocessors)
            
            # Scale features if scaler is available
            if target in preprocessors and 'scaler' in preprocessors[target]:
                scaler = preprocessors[target]['scaler']
                feature_vector_scaled = scaler.transform(feature_vector)
                pred = model.predict(feature_vector_scaled)[0]
            else:
                pred = model.predict(feature_vector)[0]
            
            predictions[target] = pred
            
            # Display prediction
            if target == 'Rating':
                print(f"📊 Predicted Rating: {pred:.2f}/5.0")
                if pred >= 4.0:
                    print("   🎉 Excellent rating potential!")
                elif pred >= 3.5:
                    print("   👍 Good rating potential")
                else:
                    print("   ⚠️  May need improvements")
                    
            elif target == 'Rating_Binary':
                success_prob = pred if hasattr(model, 'predict_proba') else pred
                print(f"📈 Success Probability: {success_prob:.1%}")
                if success_prob >= 0.7:
                    print("   🚀 High success potential!")
                elif success_prob >= 0.5:
                    print("   📈 Moderate success potential")
                else:
                    print("   🔄 Consider strategy adjustments")
                    
            elif target == 'Popularity_Score':
                print(f"🔥 Popularity Score: {pred:.2f}")
                if pred >= 50:
                    print("   🌟 High popularity potential!")
                elif pred >= 30:
                    print("   📈 Good popularity potential")
                else:
                    print("   📊 Moderate popularity potential")
                    
        except Exception as e:
            print(f"⚠️  Error predicting {target}: {str(e)}")
            predictions[target] = None
    
    return predictions

def provide_recommendations(app, predictions):
    """Provide recommendations based on predictions"""
    print(f"\n💡 Recommendations for {app['name']}:")
    print("-" * 50)
    
    recommendations = []
    
    # Rating recommendations
    if 'Rating' in predictions and predictions['Rating'] is not None:
        rating = predictions['Rating']
        if rating < 3.5:
            recommendations.append("🎯 Focus on user experience and app quality to improve ratings")
            recommendations.append("📱 Ensure smooth performance and intuitive interface")
        elif rating < 4.0:
            recommendations.append("✨ Add unique features to stand out from competitors")
            recommendations.append("🔄 Regular updates can help maintain good ratings")
        else:
            recommendations.append("🌟 Excellent rating potential! Focus on marketing and user acquisition")
    
    # Success probability recommendations
    if 'Rating_Binary' in predictions and predictions['Rating_Binary'] is not None:
        success_prob = predictions['Rating_Binary']
        if success_prob < 0.5:
            recommendations.append("📊 Consider market research to understand user needs better")
            recommendations.append("💰 Free apps typically have higher adoption rates")
        elif success_prob < 0.7:
            recommendations.append("📈 Focus on user engagement and retention strategies")
            recommendations.append("🎮 Consider gamification elements to increase user interaction")
        else:
            recommendations.append("🚀 High success potential! Invest in marketing and user acquisition")
    
    # Popularity recommendations
    if 'Popularity_Score' in predictions and predictions['Popularity_Score'] is not None:
        popularity = predictions['Popularity_Score']
        if popularity < 30:
            recommendations.append("📢 Develop a strong marketing strategy")
            recommendations.append("🎯 Target specific user segments")
        elif popularity < 50:
            recommendations.append("🔥 Focus on viral features and social sharing")
            recommendations.append("📱 Optimize for app store visibility")
        else:
            recommendations.append("🌟 High popularity potential! Focus on scaling and monetization")
    
    # General recommendations
    if app['type'] == 'Paid':
        recommendations.append("💡 Consider freemium model for better user acquisition")
    
    if app['size_mb'] > 50:
        recommendations.append("📦 Consider optimizing app size for better download rates")
    
    if app['android_version'] > 5.0:
        recommendations.append("📱 Lower minimum Android version can increase potential user base")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return recommendations

def save_prediction(app, predictions, recommendations):
    """Save prediction results"""
    result = {
        'app_details': app,
        'predictions': predictions,
        'recommendations': recommendations,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to file
    with open('results/new_app_prediction.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 Prediction saved to results/new_app_prediction.json")

def main():
    """Main function"""
    print("🎯 Google Play Store App Success Predictor")
    print("=" * 50)
    print("This tool predicts the success of your app idea using")
    print("machine learning models trained on Google Play Store data.")
    
    # Load models
    print("\n📚 Loading trained models...")
    models, preprocessors = load_models()
    
    if not models:
        print("\n❌ No models found. Please run the full pipeline first:")
        print("   python run_predictive_modeling.py")
        return
    
    # Get app input
    app = get_app_input()
    
    # Make predictions
    predictions = predict_app_success(app, models, preprocessors)
    
    # Provide recommendations
    recommendations = provide_recommendations(app, predictions)
    
    # Save results
    save_prediction(app, predictions, recommendations)
    
    print(f"\n🎉 Analysis complete! Check results/new_app_prediction.json for details.")

if __name__ == "__main__":
    main() 