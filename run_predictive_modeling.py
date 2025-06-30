#!/usr/bin/env python3
"""
Comprehensive Predictive Modeling for Google Play Store Apps
Builds models for multiple prediction targets:
1. Rating Prediction (Regression)
2. High/Low Rating Classification
3. Popularity Prediction (Installs)
4. Success Score Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set up directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class AppStorePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("Loading cleaned data...")
        
        # Load cleaned data
        self.df_apps = pd.read_csv('data/cleaned/googleplaystore_cleaned.csv')
        self.df_reviews = pd.read_csv('data/cleaned/googleplaystore_user_reviews_cleaned.csv')
        
        print(f"Apps dataset: {self.df_apps.shape}")
        print(f"Reviews dataset: {self.df_reviews.shape}")
        
        # Create features
        self.create_features()
        
        # Prepare targets
        self.prepare_targets()
        
        print("Data preparation completed!")
        
    def create_features(self):
        """Create features for modeling"""
        print("Creating features...")
        
        # Start with apps data
        self.features_df = self.df_apps.copy()
        
        # Create success score (combination of rating and popularity)
        self.features_df['Success_Score'] = (
            self.features_df['Rating'].fillna(0) * 
            np.log1p(self.features_df['Installs_Numeric'].fillna(0))
        )
        
        # Create binary targets
        self.features_df['High_Rating'] = (self.features_df['Rating'] >= 4.0).astype(int)
        self.features_df['Popular'] = (self.features_df['Installs_Numeric'] >= 1000000).astype(int)
        self.features_df['Successful'] = (self.features_df['Success_Score'] >= self.features_df['Success_Score'].median()).astype(int)
        
        # Add review-based features
        review_features = self.df_reviews.groupby('App').agg({
            'Sentiment_Polarity': ['mean', 'std', 'count'],
            'Sentiment_Subjectivity': ['mean', 'std'],
            'Sentiment_Category': lambda x: (x == 'Positive').mean()
        }).reset_index()
        
        review_features.columns = ['App', 'Avg_Sentiment', 'Sentiment_Std', 'Review_Count', 
                                 'Avg_Subjectivity', 'Subjectivity_Std', 'Positive_Ratio']
        
        # Merge with main features
        self.features_df = self.features_df.merge(review_features, on='App', how='left')
        
        # Fill missing review features
        self.features_df[['Avg_Sentiment', 'Sentiment_Std', 'Review_Count', 
                         'Avg_Subjectivity', 'Subjectivity_Std', 'Positive_Ratio']] = \
            self.features_df[['Avg_Sentiment', 'Sentiment_Std', 'Review_Count', 
                             'Avg_Subjectivity', 'Subjectivity_Std', 'Positive_Ratio']].fillna(0)
        
        print(f"Features created: {self.features_df.shape}")
        
    def prepare_targets(self):
        """Prepare target variables"""
        print("Preparing target variables...")
        
        # Define targets
        self.targets = {
            'rating': 'Rating',
            'high_rating': 'High_Rating', 
            'popularity': 'Installs_Numeric',
            'success_score': 'Success_Score'
        }
        
        # Remove rows with missing targets
        for target_name, target_col in self.targets.items():
            if target_name == 'rating':
                # For rating, remove rows with missing ratings
                mask = self.features_df[target_col].notna()
            elif target_name == 'popularity':
                # For popularity, remove rows with missing installs
                mask = self.features_df[target_col].notna()
            else:
                # For binary targets, keep all rows
                mask = self.features_df[target_col].notna()
            
            print(f"{target_name}: {mask.sum()} valid samples")
            
    def prepare_features_for_modeling(self, target_name):
        """Prepare features for a specific target"""
        print(f"Preparing features for {target_name} prediction...")
        
        # Select features
        feature_cols = [
            'Category', 'Primary_Genre', 'Size_MB', 'Is_Free', 'Price_USD',
            'Content_Rating', 'Reviews', 'Has_Rating', 'Has_Reviews',
            'Avg_Sentiment', 'Sentiment_Std', 'Review_Count', 
            'Avg_Subjectivity', 'Subjectivity_Std', 'Positive_Ratio'
        ]
        
        # Add target-specific features
        if target_name == 'rating':
            # For rating prediction, exclude installs (circular dependency)
            pass
        elif target_name == 'popularity':
            # For popularity prediction, exclude rating (circular dependency)
            feature_cols = [col for col in feature_cols if col not in ['Has_Rating']]
        else:
            # For other targets, include all features
            pass
        
        # Prepare X and y
        X = self.features_df[feature_cols].copy()
        y = self.features_df[self.targets[target_name]].copy()
        
        # Fill NaNs in categorical columns and ensure string type
        categorical_features = ['Category', 'Primary_Genre', 'Content_Rating']
        for col in categorical_features:
            X[col] = X[col].fillna('Unknown').astype(str)
        
        # Remove rows with missing target
        if target_name in ['rating', 'popularity']:
            mask = y.notna()
            X = X[mask]
            y = y[mask]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
        
    def build_model_pipeline(self, target_name, X, y):
        """Build and train model for a specific target"""
        print(f"Building model for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if target_name in ['high_rating'] else None
        )
        
        # Define preprocessing
        categorical_features = ['Category', 'Primary_Genre', 'Content_Rating']
        numerical_features = [col for col in X.columns if col not in categorical_features]
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ])
        
        # Choose model based on target type
        if target_name in ['rating', 'popularity', 'success_score']:
            # Regression models
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression()
            }
        else:
            # Classification models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
            }
        
        # Train and evaluate models
        best_model = None
        best_score = -np.inf
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            if target_name in ['rating', 'popularity', 'success_score']:
                # Regression metrics
                y_pred = pipeline.predict(X_test)
                score = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print(f"    R² Score: {score:.4f}")
                print(f"    RMSE: {rmse:.4f}")
            else:
                # Classification metrics
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
                score = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                print(f"    Accuracy: {score:.4f}")
                if auc:
                    print(f"    AUC: {auc:.4f}")
            
            # Store best model
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_model_name = model_name
        
        # Store results
        self.models[target_name] = best_model
        self.results[target_name] = {
            'model_name': best_model_name,
            'score': best_score,
            'X_test': X_test,
            'y_test': y_test
        }
        
        print(f"  Best model: {best_model_name} (Score: {best_score:.4f})")
        
        return best_model, best_score
        
    def get_feature_importance(self, target_name):
        """Extract feature importance from the best model"""
        print(f"Extracting feature importance for {target_name}...")
        
        model = self.models[target_name]
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        feature_names = []

        # Numeric features
        num_pipeline = preprocessor.named_transformers_['num']
        num_features = self.numerical_features  # original feature names
        feature_names.extend(num_features)

        # Categorical features
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_features)
        
        # Get feature importance
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importance = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            importance = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            importance = None
            
        if importance is not None:
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[target_name] = feature_importance_df
            
            # Save feature importance
            feature_importance_df.to_csv(f'results/feature_importance_{target_name}.csv', index=False)
            
            return feature_importance_df
        
        return None
        
    def create_visualizations(self, target_name):
        """Create visualizations for model results"""
        print(f"Creating visualizations for {target_name}...")
        
        # Feature importance plot
        if target_name in self.feature_importance:
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance[target_name].head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Features for {target_name.replace("_", " ").title()} Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'reports/figures/feature_importance_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Prediction vs Actual plot (for regression)
        if target_name in ['rating', 'popularity', 'success_score']:
            model = self.models[target_name]
            X_test = self.results[target_name]['X_test']
            y_test = self.results[target_name]['y_test']
            y_pred = model.predict(X_test)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{target_name.replace("_", " ").title()} Prediction vs Actual')
            plt.tight_layout()
            plt.savefig(f'reports/figures/prediction_vs_actual_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        
        for target_name, model in self.models.items():
            model_path = f'models/{target_name}_model.joblib'
            joblib.dump(model, model_path)
            print(f"  Saved {target_name} model to {model_path}")
            
    def generate_report(self):
        """Generate comprehensive modeling report"""
        print("Generating modeling report...")
        
        report = []
        report.append("# Google Play Store Apps - Predictive Modeling Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary of models
        report.append("## Model Performance Summary")
        report.append("")
        
        for target_name, result in self.results.items():
            report.append(f"### {target_name.replace('_', ' ').title()}")
            report.append(f"- **Model:** {result['model_name']}")
            report.append(f"- **Score:** {result['score']:.4f}")
            report.append("")
        
        # Feature importance summary
        report.append("## Top Features by Target")
        report.append("")
        
        for target_name in self.results.keys():
            if target_name in self.feature_importance:
                report.append(f"### {target_name.replace('_', ' ').title()}")
                top_features = self.feature_importance[target_name].head(5)
                for _, row in top_features.iterrows():
                    report.append(f"- **{row['feature']}:** {row['importance']:.4f}")
                report.append("")
        
        # Save report
        with open('results/modeling_report.md', 'w') as f:
            f.write('\n'.join(report))
            
        print("Modeling report saved to results/modeling_report.md")
        
    def run_complete_pipeline(self):
        """Run the complete predictive modeling pipeline"""
        print("Starting Predictive Modeling Pipeline")
        print("=" * 50)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Build models for each target
        for target_name in self.targets.keys():
            print(f"\n{'='*20} {target_name.upper()} {'='*20}")
            
            # Prepare features
            X, y = self.prepare_features_for_modeling(target_name)
            
            # Build model
            model, score = self.build_model_pipeline(target_name, X, y)
            
            # Get feature importance
            self.get_feature_importance(target_name)
            
            # Create visualizations
            self.create_visualizations(target_name)
            
        # Save models
        self.save_models()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "=" * 50)
        print("PREDICTIVE MODELING PIPELINE COMPLETED!")
        print("=" * 50)
        print("\nGenerated files:")
        print("- models/ - Trained models")
        print("- results/ - Model results and reports")
        print("- reports/figures/ - Model visualizations")

if __name__ == "__main__":
    # Run the complete pipeline
    predictor = AppStorePredictor()
    predictor.run_complete_pipeline() 