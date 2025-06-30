#!/usr/bin/env python3
"""
Predictive Modeling Pipeline for Google Play Store Apps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class AppSuccessPredictor:
    """
    Predictive modeling pipeline for Google Play Store app success prediction
    """
    
    def __init__(self, models_path="models"):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        models_path : str
            Path to save trained models
        """
        self.models_path = models_path
        self.models = {}
        self.preprocessors = {}
        self.feature_names = []
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs(models_path, exist_ok=True)
    
    def engineer_features(self, df):
        """
        Engineer features for predictive modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset with engineered features
        """
        print("Engineering features...")
        
        # Create a copy to avoid modifying original data
        df_eng = df.copy()
        
        # 1. App Age (days since last update)
        # Convert Last_Updated_Date to datetime if it's not already
        if 'Last_Updated_Date' in df_eng.columns:
            if df_eng['Last_Updated_Date'].dtype == 'object':
                df_eng['Last_Updated_Date'] = pd.to_datetime(df_eng['Last_Updated_Date'], errors='coerce')
            
            # Calculate app age in days
            df_eng['App_Age_Days'] = (pd.Timestamp.now() - df_eng['Last_Updated_Date']).dt.days
        else:
            df_eng['App_Age_Days'] = 0
        
        # 2. Price Category
        df_eng['Price_Category'] = pd.cut(df_eng['Price_Numeric'], 
                                         bins=[0, 0.01, 1, 5, 10, float('inf')],
                                         labels=['Free', 'Cheap', 'Low', 'Medium', 'High'])
        
        # 3. Size Category
        df_eng['Size_Category'] = pd.cut(df_eng['Size_MB'], 
                                        bins=[0, 10, 50, 100, float('inf')],
                                        labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        # 4. Install Category
        df_eng['Install_Category'] = pd.cut(df_eng['Installs_Numeric'], 
                                           bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # 5. Review Density (reviews per installation)
        df_eng['Review_Density'] = df_eng['Reviews'] / (df_eng['Installs_Numeric'] + 1)
        
        # 6. Category Popularity (number of apps in category)
        category_counts = df_eng['Category'].value_counts()
        df_eng['Category_Popularity'] = df_eng['Category'].map(category_counts)
        
        # 7. Content Rating Score
        rating_scores = {
            'Everyone': 1,
            'Everyone 10+': 2,
            'Teen': 3,
            'Mature 17+': 4,
            'Adults only 18+': 5,
            'Unrated': 0
        }
        df_eng['Content_Rating_Score'] = df_eng['Content Rating'].map(rating_scores)
        
        # 8. Android Version Score
        df_eng['Android_Version_Score'] = df_eng['Android_Min_Version']
        
        # 9. App Name Length
        df_eng['App_Name_Length'] = df_eng['App'].str.len()
        
        # 10. Has Reviews (binary)
        df_eng['Has_Reviews'] = (df_eng['Reviews'] > 0).astype(int)
        
        # 11. Success Score (composite metric)
        df_eng['Success_Score'] = (
            df_eng['Rating'] * 0.4 +
            np.log1p(df_eng['Reviews']) * 0.3 +
            np.log1p(df_eng['Installs_Numeric']) * 0.3
        )
        
        # 12. Popularity Score (based on installations)
        df_eng['Popularity_Score'] = np.log1p(df_eng['Installs_Numeric'])
        
        print(f"Engineered {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng
    
    def prepare_targets(self, df):
        """
        Prepare target variables for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with engineered features
            
        Returns:
        --------
        dict
            Dictionary of target variables
        """
        targets = {}
        
        # 1. Rating (regression)
        targets['Rating'] = df['Rating'].dropna()
        
        # 2. Popularity Score (regression)
        targets['Popularity_Score'] = df['Popularity_Score'].dropna()
        
        # 3. Success Classification (binary)
        # Define success as apps with rating > 4.0 and popularity > median
        median_popularity = df['Popularity_Score'].median()
        success_condition = (df['Rating'] > 4.0) & (df['Popularity_Score'] > median_popularity)
        targets['Success_Binary'] = success_condition.astype(int)
        
        # 4. High Rating Classification (binary)
        targets['High_Rating_Binary'] = (df['Rating'] > 4.0).astype(int)
        
        return targets
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with engineered features
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        # Select features for modeling
        feature_columns = [
            'Reviews', 'Size_MB', 'Price_Numeric', 'Android_Min_Version',
            'App_Age_Days', 'Category_Popularity', 'Content_Rating_Score',
            'Android_Version_Score', 'App_Name_Length', 'Has_Reviews',
            'Review_Density', 'Type'
        ]
        
        # Categorical features for encoding
        categorical_features = ['Type', 'Price_Category', 'Size_Category', 'Install_Category']
        
        # Add categorical features if they exist
        for col in categorical_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Remove duplicate columns
        feature_columns = list(dict.fromkeys(feature_columns))
        # Create feature matrix
        X = df[feature_columns].copy()
        
        # Convert all categorical columns to string type to avoid category mismatches
        for col in X.columns:
            if X.dtypes[col] == 'category':
                X[col] = X[col].astype(str)
        # Replace 'Unknown' with mode or empty string in object columns
        for col in X.columns:
            if X.dtypes[col] == object:
                X[col] = X[col].replace('Unknown', X[col].mode()[0] if not X[col].mode().empty else "")
        print("Feature columns and types:")
        print(X.dtypes)
        # Handle missing values: fill numeric with median, categorical with mode
        for col in X.columns:
            try:
                if X.dtypes[col] == object:
                    if not X[col].mode().empty:
                        X[col] = X[col].fillna(X[col].mode()[0])
                    else:
                        X[col] = X[col].fillna("")
                else:
                    X[col] = X[col].fillna(X[col].median())
            except Exception as e:
                print(f"Error processing column: {col}, dtype: {X.dtypes[col]}, error: {e}")
                raise
        
        self.feature_names = X.columns.tolist()
        return X
    
    def create_preprocessing_pipeline(self, X):
        """
        Create preprocessing pipeline for features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        sklearn.pipeline.Pipeline
            Preprocessing pipeline
        """
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing transformers
        preprocessors = []
        
        if numerical_features:
            preprocessors.append(('num', StandardScaler(), numerical_features))
        
        if categorical_features:
            preprocessors.append(('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features))
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_models(self, X, targets):
        """
        Train multiple models for different targets
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        targets : dict
            Dictionary of target variables
        """
        print("Training models...")
        
        for target_name, y in targets.items():
            if len(y) == 0:
                print(f"Skipping {target_name} - no valid target values")
                continue
                
            print(f"\nTraining model for {target_name}...")
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X_aligned = X.loc[common_index]
            y_aligned = y.loc[common_index]
            
            if len(X_aligned) == 0:
                print(f"Skipping {target_name} - no aligned data")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_aligned, y_aligned, test_size=0.2, random_state=42
            )
            
            # Create preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline(X_train)
            
            # Choose model based on target type
            if 'Binary' in target_name:
                # Classification model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
            else:
                # Regression model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            if 'Binary' in target_name:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"MSE: {mse:.3f}, R²: {r2:.3f}")
            
            # Store model and preprocessor
            self.models[target_name] = pipeline
            self.preprocessors[target_name] = preprocessor
            
            # Save model
            model_path = f"{self.models_path}/{target_name}_model.joblib"
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")
    
    def get_feature_importance(self, target_name):
        """
        Get feature importance for a specific model
        
        Parameters:
        -----------
        target_name : str
            Name of the target variable
            
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if target_name not in self.models:
            print(f"Model for {target_name} not found")
            return None
        
        model = self.models[target_name]
        
        # Get the actual model (not the pipeline)
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['classifier' if 'Binary' in target_name else 'regressor']
        else:
            actual_model = model
        
        # Get feature names after preprocessing
        feature_names = []
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps['preprocessor']
            for name, trans, cols in preprocessor.transformers_:
                if name == 'cat':
                    # For categorical features, get the encoded feature names
                    feature_names.extend([f"{col}_{val}" for col, vals in 
                                        zip(cols, trans.categories_) for val in vals[1:]])
                else:
                    feature_names.extend(cols)
        else:
            feature_names = self.feature_names
        
        # Get feature importance
        importance = actual_model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_new_app(self, app_features):
        """
        Predict success metrics for a new app
        
        Parameters:
        -----------
        app_features : dict
            Dictionary of app features
            
        Returns:
        --------
        dict
            Predictions for all targets
        """
        # Convert to dataframe
        app_df = pd.DataFrame([app_features])
        
        # Preprocess raw fields to numeric forms
        if 'Price' in app_df.columns:
            app_df['Price_Numeric'] = app_df['Price'].replace('[\$,]', '', regex=True).astype(float)
        if 'Size' in app_df.columns:
            def convert_size(size_str):
                if pd.isna(size_str) or size_str == 'Varies with device':
                    return np.nan
                import re
                match = re.match(r'(\d+(?:\.\d+)?)([kKmMgG])', str(size_str))
                if match:
                    number, unit = match.groups()
                    number = float(number)
                    if unit.lower() == 'k':
                        return number / 1024
                    elif unit.lower() == 'm':
                        return number
                    elif unit.lower() == 'g':
                        return number * 1024
                return np.nan
            app_df['Size_MB'] = app_df['Size'].apply(convert_size)
        if 'Installs' in app_df.columns:
            app_df['Installs_Numeric'] = app_df['Installs'].replace('[\+,]', '', regex=True).astype(float)
        if 'Android Ver' in app_df.columns:
            def extract_min_version(version_str):
                if pd.isna(version_str) or version_str == 'Varies with device':
                    return np.nan
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', str(version_str))
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        return np.nan
                return np.nan
            app_df['Android_Min_Version'] = app_df['Android Ver'].apply(extract_min_version)
        
        # Engineer features
        app_df = self.engineer_features(app_df)
        
        # Prepare features
        X = self.prepare_features(app_df)
        
        predictions = {}
        
        for target_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                predictions[target_name] = pred
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
                predictions[target_name] = None
        
        return predictions
    
    def generate_model_report(self, X, targets):
        """
        Generate comprehensive model report
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        targets : dict
            Dictionary of target variables
            
        Returns:
        --------
        dict
            Model performance report
        """
        report = {
            'model_performance': {},
            'feature_importance': {},
            'predictions_sample': {}
        }
        
        for target_name, y in targets.items():
            if target_name not in self.models:
                continue
                
            # Align data
            common_index = X.index.intersection(y.index)
            X_aligned = X.loc[common_index]
            y_aligned = y.loc[common_index]
            
            if len(X_aligned) == 0:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_aligned, y_aligned, test_size=0.2, random_state=42
            )
            
            # Get predictions
            model = self.models[target_name]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if 'Binary' in target_name:
                accuracy = accuracy_score(y_test, y_pred)
                report['model_performance'][target_name] = {
                    'accuracy': accuracy,
                    'target_type': 'classification'
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                report['model_performance'][target_name] = {
                    'mse': mse,
                    'r2': r2,
                    'target_type': 'regression'
                }
            
            # Get feature importance
            importance_df = self.get_feature_importance(target_name)
            if importance_df is not None:
                report['feature_importance'][target_name] = importance_df.head(10).to_dict('records')
        
        return report

# Example usage
if __name__ == "__main__":
    predictor = AppSuccessPredictor()
    print("Predictor module ready for use") 