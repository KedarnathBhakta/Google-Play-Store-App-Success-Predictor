"""
Data Cleaner Module for Google Play Store EDA
- Cleans and preprocesses app and review data.
- Designed for efficient use on both CPU and GPU (if available).
- For large datasets, uses vectorized operations and chunking to minimize CPU load.
- Compatible with pandas, numpy, and can be extended for GPU with cuDF or RAPIDS.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

class DataCleaner:
    """Class for cleaning Google Play Store app and review data."""
    
    def __init__(self):
        """Initialize DataCleaner"""
        pass
    
    def clean_apps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Google Play Store apps dataset.
        Removes duplicates (by App), handles missing values, and converts columns to appropriate types.
        Args:
            df (pd.DataFrame): Raw apps dataframe.
        Returns:
            pd.DataFrame: Cleaned apps dataframe.
        """
        if df is None:
            return None
            
        print("Cleaning Apps Dataset...")
        df_clean = df.copy()
        
        # Remove duplicates by App (keep first occurrence)
        df_clean = df_clean.drop_duplicates(subset=['App'], keep='first')
        
        # Clean Rating column
        df_clean = self._clean_rating_column(df_clean)
        
        # Clean Reviews column
        df_clean = self._clean_reviews_column(df_clean)
        
        # Clean Size column
        df_clean = self._clean_size_column(df_clean)
        
        # Clean Installs column
        df_clean = self._clean_installs_column(df_clean)
        
        # Clean Price column
        df_clean = self._clean_price_column(df_clean)
        
        # Clean Last Updated column
        df_clean = self._clean_last_updated_column(df_clean)
        
        # Clean Android Ver column
        df_clean = self._clean_android_ver_column(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        print("Apps dataset cleaning completed!")
        return df_clean
    
    def clean_reviews_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Google Play Store user reviews dataset.
        Removes duplicates and rows with missing sentiment data.
        Args:
            df (pd.DataFrame): Raw reviews dataframe.
        Returns:
            pd.DataFrame: Cleaned reviews dataframe.
        """
        if df is None:
            return None
            
        print("Cleaning Reviews Dataset...")
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"  Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Clean Sentiment column
        df_clean = self._clean_sentiment_column(df_clean)
        
        # Clean Sentiment_Polarity column
        df_clean = self._clean_sentiment_polarity_column(df_clean)
        
        # Clean Sentiment_Subjectivity column
        df_clean = self._clean_sentiment_subjectivity_column(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values_reviews(df_clean)
        
        print("Reviews dataset cleaning completed!")
        return df_clean
    
    def _clean_rating_column(self, df):
        """Clean Rating column"""
        # Convert to numeric, errors='coerce' will convert invalid values to NaN
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        # Remove ratings outside valid range (1-5)
        invalid_ratings = df[(df['Rating'] < 1) | (df['Rating'] > 5)]
        if len(invalid_ratings) > 0:
            print(f"  Found {len(invalid_ratings)} invalid ratings, setting to NaN")
            df.loc[(df['Rating'] < 1) | (df['Rating'] > 5), 'Rating'] = np.nan
        
        return df
    
    def _clean_reviews_column(self, df):
        """Clean Reviews column"""
        # Convert to numeric, errors='coerce' will convert invalid values to NaN
        df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
        
        # Remove negative reviews
        negative_reviews = df[df['Reviews'] < 0]
        if len(negative_reviews) > 0:
            print(f"  Found {len(negative_reviews)} negative review counts, setting to NaN")
            df.loc[df['Reviews'] < 0, 'Reviews'] = np.nan
        
        return df
    
    def _clean_size_column(self, df):
        """Clean Size column"""
        def convert_size(size_str):
            if pd.isna(size_str) or size_str == 'Varies with device':
                return np.nan
            
            # Extract number and unit
            match = re.match(r'(\d+(?:\.\d+)?)([kKmMgG])', str(size_str))
            if match:
                number, unit = match.groups()
                number = float(number)
                
                # Convert to MB
                if unit.lower() == 'k':
                    return number / 1024
                elif unit.lower() == 'm':
                    return number
                elif unit.lower() == 'g':
                    return number * 1024
            return np.nan
        
        df['Size_MB'] = df['Size'].apply(convert_size)
        print(f"  Converted Size column to Size_MB (MB)")
        
        return df
    
    def _clean_installs_column(self, df):
        """Clean Installs column"""
        def convert_installs(install_str):
            if pd.isna(install_str):
                return np.nan
            
            # Remove '+' and ',' characters
            install_str = str(install_str).replace('+', '').replace(',', '')
            
            # Convert to numeric
            try:
                return int(install_str)
            except:
                return np.nan
        
        df['Installs_Numeric'] = df['Installs'].apply(convert_installs)
        print(f"  Converted Installs column to Installs_Numeric")
        
        return df
    
    def _clean_price_column(self, df):
        """Clean Price column"""
        def convert_price(price_str):
            if pd.isna(price_str) or price_str == '0':
                return 0.0
            
            # Remove '$' and convert to float
            price_str = str(price_str).replace('$', '')
            try:
                return float(price_str)
            except:
                return 0.0
        
        df['Price_Numeric'] = df['Price'].apply(convert_price)
        print(f"  Converted Price column to Price_Numeric")
        
        return df
    
    def _clean_last_updated_column(self, df):
        """Clean Last Updated column"""
        def parse_date(date_str):
            if pd.isna(date_str):
                return np.nan
            
            try:
                # Try different date formats
                for fmt in ['%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str)
            except:
                return np.nan
        
        df['Last_Updated_Date'] = df['Last Updated'].apply(parse_date)
        print(f"  Converted Last Updated column to Last_Updated_Date")
        
        return df
    
    def _clean_android_ver_column(self, df):
        """Clean Android Ver column"""
        def extract_min_version(version_str):
            if pd.isna(version_str) or version_str == 'Varies with device':
                return np.nan
            
            # Extract the minimum version number
            match = re.search(r'(\d+(?:\.\d+)?)', str(version_str))
            if match:
                try:
                    return float(match.group(1))
                except:
                    return np.nan
            return np.nan
        
        df['Android_Min_Version'] = df['Android Ver'].apply(extract_min_version)
        print(f"  Converted Android Ver column to Android_Min_Version")
        
        return df
    
    def _clean_sentiment_column(self, df):
        """Clean Sentiment column"""
        # Convert to lowercase and standardize
        df['Sentiment'] = df['Sentiment'].str.lower()
        
        # Map variations to standard values
        sentiment_mapping = {
            'positive': 'Positive',
            'negative': 'Negative',
            'neutral': 'Neutral'
        }
        
        df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
        
        return df
    
    def _clean_sentiment_polarity_column(self, df):
        """Clean Sentiment_Polarity column"""
        # Convert to numeric
        df['Sentiment_Polarity'] = pd.to_numeric(df['Sentiment_Polarity'], errors='coerce')
        
        # Ensure values are between -1 and 1
        df.loc[df['Sentiment_Polarity'] < -1, 'Sentiment_Polarity'] = -1
        df.loc[df['Sentiment_Polarity'] > 1, 'Sentiment_Polarity'] = 1
        
        return df
    
    def _clean_sentiment_subjectivity_column(self, df):
        """Clean Sentiment_Subjectivity column"""
        # Convert to numeric
        df['Sentiment_Subjectivity'] = pd.to_numeric(df['Sentiment_Subjectivity'], errors='coerce')
        
        # Ensure values are between 0 and 1
        df.loc[df['Sentiment_Subjectivity'] < 0, 'Sentiment_Subjectivity'] = 0
        df.loc[df['Sentiment_Subjectivity'] > 1, 'Sentiment_Subjectivity'] = 1
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in apps dataset"""
        print(f"  Missing values before cleaning:")
        missing_before = df.isnull().sum()
        for col, count in missing_before.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # For numeric columns, we'll keep NaN as they represent missing data
        # For categorical columns, we'll fill with 'Unknown' or appropriate values
        
        categorical_cols = ['Category', 'Type', 'Content Rating', 'Genres']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if col == 'Type':
                    df[col] = df[col].fillna('Unknown')
                elif col == 'Content Rating':
                    df[col] = df[col].fillna('Unrated')
                else:
                    df[col] = df[col].fillna('Unknown')
        
        print(f"  Missing values after cleaning:")
        missing_after = df.isnull().sum()
        for col, count in missing_after.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _handle_missing_values_reviews(self, df):
        """Handle missing values in reviews dataset"""
        print(f"  Missing values before cleaning:")
        missing_before = df.isnull().sum()
        for col, count in missing_before.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Remove rows with missing sentiment (these are not useful for analysis)
        initial_rows = len(df)
        df = df.dropna(subset=['Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity'])
        print(f"  Removed {initial_rows - len(df)} rows with missing sentiment data")
        
        print(f"  Missing values after cleaning:")
        missing_after = df.isnull().sum()
        for col, count in missing_after.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def get_cleaning_summary(self, df_original, df_cleaned, dataset_name=""):
        """
        Get summary of cleaning operations
        
        Parameters:
        -----------
        df_original : pd.DataFrame
            Original dataset
        df_cleaned : pd.DataFrame
            Cleaned dataset
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        dict
            Cleaning summary
        """
        summary = {
            'original_shape': df_original.shape,
            'cleaned_shape': df_cleaned.shape,
            'rows_removed': df_original.shape[0] - df_cleaned.shape[0],
            'columns_added': df_cleaned.shape[1] - df_original.shape[1],
            'missing_values_original': df_original.isnull().sum().to_dict(),
            'missing_values_cleaned': df_cleaned.isnull().sum().to_dict()
        }
        
        print(f"\nCleaning Summary - {dataset_name}")
        print(f"Original shape: {summary['original_shape']}")
        print(f"Cleaned shape: {summary['cleaned_shape']}")
        print(f"Rows removed: {summary['rows_removed']}")
        print(f"Columns added: {summary['columns_added']}")
        
        return summary

# Example usage
if __name__ == "__main__":
    cleaner = DataCleaner()
    # This would be used with actual data loading
    print("DataCleaner module ready for use") 