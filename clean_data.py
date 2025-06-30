#!/usr/bin/env python3
"""
Data Cleaning Script for Google Play Store Apps Dataset
Cleans both googleplaystore.csv and googleplaystore_user_reviews.csv
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_installs(installs_str):
    """Clean installs column by converting to numeric values"""
    if pd.isna(installs_str) or installs_str == 'Varies with device':
        return np.nan
    
    # Remove commas and plus signs
    installs_str = str(installs_str).replace(',', '').replace('+', '')
    
    # Convert to numeric
    try:
        return float(installs_str)
    except:
        return np.nan

def clean_size(size_str):
    """Clean size column by converting to MB"""
    if pd.isna(size_str) or size_str == 'Varies with device':
        return np.nan
    
    size_str = str(size_str).strip()
    
    # Extract number and unit
    match = re.match(r'([\d.]+)([kKmMgG])', size_str)
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
        else:
            return np.nan
    
    # Try to convert directly to float (already in MB)
    try:
        return float(size_str)
    except:
        return np.nan

def clean_price(price_str):
    """Clean price column by converting to numeric USD"""
    if pd.isna(price_str) or price_str == '0':
        return 0.0
    
    price_str = str(price_str).strip()
    
    # Remove dollar sign and convert to float
    if price_str.startswith('$'):
        try:
            return float(price_str[1:])
        except:
            return 0.0
    
    try:
        return float(price_str)
    except:
        return 0.0

def clean_rating(rating):
    """Clean rating column"""
    if pd.isna(rating):
        return np.nan
    
    try:
        rating = float(rating)
        # Ensure rating is between 0 and 5
        if 0 <= rating <= 5:
            return rating
        else:
            return np.nan
    except:
        return np.nan

def clean_reviews(reviews):
    """Clean reviews column"""
    if pd.isna(reviews):
        return np.nan
    
    try:
        return int(float(reviews))
    except:
        return np.nan

def clean_android_version(version):
    """Clean Android version column"""
    if pd.isna(version) or version == 'Varies with device':
        return 'Unknown'
    
    # Extract the minimum version number
    match = re.search(r'(\d+\.\d+)', str(version))
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def clean_date(date_str):
    """Clean date column"""
    if pd.isna(date_str):
        return np.nan
    
    try:
        # Parse common date formats
        date_formats = ['%B %d, %Y', '%B %d %Y', '%Y-%m-%d', '%d/%m/%Y']
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Try pandas default parsing
        return pd.to_datetime(date_str)
    except:
        return np.nan

def clean_googleplaystore_data():
    """Clean the main Google Play Store dataset"""
    print("Loading Google Play Store data...")
    
    # Read the data
    df = pd.read_csv('data/googleplaystore.csv')
    print(f"Original shape: {df.shape}")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Clean numeric columns
    print("Cleaning numeric columns...")
    df_clean['Rating'] = df_clean['Rating'].apply(clean_rating)
    df_clean['Reviews'] = df_clean['Reviews'].apply(clean_reviews)
    df_clean['Size_MB'] = df_clean['Size'].apply(clean_size)
    df_clean['Installs_Numeric'] = df_clean['Installs'].apply(clean_installs)
    df_clean['Price_USD'] = df_clean['Price'].apply(clean_price)
    
    # Clean date column
    print("Cleaning date column...")
    df_clean['Last_Updated_Clean'] = df_clean['Last Updated'].apply(clean_date)
    
    # Clean Android version
    print("Cleaning Android version...")
    df_clean['Android_Version_Clean'] = df_clean['Android Ver'].apply(clean_android_version)
    
    # Clean categorical columns
    print("Cleaning categorical columns...")
    df_clean['Type'] = df_clean['Type'].fillna('Unknown')
    df_clean['Content_Rating'] = df_clean['Content Rating'].fillna('Unknown')
    df_clean['Category'] = df_clean['Category'].fillna('Unknown')
    
    # Clean Genres column (split by semicolon and take first)
    df_clean['Primary_Genre'] = df_clean['Genres'].apply(
        lambda x: str(x).split(';')[0] if pd.notna(x) else 'Unknown'
    )
    
    # Handle missing values in Current Ver
    df_clean['Current_Version'] = df_clean['Current Ver'].fillna('Unknown')
    
    # Remove duplicates
    print("Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['App'], keep='first')
    final_rows = len(df_clean)
    print(f"Removed {initial_rows - final_rows} duplicate apps")
    
    # Create cleaned columns for analysis
    df_clean['Is_Free'] = (df_clean['Price_USD'] == 0.0).astype(int)
    df_clean['Has_Rating'] = df_clean['Rating'].notna().astype(int)
    df_clean['Has_Reviews'] = df_clean['Reviews'].notna().astype(int)
    
    # Select and reorder columns for cleaned dataset
    columns_to_keep = [
        'App', 'Category', 'Primary_Genre', 'Rating', 'Reviews', 
        'Size_MB', 'Installs_Numeric', 'Type', 'Price_USD', 'Is_Free',
        'Content_Rating', 'Last_Updated_Clean', 'Current_Version', 
        'Android_Version_Clean', 'Has_Rating', 'Has_Reviews'
    ]
    
    df_clean = df_clean[columns_to_keep]
    
    print(f"Cleaned shape: {df_clean.shape}")
    print("Google Play Store data cleaning completed!")
    
    return df_clean

def clean_user_reviews_data():
    """Clean the user reviews dataset"""
    print("\nLoading user reviews data...")
    
    # Read the data
    df = pd.read_csv('data/googleplaystore_user_reviews.csv')
    print(f"Original shape: {df.shape}")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Clean sentiment column
    print("Cleaning sentiment column...")
    df_clean['Sentiment'] = df_clean['Sentiment'].fillna('Unknown')
    
    # Clean translated review column
    print("Cleaning review text...")
    df_clean['Translated_Review'] = df_clean['Translated_Review'].fillna('')
    
    # Clean sentiment polarity and subjectivity
    print("Cleaning sentiment scores...")
    df_clean['Sentiment_Polarity'] = pd.to_numeric(df_clean['Sentiment_Polarity'], errors='coerce')
    df_clean['Sentiment_Subjectivity'] = pd.to_numeric(df_clean['Sentiment_Subjectivity'], errors='coerce')
    
    # Remove rows with completely missing sentiment data
    print("Removing rows with missing sentiment data...")
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['Sentiment_Polarity', 'Sentiment_Subjectivity'], how='all')
    final_rows = len(df_clean)
    print(f"Removed {initial_rows - final_rows} rows with missing sentiment data")
    
    # Remove duplicates
    print("Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    final_rows = len(df_clean)
    print(f"Removed {initial_rows - final_rows} duplicate reviews")
    
    # Create sentiment categories
    print("Creating sentiment categories...")
    def categorize_sentiment(polarity):
        if pd.isna(polarity):
            return 'Unknown'
        elif polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    df_clean['Sentiment_Category'] = df_clean['Sentiment_Polarity'].apply(categorize_sentiment)
    
    # Select and reorder columns
    columns_to_keep = [
        'App', 'Translated_Review', 'Sentiment', 'Sentiment_Category',
        'Sentiment_Polarity', 'Sentiment_Subjectivity'
    ]
    
    df_clean = df_clean[columns_to_keep]
    
    print(f"Cleaned shape: {df_clean.shape}")
    print("User reviews data cleaning completed!")
    
    return df_clean

def save_cleaned_data(df_apps, df_reviews):
    """Save cleaned datasets"""
    print("\nSaving cleaned datasets...")
    
    # Create cleaned directory if it doesn't exist
    os.makedirs('data/cleaned', exist_ok=True)
    
    # Save cleaned datasets
    df_apps.to_csv('data/cleaned/googleplaystore_cleaned.csv', index=False)
    df_reviews.to_csv('data/cleaned/googleplaystore_user_reviews_cleaned.csv', index=False)
    
    print("Cleaned datasets saved to:")
    print("- data/cleaned/googleplaystore_cleaned.csv")
    print("- data/cleaned/googleplaystore_user_reviews_cleaned.csv")

def generate_cleaning_report(df_apps_original, df_apps_clean, df_reviews_original, df_reviews_clean):
    """Generate a cleaning report"""
    print("\n" + "="*50)
    print("DATA CLEANING REPORT")
    print("="*50)
    
    print(f"\nGoogle Play Store Apps:")
    print(f"  Original records: {len(df_apps_original):,}")
    print(f"  Cleaned records: {len(df_apps_clean):,}")
    print(f"  Records removed: {len(df_apps_original) - len(df_apps_clean):,}")
    
    print(f"\nUser Reviews:")
    print(f"  Original records: {len(df_reviews_original):,}")
    print(f"  Cleaned records: {len(df_reviews_clean):,}")
    print(f"  Records removed: {len(df_reviews_original) - len(df_reviews_clean):,}")
    
    print(f"\nMissing Values in Cleaned Apps Dataset:")
    missing_counts = df_apps_clean.isnull().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        percentage = (count / len(df_apps_clean)) * 100
        print(f"  {col}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nMissing Values in Cleaned Reviews Dataset:")
    missing_counts = df_reviews_clean.isnull().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        percentage = (count / len(df_reviews_clean)) * 100
        print(f"  {col}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nSentiment Distribution:")
    sentiment_dist = df_reviews_clean['Sentiment_Category'].value_counts()
    for sentiment, count in sentiment_dist.items():
        percentage = (count / len(df_reviews_clean)) * 100
        print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function to run the data cleaning process"""
    print("Starting Google Play Store Data Cleaning Process...")
    print("="*60)
    
    try:
        # Clean Google Play Store data
        df_apps_original = pd.read_csv('data/googleplaystore.csv')
        df_apps_clean = clean_googleplaystore_data()
        
        # Clean user reviews data
        df_reviews_original = pd.read_csv('data/googleplaystore_user_reviews.csv')
        df_reviews_clean = clean_user_reviews_data()
        
        # Save cleaned data
        save_cleaned_data(df_apps_clean, df_reviews_clean)
        
        # Generate cleaning report
        generate_cleaning_report(df_apps_original, df_apps_clean, 
                               df_reviews_original, df_reviews_clean)
        
        print("\n" + "="*60)
        print("DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 