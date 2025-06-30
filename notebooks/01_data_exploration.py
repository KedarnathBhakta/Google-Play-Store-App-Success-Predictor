"""
Google Play Store Apps - Data Exploration
This script explores the Google Play Store datasets to understand the data structure and identify initial patterns.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from data_loader import DataLoader
from data_cleaner import DataCleaner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully!")

def main():
    print("Starting Data Exploration")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load data
    apps_df, reviews_df = loader.load_all_data()
    
    print(f"\nApps Dataset Shape: {apps_df.shape}")
    print(f"Reviews Dataset Shape: {reviews_df.shape}")
    
    # Explore datasets
    exploration_results = loader.explore_datasets()
    
    # Initialize data cleaner
    cleaner = DataCleaner()
    
    # Clean data
    apps_clean = cleaner.clean_apps_data(apps_df)
    reviews_clean = cleaner.clean_reviews_data(reviews_df)
    
    print(f"\nCleaned Apps Dataset Shape: {apps_clean.shape}")
    print(f"Cleaned Reviews Dataset Shape: {reviews_clean.shape}")
    
    # Basic statistics for apps dataset
    print("\nApps Dataset - Basic Statistics")
    print("=" * 50)
    print(apps_clean.describe())
    
    print("\nApps Dataset - Missing Values")
    print("=" * 50)
    missing_values = apps_clean.isnull().sum()
    missing_percentage = (missing_values / len(apps_clean)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Category distribution
    print("\nTop 15 Categories by Number of Apps")
    print("=" * 50)
    top_categories = apps_clean['Category'].value_counts().head(15)
    print(top_categories)
    
    # Visualize category distribution
    plt.figure(figsize=(15, 8))
    top_categories.plot(kind='barh', color='lightcoral')
    plt.title('Top 15 Categories by Number of Apps')
    plt.xlabel('Number of Apps')
    plt.ylabel('Category')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/category_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Rating distribution
    print("\nRating Distribution")
    print("=" * 50)
    rating_stats = apps_clean['Rating'].describe()
    print(rating_stats)
    
    # Visualize rating distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(apps_clean['Rating'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of App Ratings')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(apps_clean['Rating'].dropna(), patch_artist=True, 
               boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_ylabel('Rating')
    ax2.set_title('Box Plot of App Ratings')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Free vs Paid apps analysis
    print("\nFree vs Paid Apps Analysis")
    print("=" * 50)
    
    app_types = apps_clean['Type'].value_counts()
    print(f"App Types Distribution:")
    print(app_types)
    print(f"\nPercentage of Free Apps: {(app_types['Free'] / app_types.sum()) * 100:.1f}%")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    ax1.pie(app_types.values, labels=app_types.index, autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Free vs Paid Apps')
    
    # Average rating by type
    type_ratings = apps_clean.groupby('Type')['Rating'].mean()
    bars = ax2.bar(type_ratings.index, type_ratings.values, color=['lightgreen', 'lightcoral'])
    ax2.set_ylabel('Average Rating')
    ax2.set_title('Average Rating by App Type')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reports/figures/free_vs_paid_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Installation analysis
    print("\nInstallation Analysis")
    print("=" * 50)
    
    install_stats = apps_clean['Installs_Numeric'].describe()
    print(install_stats)
    
    # Create installation categories
    apps_clean['Install_Category'] = pd.cut(apps_clean['Installs_Numeric'], 
                                           bins=[0, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
                                           labels=['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M+'])
    
    # Visualize installation distribution
    install_counts = apps_clean['Install_Category'].value_counts()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(install_counts.index, install_counts.values, color='lightblue')
    plt.xlabel('Installation Range')
    plt.ylabel('Number of Apps')
    plt.title('Distribution of App Installations')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reports/figures/installation_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reviews analysis
    print("\nReviews Analysis")
    print("=" * 50)
    
    if len(reviews_clean) > 0:
        print(f"Total Reviews: {len(reviews_clean):,}")
        print(f"\nSentiment Distribution:")
        sentiment_dist = reviews_clean['Sentiment'].value_counts()
        print(sentiment_dist)
        
        # Visualize sentiment distribution
        plt.figure(figsize=(10, 6))
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        plt.pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%', colors=colors)
        plt.title('Overall Sentiment Distribution')
        plt.savefig('reports/figures/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Sentiment polarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reviews_clean['Sentiment_Polarity'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sentiment Polarity')
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/figures/sentiment_polarity.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No reviews data available")
    
    # Correlation analysis
    print("\nCorrelation Analysis")
    print("=" * 50)
    
    # Correlation matrix for numeric variables
    numeric_cols = ['Rating', 'Reviews', 'Size_MB', 'Installs_Numeric', 'Price_Numeric']
    numeric_df = apps_clean[numeric_cols].dropna()
    
    print("Correlation Matrix:")
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\nDATA EXPLORATION SUMMARY")
    print("=" * 50)
    print(f"• Apps Dataset: {apps_clean.shape[0]:,} apps, {apps_clean.shape[1]} features")
    print(f"• Reviews Dataset: {len(reviews_clean):,} reviews")
    print(f"• Categories: {apps_clean['Category'].nunique()}")
    print(f"• Average Rating: {apps_clean['Rating'].mean():.2f}")
    print(f"• Free Apps: {(apps_clean['Type'] == 'Free').mean() * 100:.1f}%")
    print(f"• Total Installations: {apps_clean['Installs_Numeric'].sum():,.0f}")
    
    if len(reviews_clean) > 0:
        print(f"• Positive Reviews: {(reviews_clean['Sentiment'] == 'Positive').mean() * 100:.1f}%")
        print(f"• Average Sentiment Polarity: {reviews_clean['Sentiment_Polarity'].mean():.3f}")
    
    print("\nData exploration completed! Ready for detailed analysis.")

if __name__ == "__main__":
    main() 