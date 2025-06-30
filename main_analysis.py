"""
Main Analysis Script for Google Play Store Apps EDA
Orchestrates the entire exploratory data analysis process
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from data_cleaner import DataCleaner
from visualizer import Visualizer
from analyzer import Analyzer
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Directories
CLEANED_DATA_DIR = 'data/cleaned'
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the complete EDA process
    """
    print("Starting Google Play Store Apps EDA")
    print("=" * 60)
    
    # Initialize components
    loader = DataLoader()
    cleaner = DataCleaner()
    visualizer = Visualizer()
    analyzer = Analyzer()
    
    # Step 1: Load Data
    print("\nSTEP 1: Loading Data")
    print("-" * 30)
    apps_df, reviews_df = loader.load_all_data()
    
    if apps_df is None or reviews_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Step 2: Explore Raw Data
    print("\nSTEP 2: Exploring Raw Data")
    print("-" * 30)
    exploration_results = loader.explore_datasets()
    
    # Step 3: Clean Data
    print("\nSTEP 3: Cleaning Data")
    print("-" * 30)
    apps_clean = cleaner.clean_apps_data(apps_df)
    reviews_clean = cleaner.clean_reviews_data(reviews_df)
    
    # Get cleaning summaries
    apps_cleaning_summary = cleaner.get_cleaning_summary(apps_df, apps_clean, "Apps Dataset")
    reviews_cleaning_summary = cleaner.get_cleaning_summary(reviews_df, reviews_clean, "Reviews Dataset")
    
    # Step 4: Perform Analysis
    print("\nSTEP 4: Performing Analysis")
    print("-" * 30)
    
    # App performance analysis
    performance_analysis = analyzer.analyze_app_performance(apps_clean)
    
    # Category analysis
    category_analysis = analyzer.analyze_category_performance(apps_clean)
    
    # Pricing analysis
    pricing_analysis = analyzer.analyze_pricing_strategy(apps_clean)
    
    # Sentiment analysis
    sentiment_analysis = analyzer.analyze_user_sentiment(reviews_clean, apps_clean)
    
    # Statistical tests
    statistical_tests = analyzer.perform_statistical_tests(apps_clean)
    
    # Clustering analysis
    clustering_results = analyzer.perform_clustering_analysis(apps_clean)
    
    # Generate comprehensive insights
    insights_report = analyzer.generate_insights_report(apps_clean, reviews_clean)
    
    # Step 5: Create Visualizations
    print("\nSTEP 5: Creating Visualizations")
    print("-" * 30)
    
    # Basic visualizations
    visualizer.plot_rating_distribution(apps_clean)
    visualizer.plot_category_analysis(apps_clean)
    visualizer.plot_price_analysis(apps_clean)
    visualizer.plot_install_analysis(apps_clean)
    visualizer.plot_sentiment_analysis(reviews_clean, apps_clean)
    visualizer.plot_correlation_matrix(apps_clean)
    
    # Interactive visualizations
    visualizer.plot_interactive_rating_trends(apps_clean)
    visualizer.create_dashboard_summary(apps_clean, reviews_clean)
    
    # Step 6: Print Analysis Summary
    print("\nSTEP 6: Analysis Summary")
    print("-" * 30)
    analyzer.print_analysis_summary(insights_report)
    
    # Step 7: Save Results
    print("\nSTEP 7: Saving Results")
    print("-" * 30)
    
    # Save cleaned data
    apps_clean.to_csv('data/apps_cleaned.csv', index=False)
    reviews_clean.to_csv('data/reviews_cleaned.csv', index=False)
    print("Cleaned data saved to data/ directory")
    
    # Save analysis results
    results = {
        'performance_analysis': performance_analysis,
        'category_analysis': category_analysis,
        'pricing_analysis': pricing_analysis,
        'sentiment_analysis': sentiment_analysis,
        'statistical_tests': statistical_tests,
        'clustering_results': clustering_results,
        'insights_report': insights_report,
        'cleaning_summaries': {
            'apps': apps_cleaning_summary,
            'reviews': reviews_cleaning_summary
        }
    }
    
    # Save results summary
    save_results_summary(results)
    
    print("\nEDA Process Completed Successfully!")
    print("=" * 60)
    print("Generated files:")
    print("  • data/apps_cleaned.csv - Cleaned apps data")
    print("  • data/reviews_cleaned.csv - Cleaned reviews data")
    print("  • reports/figures/ - All visualizations")
    print("  • results/insights_summary.md - Analysis summary")
    print("  • results/analysis_results.txt - Detailed results")
    
    return results

def save_results_summary(results):
    """
    Save analysis results to files
    
    Parameters:
    -----------
    results : dict
        Analysis results
    """
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save insights summary
    insights = results['insights_report']
    
    with open('results/insights_summary.md', 'w') as f:
        f.write("# Google Play Store Apps - Insights Summary\n\n")
        
        # Key metrics
        metrics = insights['key_metrics']
        f.write("## Key Metrics\n\n")
        f.write(f"- **Total Apps**: {metrics['total_apps']:,}\n")
        f.write(f"- **Total Reviews**: {metrics['total_reviews']:,}\n")
        f.write(f"- **Average Rating**: {metrics['avg_rating']:.2f}\n")
        f.write(f"- **Free Apps**: {metrics['free_apps_percentage']:.1f}%\n")
        f.write(f"- **Total Installations**: {metrics['total_installations']:,.0f}\n")
        f.write(f"- **Categories**: {metrics['total_categories']}\n\n")
        
        # Category insights
        cat_insights = insights['category_insights']
        f.write("## Category Insights\n\n")
        f.write(f"- **Most Popular Category**: {cat_insights['most_popular_category']}\n")
        f.write(f"- **Highest Rated Category**: {cat_insights['highest_rated_category']}\n")
        f.write(f"- **Most Installed Category**: {cat_insights['most_installed_category']}\n\n")
        
        # Pricing insights
        price_insights = insights['pricing_insights']
        f.write("## Pricing Insights\n\n")
        f.write(f"- **Free Apps**: {price_insights['free_apps_count']:,}\n")
        f.write(f"- **Paid Apps**: {price_insights['paid_apps_count']:,}\n")
        f.write(f"- **Average Paid Price**: ${price_insights['avg_paid_app_price']:.2f}\n")
        f.write(f"- **Rating Difference (Free - Paid)**: {price_insights['free_vs_paid_rating_diff']:.2f}\n\n")
        
        # Sentiment insights
        if 'sentiment_insights' in insights:
            sent_insights = insights['sentiment_insights']
            f.write("## Sentiment Insights\n\n")
            f.write(f"- **Positive Reviews**: {sent_insights['positive_sentiment_pct']:.1f}%\n")
            f.write(f"- **Negative Reviews**: {sent_insights['negative_sentiment_pct']:.1f}%\n")
            f.write(f"- **Average Polarity**: {sent_insights['avg_sentiment_polarity']:.3f}\n")
            f.write(f"- **Average Subjectivity**: {sent_insights['avg_sentiment_subjectivity']:.3f}\n\n")
        
        # Market trends
        market_trends = insights['market_trends']
        f.write("## Market Trends\n\n")
        f.write(f"- **Average App Size**: {market_trends['avg_app_size']:.1f} MB\n")
        f.write(f"- **Average Android Version**: {market_trends['avg_android_version']:.1f}\n")
        f.write(f"- **Recent Updates**: {market_trends['recent_updates_pct']:.1f}%\n\n")
        
        # Top performers
        top_performers = insights['top_performers']
        f.write("## Top Performers\n\n")
        
        f.write("### Highest Rated Apps\n")
        for _, row in top_performers['highest_rated_apps'].iterrows():
            f.write(f"- {row['App']} ({row['Category']}) - Rating: {row['Rating']:.2f}\n")
        
        f.write("\n### Most Installed Apps\n")
        for _, row in top_performers['most_installed_apps'].iterrows():
            f.write(f"- {row['App']} ({row['Category']}) - {row['Installs_Numeric']:,.0f} installs\n")
    
    # Save detailed results
    with open('results/analysis_results.txt', 'w') as f:
        f.write("Google Play Store Apps - Detailed Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        
        # Performance analysis
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(str(results['performance_analysis']['correlations']))
        f.write("\n\n")
        
        # Category analysis
        f.write("CATEGORY ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write("Top Rated Categories:\n")
        for cat, rating in results['category_analysis']['top_rated_categories'].items():
            f.write(f"  {cat}: {rating:.2f}\n")
        f.write("\n")
        
        # Statistical tests
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 15 + "\n")
        for test_name, test_result in results['statistical_tests'].items():
            f.write(f"{test_name}:\n")
            for key, value in test_result.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print("Cleaned data saved to results/ directory")

if __name__ == "__main__":
    main()

# Load cleaned data
df_apps = pd.read_csv(os.path.join(CLEANED_DATA_DIR, 'googleplaystore_cleaned.csv'))
df_reviews = pd.read_csv(os.path.join(CLEANED_DATA_DIR, 'googleplaystore_user_reviews_cleaned.csv'))

# 1. Summary Statistics
summary_stats = df_apps.describe(include='all').transpose()
summary_stats.to_csv(os.path.join(FIGURES_DIR, 'apps_summary_stats.csv'))

# 2. Missing Values
missing = df_apps.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
missing.to_csv(os.path.join(FIGURES_DIR, 'apps_missing_values.csv'))

# 3. Distribution of Ratings
plt.figure(figsize=(8,5))
sns.histplot(df_apps['Rating'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'rating_distribution.png'))
plt.close()

# 4. Distribution of Installs
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df_apps['Installs_Numeric'].dropna()), bins=30, kde=True, color='orange')
plt.title('Distribution of Log(Installs + 1)')
plt.xlabel('Log(Installs + 1)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'install_analysis.png'))
plt.close()

# 5. Free vs Paid Apps
plt.figure(figsize=(6,4))
sns.countplot(x='Is_Free', data=df_apps, palette='Set2')
plt.title('Free vs Paid Apps')
plt.xlabel('Is Free (1=Free, 0=Paid)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'free_vs_paid_analysis.png'))
plt.close()

# 6. Category Distribution
plt.figure(figsize=(12,6))
top_categories = df_apps['Category'].value_counts().head(15)
sns.barplot(x=top_categories.index, y=top_categories.values, palette='viridis')
plt.title('Top 15 App Categories')
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'category_distribution.png'))
plt.close()

# 7. Price Distribution (Paid Apps)
plt.figure(figsize=(8,5))
paid = df_apps[df_apps['Is_Free'] == 0]
sns.histplot(paid['Price_USD'], bins=30, kde=True, color='red')
plt.title('Price Distribution of Paid Apps')
plt.xlabel('Price (USD)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'price_analysis.png'))
plt.close()

# 8. Correlation Matrix
plt.figure(figsize=(10,8))
corr = df_apps[['Rating','Reviews','Size_MB','Installs_Numeric','Price_USD']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'))
plt.close()

# 9. Sentiment Analysis (User Reviews)
sentiment_counts = df_reviews['Sentiment_Category'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel')
plt.title('Sentiment Distribution in User Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_distribution.png'))
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(df_reviews['Sentiment_Polarity'].dropna(), bins=30, kde=True, color='purple')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_polarity.png'))
plt.close()

# 10. Category vs. Rating
plt.figure(figsize=(14,6))
top_cats = df_apps['Category'].value_counts().head(10).index
sns.boxplot(x='Category', y='Rating', data=df_apps[df_apps['Category'].isin(top_cats)], palette='Set3')
plt.title('App Rating by Top Categories')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'category_analysis.png'))
plt.close()

# 11. Save a summary report
with open(os.path.join(FIGURES_DIR, 'eda_summary.txt'), 'w') as f:
    f.write('Google Play Store Apps EDA Summary\n')
    f.write('='*40 + '\n')
    f.write(f'Total apps: {len(df_apps):,}\n')
    f.write(f'Total user reviews: {len(df_reviews):,}\n')
    f.write(f'Columns in apps: {list(df_apps.columns)}\n')
    f.write(f'Columns in reviews: {list(df_reviews.columns)}\n')
    f.write('\nMissing values in apps:\n')
    f.write(missing.to_string())
    f.write('\n\nSentiment distribution in reviews:\n')
    f.write(sentiment_counts.to_string())
    f.write('\n')

print('EDA completed. Summary statistics and plots saved to reports/figures/.') 