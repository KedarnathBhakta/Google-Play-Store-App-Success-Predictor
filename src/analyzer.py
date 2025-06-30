"""
Analyzer Module for Google Play Store EDA
- Performs statistical and business analysis on app and review data.
- Designed for efficient use on both CPU and GPU (if available).
- Uses vectorized operations and can be extended for GPU with RAPIDS or cuML.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
from typing import Dict, Any
warnings.filterwarnings('ignore')

class Analyzer:
    """Class for analyzing Google Play Store app and review data."""
    
    def __init__(self):
        """Initialize Analyzer"""
        pass
    
    def analyze_app_performance(self, apps_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze app performance metrics (e.g., average rating, installs).
        Args:
            apps_data (pd.DataFrame): Cleaned apps dataframe.
        Returns:
            Dict[str, Any]: Summary statistics and insights.
        """
        print("Analyzing App Performance...")
        print("=" * 40)
        
        analysis = {}
        
        # Rating analysis
        rating_stats = apps_data['Rating'].describe()
        analysis['rating_stats'] = rating_stats
        
        # Reviews analysis
        review_stats = apps_data['Reviews'].describe()
        analysis['review_stats'] = review_stats
        
        # Installation analysis
        install_stats = apps_data['Installs_Numeric'].describe()
        analysis['install_stats'] = install_stats
        
        # Size analysis
        size_stats = apps_data['Size_MB'].describe()
        analysis['size_stats'] = size_stats
        
        # Performance correlations
        numeric_cols = ['Rating', 'Reviews', 'Installs_Numeric', 'Size_MB']
        correlations = apps_data[numeric_cols].corr()
        analysis['correlations'] = correlations
        
        # Example: Calculate average rating and installs
        avg_rating = apps_data['Rating'].mean()
        total_installs = apps_data['Installs_Numeric'].sum()
        analysis['average_rating'] = avg_rating
        analysis['total_installs'] = total_installs
        
        print("App performance analysis completed!")
        return analysis
    
    def analyze_category_performance(self, df):
        """
        Analyze performance by category
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
            
        Returns:
        --------
        dict
            Category analysis results
        """
        print("Analyzing Category Performance...")
        print("=" * 40)
        
        analysis = {}
        
        # Category statistics
        category_stats = df.groupby('Category').agg({
            'Rating': ['mean', 'std', 'count'],
            'Reviews': ['mean', 'sum'],
            'Installs_Numeric': ['mean', 'sum'],
            'Price_Numeric': ['mean', 'sum']
        }).round(2)
        
        analysis['category_stats'] = category_stats
        
        # Top performing categories by rating
        top_rated_categories = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
        analysis['top_rated_categories'] = top_rated_categories
        
        # Most popular categories by installs
        most_popular_categories = df.groupby('Category')['Installs_Numeric'].sum().sort_values(ascending=False).head(10)
        analysis['most_popular_categories'] = most_popular_categories
        
        # Categories with most reviews
        most_reviewed_categories = df.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(10)
        analysis['most_reviewed_categories'] = most_reviewed_categories
        
        print("Category performance analysis completed!")
        return analysis
    
    def analyze_pricing_strategy(self, df):
        """
        Analyze pricing strategy and its impact
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
            
        Returns:
        --------
        dict
            Pricing analysis results
        """
        print("Analyzing Pricing Strategy...")
        print("=" * 40)
        
        analysis = {}
        
        # Price distribution
        price_stats = df['Price_Numeric'].describe()
        analysis['price_stats'] = price_stats
        
        # Free vs Paid comparison
        type_comparison = df.groupby('Type').agg({
            'Rating': ['mean', 'std', 'count'],
            'Reviews': ['mean', 'sum'],
            'Installs_Numeric': ['mean', 'sum']
        }).round(2)
        analysis['type_comparison'] = type_comparison
        
        # Price vs Performance correlation
        paid_apps = df[df['Price_Numeric'] > 0]
        if len(paid_apps) > 0:
            price_rating_corr = paid_apps['Price_Numeric'].corr(paid_apps['Rating'])
            price_install_corr = paid_apps['Price_Numeric'].corr(paid_apps['Installs_Numeric'])
            analysis['price_rating_correlation'] = price_rating_corr
            analysis['price_install_correlation'] = price_install_corr
        
        # Price ranges analysis
        df['Price_Range'] = pd.cut(df['Price_Numeric'], 
                                  bins=[0, 1, 5, 10, 20, 50, float('inf')],
                                  labels=['Free', '$1-5', '$5-10', '$10-20', '$20-50', '$50+'])
        
        price_range_stats = df.groupby('Price_Range').agg({
            'Rating': ['mean', 'count'],
            'Installs_Numeric': ['mean', 'sum']
        }).round(2)
        analysis['price_range_stats'] = price_range_stats
        
        print("Pricing strategy analysis completed!")
        return analysis
    
    def analyze_user_sentiment(self, reviews_df, apps_df):
        """
        Analyze user sentiment patterns
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            Reviews dataset
        apps_df : pd.DataFrame
            Apps dataset
            
        Returns:
        --------
        dict
            Sentiment analysis results
        """
        print("Analyzing User Sentiment...")
        
        analysis = {}
        
        # Overall sentiment distribution
        sentiment_dist = reviews_df['Sentiment'].value_counts(normalize=True)
        analysis['sentiment_distribution'] = sentiment_dist
        
        # Sentiment statistics
        sentiment_stats = reviews_df[['Sentiment_Polarity', 'Sentiment_Subjectivity']].describe()
        analysis['sentiment_stats'] = sentiment_stats
        
        # Sentiment by category
        merged_data = reviews_df.merge(apps_df[['App', 'Category']], on='App', how='inner')
        category_sentiment = merged_data.groupby('Category').agg({
            'Sentiment_Polarity': ['mean', 'std', 'count'],
            'Sentiment_Subjectivity': ['mean', 'std']
        }).round(3)
        analysis['category_sentiment'] = category_sentiment
        
        # Top positive and negative categories
        top_positive_categories = merged_data.groupby('Category')['Sentiment_Polarity'].mean().sort_values(ascending=False).head(10)
        top_negative_categories = merged_data.groupby('Category')['Sentiment_Polarity'].mean().sort_values(ascending=True).head(10)
        
        analysis['top_positive_categories'] = top_positive_categories
        analysis['top_negative_categories'] = top_negative_categories
        
        # Sentiment vs App Performance
        app_sentiment = merged_data.groupby('App').agg({
            'Sentiment_Polarity': 'mean',
            'Sentiment_Subjectivity': 'mean',
            'Sentiment': 'count'
        }).reset_index()
        
        app_performance = apps_df[['App', 'Rating', 'Reviews', 'Installs_Numeric']]
        sentiment_performance = app_sentiment.merge(app_performance, on='App', how='inner')
        
        # Correlations
        sentiment_correlations = sentiment_performance[['Sentiment_Polarity', 'Sentiment_Subjectivity', 'Rating', 'Reviews', 'Installs_Numeric']].corr()
        analysis['sentiment_performance_correlations'] = sentiment_correlations
        
        print("User sentiment analysis completed!")
        return analysis
    
    def perform_statistical_tests(self, apps_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical tests on the cleaned data.
        Args:
            apps_data (pd.DataFrame): Cleaned apps dataframe.
        Returns:
            Dict[str, Any]: Results of statistical tests.
        """
        print("Performing Statistical Tests...")
        
        # Remove rows with missing values for statistical tests
        test_data = apps_data.dropna(subset=['Rating', 'Reviews', 'Installs_Numeric', 'Price_Numeric'])
        
        results = {}
        
        # Example: Correlation between Rating and Reviews
        if len(test_data) > 1:
            rating_reviews_corr, rating_reviews_p = stats.pearsonr(
                test_data['Rating'], test_data['Reviews']
            )
            results['rating_reviews_correlation'] = {
                'correlation': rating_reviews_corr,
                'p_value': rating_reviews_p,
                'interpretation': 'Strong positive correlation' if abs(rating_reviews_corr) > 0.7 else 
                                'Moderate correlation' if abs(rating_reviews_corr) > 0.3 else 'Weak correlation'
            }
        
        # 2. Correlation between Rating and Installs
        if len(test_data) > 1:
            rating_installs_corr, rating_installs_p = stats.pearsonr(
                test_data['Rating'], test_data['Installs_Numeric']
            )
            results['rating_installs_correlation'] = {
                'correlation': rating_installs_corr,
                'p_value': rating_installs_p,
                'interpretation': 'Strong positive correlation' if abs(rating_installs_corr) > 0.7 else 
                                'Moderate correlation' if abs(rating_installs_corr) > 0.3 else 'Weak correlation'
            }
        
        # 3. T-test: Free vs Paid apps ratings
        free_apps = test_data[test_data['Type'] == 'Free']['Rating'].dropna()
        paid_apps = test_data[test_data['Type'] == 'Paid']['Rating'].dropna()
        
        if len(free_apps) > 0 and len(paid_apps) > 0:
            t_stat, p_value = stats.ttest_ind(free_apps, paid_apps)
            results['free_vs_paid_ratings'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'free_mean': free_apps.mean(),
                'paid_mean': paid_apps.mean(),
                'significant': p_value < 0.05
            }
        
        # 4. ANOVA: Ratings across categories
        categories = test_data['Category'].unique()
        category_ratings = []
        category_names = []
        
        for category in categories[:10]:  # Limit to top 10 categories for performance
            cat_ratings = test_data[test_data['Category'] == category]['Rating'].dropna()
            if len(cat_ratings) > 5:  # Only include categories with sufficient data
                category_ratings.append(cat_ratings)
                category_names.append(category)
        
        if len(category_ratings) > 1:
            f_stat, p_value = stats.f_oneway(*category_ratings)
            results['category_ratings_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'categories_tested': len(category_names),
                'significant': p_value < 0.05
            }
        
        # 5. Chi-square test: App Type vs Content Rating
        contingency_table = pd.crosstab(test_data['Type'], test_data['Content Rating'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            results['type_content_rating_chi2'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
        
        print("Statistical tests completed!")
        return results
    
    def perform_clustering_analysis(self, df, n_clusters=5):
        """
        Perform clustering analysis to identify app segments
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        dict
            Clustering results
        """
        print("Performing Clustering Analysis...")
        
        # Prepare data for clustering
        features = ['Rating', 'Reviews', 'Installs_Numeric', 'Size_MB', 'Price_Numeric']
        clustering_data = df[features].dropna()
        
        if len(clustering_data) == 0:
            print("No data available for clustering")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        clustering_data['Cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = clustering_data.groupby('Cluster').agg({
            'Rating': ['mean', 'std'],
            'Reviews': ['mean', 'sum'],
            'Installs_Numeric': ['mean', 'sum'],
            'Size_MB': ['mean', 'std'],
            'Price_Numeric': ['mean', 'sum']
        }).round(2)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        results = {
            'cluster_analysis': cluster_analysis,
            'cluster_labels': clusters,
            'pca_data': pca_data,
            'feature_importance': pca.explained_variance_ratio_
        }
        
        print("Clustering analysis completed!")
        return results
    
    def generate_insights_report(self, df, reviews_df):
        """
        Generate comprehensive insights report
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        reviews_df : pd.DataFrame
            Reviews dataset
            
        Returns:
        --------
        dict
            Comprehensive insights report
        """
        print("Generating Insights Report...")
        
        insights = {}
        
        # Key metrics
        insights['key_metrics'] = {
            'total_apps': len(df),
            'total_reviews': len(reviews_df),
            'avg_rating': df['Rating'].mean(),
            'free_apps_percentage': (df['Type'] == 'Free').mean() * 100,
            'total_installations': df['Installs_Numeric'].sum(),
            'total_categories': df['Category'].nunique()
        }
        
        # Top performers
        insights['top_performers'] = {
            'highest_rated_apps': df.nlargest(10, 'Rating')[['App', 'Category', 'Rating']],
            'most_installed_apps': df.nlargest(10, 'Installs_Numeric')[['App', 'Category', 'Installs_Numeric']],
            'most_reviewed_apps': df.nlargest(10, 'Reviews')[['App', 'Category', 'Reviews']]
        }
        
        # Category insights
        insights['category_insights'] = {
            'most_popular_category': df['Category'].value_counts().index[0],
            'highest_rated_category': df.groupby('Category')['Rating'].mean().idxmax(),
            'most_installed_category': df.groupby('Category')['Installs_Numeric'].sum().idxmax()
        }
        
        # Pricing insights
        insights['pricing_insights'] = {
            'free_apps_count': (df['Type'] == 'Free').sum(),
            'paid_apps_count': (df['Type'] == 'Paid').sum(),
            'avg_paid_app_price': df[df['Price_Numeric'] > 0]['Price_Numeric'].mean(),
            'free_vs_paid_rating_diff': df[df['Type'] == 'Free']['Rating'].mean() - df[df['Type'] == 'Paid']['Rating'].mean()
        }
        
        # Sentiment insights
        if len(reviews_df) > 0:
            insights['sentiment_insights'] = {
                'positive_sentiment_pct': (reviews_df['Sentiment'] == 'Positive').mean() * 100,
                'negative_sentiment_pct': (reviews_df['Sentiment'] == 'Negative').mean() * 100,
                'avg_sentiment_polarity': reviews_df['Sentiment_Polarity'].mean(),
                'avg_sentiment_subjectivity': reviews_df['Sentiment_Subjectivity'].mean()
            }
        
        # Market trends
        insights['market_trends'] = {
            'avg_app_size': df['Size_MB'].mean(),
            'avg_android_version': df['Android_Min_Version'].mean(),
            'recent_updates_pct': (df['Last_Updated_Date'] >= '2018-01-01').mean() * 100
        }
        
        print("Insights report generated!")
        return insights
    
    def print_analysis_summary(self, analysis_results):
        """
        Print a summary of analysis results
        
        Parameters:
        -----------
        analysis_results : dict
            Results from various analyses
        """
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if 'key_metrics' in analysis_results:
            metrics = analysis_results['key_metrics']
            print(f"\nKey Metrics:")
            print(f"  • Total Apps: {metrics['total_apps']:,}")
            print(f"  • Total Reviews: {metrics['total_reviews']:,}")
            print(f"  • Average Rating: {metrics['avg_rating']:.2f}")
            print(f"  • Free Apps: {metrics['free_apps_percentage']:.1f}%")
            print(f"  • Total Installations: {metrics['total_installations']:,.0f}")
            print(f"  • Categories: {metrics['total_categories']}")
        
        if 'category_insights' in analysis_results:
            cat_insights = analysis_results['category_insights']
            print(f"\nCategory Insights:")
            print(f"  • Most Popular: {cat_insights['most_popular_category']}")
            print(f"  • Highest Rated: {cat_insights['highest_rated_category']}")
            print(f"  • Most Installed: {cat_insights['most_installed_category']}")
        
        if 'pricing_insights' in analysis_results:
            price_insights = analysis_results['pricing_insights']
            print(f"\nPricing Insights:")
            print(f"  • Free Apps: {price_insights['free_apps_count']:,}")
            print(f"  • Paid Apps: {price_insights['paid_apps_count']:,}")
            print(f"  • Average Paid Price: ${price_insights['avg_paid_app_price']:.2f}")
            print(f"  • Rating Difference (Free - Paid): {price_insights['free_vs_paid_rating_diff']:.2f}")
        
        if 'sentiment_insights' in analysis_results:
            sent_insights = analysis_results['sentiment_insights']
            print(f"\nSentiment Insights:")
            print(f"  • Positive Reviews: {sent_insights['positive_sentiment_pct']:.1f}%")
            print(f"  • Negative Reviews: {sent_insights['negative_sentiment_pct']:.1f}%")
            print(f"  • Average Polarity: {sent_insights['avg_sentiment_polarity']:.3f}")
            print(f"  • Average Subjectivity: {sent_insights['avg_sentiment_subjectivity']:.3f}")
        
        print("\n" + "="*60)

# Example usage
if __name__ == "__main__":
    analyzer = Analyzer()
    print("Analyzer module ready for use") 