"""
Visualizer Module for Google Play Store Apps EDA
Handles all plotting and visualization tasks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class Visualizer:
    """Class to create visualizations for Google Play Store Apps EDA"""
    
    def __init__(self, save_path="reports/figures"):
        """
        Initialize Visualizer
        
        Parameters:
        -----------
        save_path : str
            Path to save generated figures
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def plot_rating_distribution(self, df, save=True):
        """
        Plot rating distribution
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(df['Rating'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of App Ratings')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df['Rating'].dropna(), patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        ax2.set_ylabel('Rating')
        ax2.set_title('Box Plot of App Ratings')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_path / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_category_analysis(self, df, top_n=15, save=True):
        """
        Plot category analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        top_n : int
            Number of top categories to show
        save : bool
            Whether to save the plot
        """
        # Get top categories by count
        top_categories = df['Category'].value_counts().head(top_n)
        
        # Get average rating by category for top categories
        category_ratings = df[df['Category'].isin(top_categories.index)].groupby('Category')['Rating'].mean().sort_values(ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Category count plot
        bars1 = ax1.barh(range(len(top_categories)), top_categories.values, color='lightcoral')
        ax1.set_yticks(range(len(top_categories)))
        ax1.set_yticklabels(top_categories.index)
        ax1.set_xlabel('Number of Apps')
        ax1.set_title(f'Top {top_n} Categories by Number of Apps')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        # Category rating plot
        bars2 = ax2.barh(range(len(category_ratings)), category_ratings.values, color='lightblue')
        ax2.set_yticks(range(len(category_ratings)))
        ax2.set_yticklabels(category_ratings.index)
        ax2.set_xlabel('Average Rating')
        ax2.set_title('Average Rating by Category')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_path / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_price_analysis(self, df, save=True):
        """
        Plot price analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price distribution (excluding free apps)
        paid_apps = df[df['Price_Numeric'] > 0]
        
        # Price distribution
        ax1.hist(paid_apps['Price_Numeric'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Paid App Prices')
        ax1.grid(True, alpha=0.3)
        
        # Free vs Paid apps
        app_types = df['Type'].value_counts()
        ax2.pie(app_types.values, labels=app_types.index, autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Free vs Paid Apps')
        
        # Price vs Rating scatter
        ax3.scatter(paid_apps['Price_Numeric'], paid_apps['Rating'], alpha=0.6, color='purple')
        ax3.set_xlabel('Price ($)')
        ax3.set_ylabel('Rating')
        ax3.set_title('Price vs Rating (Paid Apps)')
        ax3.grid(True, alpha=0.3)
        
        # Average rating by type
        type_ratings = df.groupby('Type')['Rating'].mean()
        bars = ax4.bar(type_ratings.index, type_ratings.values, color=['lightgreen', 'lightcoral'])
        ax4.set_ylabel('Average Rating')
        ax4.set_title('Average Rating by App Type')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_path / 'price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_install_analysis(self, df, save=True):
        """
        Plot installation analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        # Create installation categories
        df['Install_Category'] = pd.cut(df['Installs_Numeric'], 
                                      bins=[0, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
                                      labels=['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M+'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Installation distribution
        install_counts = df['Install_Category'].value_counts()
        bars1 = ax1.bar(install_counts.index, install_counts.values, color='lightblue')
        ax1.set_xlabel('Installation Range')
        ax1.set_ylabel('Number of Apps')
        ax1.set_title('Distribution of App Installations')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Installations vs Rating
        install_rating = df.groupby('Install_Category')['Rating'].mean()
        bars2 = ax2.bar(install_rating.index, install_rating.values, color='lightgreen')
        ax2.set_xlabel('Installation Range')
        ax2.set_ylabel('Average Rating')
        ax2.set_title('Average Rating by Installation Range')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Size vs Installations
        ax3.scatter(df['Size_MB'], df['Installs_Numeric'], alpha=0.5, color='orange')
        ax3.set_xlabel('Size (MB)')
        ax3.set_ylabel('Installations')
        ax3.set_title('App Size vs Installations')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Top 10 apps by installations
        top_apps = df.nlargest(10, 'Installs_Numeric')[['App', 'Installs_Numeric']]
        bars3 = ax4.barh(range(len(top_apps)), top_apps['Installs_Numeric'], color='lightcoral')
        ax4.set_yticks(range(len(top_apps)))
        ax4.set_yticklabels(top_apps['App'], fontsize=8)
        ax4.set_xlabel('Installations')
        ax4.set_title('Top 10 Apps by Installations')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_path / 'install_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_sentiment_analysis(self, reviews_df, apps_df, save=True):
        """
        Plot sentiment analysis
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            Reviews dataset
        apps_df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall sentiment distribution
        sentiment_counts = reviews_df['Sentiment'].value_counts()
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Overall Sentiment Distribution')
        
        # Sentiment polarity distribution
        ax2.hist(reviews_df['Sentiment_Polarity'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Sentiment Polarity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Sentiment Polarity')
        ax2.grid(True, alpha=0.3)
        
        # Sentiment subjectivity distribution
        ax3.hist(reviews_df['Sentiment_Subjectivity'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Sentiment Subjectivity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Sentiment Subjectivity')
        ax3.grid(True, alpha=0.3)
        
        # Merge reviews with apps data for category analysis
        merged_data = reviews_df.merge(apps_df[['App', 'Category']], on='App', how='inner')
        
        # Top categories by sentiment
        category_sentiment = merged_data.groupby('Category')['Sentiment_Polarity'].mean().sort_values(ascending=False).head(10)
        bars = ax4.barh(range(len(category_sentiment)), category_sentiment.values, color='lightgreen')
        ax4.set_yticks(range(len(category_sentiment)))
        ax4.set_yticklabels(category_sentiment.index, fontsize=8)
        ax4.set_xlabel('Average Sentiment Polarity')
        ax4.set_title('Top 10 Categories by Average Sentiment')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_path / 'sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_matrix(self, df, save=True):
        """
        Plot correlation matrix
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        # Select numeric columns
        numeric_cols = ['Rating', 'Reviews', 'Size_MB', 'Installs_Numeric', 'Price_Numeric', 'Android_Min_Version']
        numeric_df = df[numeric_cols].dropna()
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numeric Features')
        
        if save:
            plt.savefig(self.save_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_interactive_rating_trends(self, df, save=True):
        """
        Create interactive rating trends plot using Plotly
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        save : bool
            Whether to save the plot
        """
        # Prepare data
        df['Last_Updated_Year'] = df['Last_Updated_Date'].dt.year
        yearly_ratings = df.groupby('Last_Updated_Year')['Rating'].agg(['mean', 'count']).reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Rating by Year', 'Number of Apps by Year', 
                          'Rating Distribution by Type', 'Top Categories by Rating'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average rating by year
        fig.add_trace(
            go.Scatter(x=yearly_ratings['Last_Updated_Year'], y=yearly_ratings['mean'],
                      mode='lines+markers', name='Average Rating'),
            row=1, col=1
        )
        
        # Number of apps by year
        fig.add_trace(
            go.Bar(x=yearly_ratings['Last_Updated_Year'], y=yearly_ratings['count'],
                  name='Number of Apps'),
            row=1, col=2
        )
        
        # Rating distribution by type
        type_ratings = df.groupby('Type')['Rating'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=type_ratings['Type'], y=type_ratings['Rating'],
                  name='Average Rating by Type'),
            row=2, col=1
        )
        
        # Top categories by rating
        top_categories = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=top_categories.values, y=top_categories.index, orientation='h',
                  name='Top Categories'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Rating Trends Analysis",
            showlegend=False,
            height=800
        )
        
        if save:
            fig.write_html(self.save_path / 'interactive_rating_trends.html')
        
    def create_dashboard_summary(self, df, reviews_df, save=True):
        """
        Create a comprehensive dashboard summary
        
        Parameters:
        -----------
        df : pd.DataFrame
            Apps dataset
        reviews_df : pd.DataFrame
            Reviews dataset
        save : bool
            Whether to save the plot
        """
        # Create summary statistics
        total_apps = len(df)
        total_reviews = len(reviews_df)
        avg_rating = df['Rating'].mean()
        free_apps_pct = (df['Type'] == 'Free').mean() * 100
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Add summary boxes
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=total_apps,
            title={"text": "Total Apps"},
            domain={'x': [0, 0.25], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=total_reviews,
            title={"text": "Total Reviews"},
            domain={'x': [0.25, 0.5], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=avg_rating,
            title={"text": "Average Rating"},
            domain={'x': [0.5, 0.75], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=free_apps_pct,
            title={"text": "Free Apps (%)"},
            domain={'x': [0.75, 1], 'y': [0.5, 1]}
        ))
        
        fig.update_layout(
            title="Google Play Store Apps - Dashboard Summary",
            height=400
        )
        
        if save:
            fig.write_html(self.save_path / 'dashboard_summary.html')
        
        # Removed fig.show() to avoid Unicode errors in Windows terminal

# Example usage
if __name__ == "__main__":
    visualizer = Visualizer()
    print("Visualizer module ready for use") 