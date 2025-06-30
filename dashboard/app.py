#!/usr/bin/env python3
"""
Interactive Dashboard for Google Play Store Apps Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys
import os
import fpdf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import AppSuccessPredictor
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner

# Page configuration
st.set_page_config(
    page_title="Google Play Store App Success Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        apps_df = pd.read_csv('data/apps_cleaned.csv')
        reviews_df = pd.read_csv('data/reviews_cleaned.csv')
        return apps_df, reviews_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the analysis pipeline first.")
        return None, None

@st.cache_resource
def load_models():
    """Load and cache the trained models"""
    try:
        models = {}
        for name in ["popularity", "high_rating"]:
            path = os.path.join("..", "models", f"{name}_model.joblib")
            if not os.path.exists(path):
                path = os.path.join("models", f"{name}_model.joblib")
            models[name] = joblib.load(path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📱 Google Play Store App Success Predictor</h1>', unsafe_allow_html=True)
    
    # Load data
    apps_df, reviews_df = load_data()
    if apps_df is None:
        st.stop()
    
    # Load models
    models = load_models()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📊 Data Exploration", "📈 Visualizations", "🤖 Predictions", "📋 Reports"]
    )
    
    if page == "🏠 Overview":
        show_overview(apps_df, reviews_df)
    elif page == "📊 Data Exploration":
        show_data_exploration(apps_df, reviews_df)
    elif page == "📈 Visualizations":
        show_visualizations(apps_df, reviews_df)
    elif page == "🤖 Predictions":
        show_predictions(models)
    elif page == "📋 Reports":
        show_reports()

def show_overview(apps_df, reviews_df):
    """Show overview page"""
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Apps", f"{len(apps_df):,}")
    
    with col2:
        st.metric("Total Reviews", f"{len(reviews_df):,}")
    
    with col3:
        avg_rating = apps_df['Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    with col4:
        free_pct = (apps_df['Type'] == 'Free').mean() * 100
        st.metric("Free Apps", f"{free_pct:.1f}%")
    
    # Data summary
    st.subheader("📋 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Apps Dataset:**")
        st.write(f"- Shape: {apps_df.shape}")
        st.write(f"- Categories: {apps_df['Category'].nunique()}")
        st.write(f"- Content Ratings: {apps_df['Content Rating'].nunique()}")
        st.write(f"- Average Price: ${apps_df['Price_Numeric'].mean():.2f}")
    
    with col2:
        st.write("**Reviews Dataset:**")
        st.write(f"- Shape: {reviews_df.shape}")
        st.write(f"- Sentiment Distribution:")
        sentiment_counts = reviews_df['Sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            st.write(f"  - {sentiment}: {count:,} ({count/len(reviews_df)*100:.1f}%)")
    
    # Top categories
    st.subheader("🏆 Top Categories")
    top_categories = apps_df['Category'].value_counts().head(10)
    
    fig = px.bar(
        x=top_categories.values,
        y=top_categories.index,
        orientation='h',
        title="Top 10 Categories by Number of Apps",
        labels={'x': 'Number of Apps', 'y': 'Category'}
    )
    fig.update_layout(height=400)
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

def show_data_exploration(apps_df, reviews_df):
    """Show data exploration page"""
    st.header("📊 Data Exploration")
    
    # Data filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_categories = st.multiselect(
            "Select Categories:",
            options=sorted(apps_df['Category'].unique()),
            default=sorted(apps_df['Category'].unique())[:5]
        )
    
    with col2:
        selected_types = st.multiselect(
            "Select App Types:",
            options=sorted(apps_df['Type'].unique()),
            default=sorted(apps_df['Type'].unique())
        )
    
    with col3:
        rating_range = st.slider(
            "Rating Range:",
            min_value=float(apps_df['Rating'].min()),
            max_value=float(apps_df['Rating'].max()),
            value=(float(apps_df['Rating'].min()), float(apps_df['Rating'].max())),
            step=0.1
        )
    
    # Filter data
    filtered_df = apps_df[
        (apps_df['Category'].isin(selected_categories)) &
        (apps_df['Type'].isin(selected_types)) &
        (apps_df['Rating'] >= rating_range[0]) &
        (apps_df['Rating'] <= rating_range[1])
    ]
    
    st.write(f"**Filtered Results:** {len(filtered_df)} apps")
    
    # Data table
    st.subheader("📋 Data Table")
    
    # Select columns to display
    available_columns = ['App', 'Category', 'Rating', 'Reviews', 'Size_MB', 'Installs_Numeric', 'Type', 'Price_Numeric', 'Content Rating']
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=available_columns,
        default=['App', 'Category', 'Rating', 'Reviews', 'Type', 'Price_Numeric']
    )
    
    if selected_columns:
        st.dataframe(filtered_df[selected_columns].head(100), use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Statistics:**")
        numeric_cols = ['Rating', 'Reviews', 'Size_MB', 'Installs_Numeric', 'Price_Numeric']
        st.dataframe(filtered_df[numeric_cols].describe())
    
    with col2:
        st.write("**Categorical Statistics:**")
        categorical_cols = ['Category', 'Type', 'Content Rating']
        for col in categorical_cols:
            if col in filtered_df.columns:
                st.write(f"**{col}:**")
                st.write(filtered_df[col].value_counts().head())

def show_visualizations(apps_df, reviews_df):
    """Show visualizations page"""
    st.header("📈 Visualizations")
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization:",
        ["Rating Distribution", "Category Analysis", "Price Analysis", "Installation Analysis", "Sentiment Analysis", "Correlation Matrix"]
    )
    
    if viz_option == "Rating Distribution":
        st.subheader("📊 Rating Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                apps_df, x='Rating', nbins=30,
                title="Rating Distribution",
                labels={'Rating': 'Rating', 'count': 'Number of Apps'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                apps_df, x='Type', y='Rating',
                title="Rating by App Type",
                labels={'Type': 'App Type', 'Rating': 'Rating'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Category Analysis":
        st.subheader("📊 Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top categories by average rating
            category_ratings = apps_df.groupby('Category')['Rating'].agg(['mean', 'count']).reset_index()
            category_ratings = category_ratings[category_ratings['count'] >= 10]  # Filter for categories with at least 10 apps
            top_rated = category_ratings.nlargest(10, 'mean')
            
            fig = px.bar(
                top_rated, x='mean', y='Category',
                orientation='h',
                title="Top 10 Categories by Average Rating",
                labels={'mean': 'Average Rating', 'Category': 'Category'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category popularity
            category_counts = apps_df['Category'].value_counts().head(15)
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Top 15 Categories by Number of Apps"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Price Analysis":
        st.subheader("💰 Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(
                apps_df[apps_df['Price_Numeric'] > 0], x='Price_Numeric', nbins=30,
                title="Price Distribution (Paid Apps Only)",
                labels={'Price_Numeric': 'Price ($)'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Rating
            fig = px.scatter(
                apps_df, x='Price_Numeric', y='Rating', color='Type',
                title="Price vs Rating",
                labels={'Price_Numeric': 'Price ($)', 'Rating': 'Rating'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Installation Analysis":
        st.subheader("📥 Installation Analysis")
        
        # Log scale for better visualization
        apps_df['Installs_Log'] = np.log10(apps_df['Installs_Numeric'] + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                apps_df, x='Installs_Log', nbins=30,
                title="Installation Distribution (Log Scale)",
                labels={'Installs_Log': 'Log10(Installations)'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Installations by category
            category_installs = apps_df.groupby('Category')['Installs_Numeric'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=category_installs.values,
                y=category_installs.index,
                orientation='h',
                title="Top 10 Categories by Total Installations",
                labels={'x': 'Total Installations'}
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_line_color='black', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Sentiment Analysis":
        st.subheader("😊 Sentiment Analysis")
        
        if len(reviews_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = reviews_df['Sentiment'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment polarity distribution
                fig = px.histogram(
                    reviews_df, x='Sentiment_Polarity', nbins=30,
                    title="Sentiment Polarity Distribution",
                    labels={'Sentiment_Polarity': 'Sentiment Polarity'}
                )
                fig.update_layout(height=400)
                fig.update_traces(marker_line_color='black', marker_line_width=1.5)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No reviews data available for sentiment analysis.")
    
    elif viz_option == "Correlation Matrix":
        st.subheader("🔗 Correlation Matrix")
        
        # Select numeric columns
        numeric_cols = ['Rating', 'Reviews', 'Size_MB', 'Installs_Numeric', 'Price_Numeric']
        numeric_df = apps_df[numeric_cols].dropna()
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Numeric Features",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_predictions(models):
    """Show predictions page"""
    st.header("🤖 App Success Predictions")
    
    if not models:
        st.error("No trained models found. Please run the predictive modeling pipeline first.")
        return
    
    st.subheader("📱 Predict App Success")
    
    # Load cleaned app data
    @st.cache_data
    def load_app_data():
        path = os.path.join("..", "data", "cleaned", "googleplaystore_cleaned.csv")
        if not os.path.exists(path):
            path = os.path.join("data", "cleaned", "googleplaystore_cleaned.csv")
        df = pd.read_csv(path)
        return df

    df_apps = load_app_data()

    # Load cleaned user reviews data
    @st.cache_data
    def load_reviews_data():
        path = os.path.join("..", "data", "cleaned", "googleplaystore_user_reviews_cleaned.csv")
        if not os.path.exists(path):
            path = os.path.join("data", "cleaned", "googleplaystore_user_reviews_cleaned.csv")
        df = pd.read_csv(path)
        return df

    df_reviews = load_reviews_data()

    # App selection
    app_names = ["Custom"] + sorted(df_apps["App"].unique())
    selected_app = st.sidebar.selectbox("Choose App", app_names)

    # If an app is selected, fill fields with its data, else use manual input
    feature_defaults = {
        'Category': "FAMILY",
        'Primary_Genre': "Entertainment",
        'Size_MB': 20.0,
        'Is_Free': 1,
        'Price_USD': 0.0,
        'Content_Rating': "Everyone",
        'Reviews': 100,
        'Has_Rating': 1,
        'Has_Reviews': 1,
        'Avg_Sentiment': 0.0,
        'Sentiment_Std': 0.1,
        'Review_Count': 100,
        'Avg_Subjectivity': 0.5,
        'Subjectivity_Std': 0.1,
        'Positive_Ratio': 0.7
    }

    if selected_app != "Custom":
        app_row = df_apps[df_apps["App"] == selected_app].iloc[0]
        def get_feature(name):
            if name in app_row and pd.notnull(app_row[name]):
                return app_row[name]
            return feature_defaults[name]
        category = str(get_feature("Category"))
        primary_genre = str(get_feature("Primary_Genre"))
        size_mb = float(get_feature("Size_MB"))
        price_usd = float(get_feature("Price_USD"))
        is_free = 1 if price_usd == 0 else 0
        content_rating = str(get_feature("Content_Rating"))
        reviews = int(get_feature("Reviews"))
        has_rating = int(get_feature("Has_Rating")) if "Has_Rating" in app_row else 1
        has_reviews = int(get_feature("Has_Reviews")) if "Has_Reviews" in app_row else 1

        # Aggregate review features from reviews CSV
        app_reviews = df_reviews[df_reviews["App"] == selected_app]
        if not app_reviews.empty:
            avg_sentiment = app_reviews["Sentiment_Polarity"].mean()
            sentiment_std = app_reviews["Sentiment_Polarity"].std(ddof=0) if app_reviews.shape[0] > 1 else 0.1
            review_count = app_reviews.shape[0]
            avg_subjectivity = app_reviews["Sentiment_Subjectivity"].mean()
            subjectivity_std = app_reviews["Sentiment_Subjectivity"].std(ddof=0) if app_reviews.shape[0] > 1 else 0.1
            positive_ratio = (app_reviews["Sentiment_Category"] == "Positive").mean()
        else:
            avg_sentiment = feature_defaults['Avg_Sentiment']
            sentiment_std = feature_defaults['Sentiment_Std']
            review_count = reviews
            avg_subjectivity = feature_defaults['Avg_Subjectivity']
            subjectivity_std = feature_defaults['Subjectivity_Std']
            positive_ratio = feature_defaults['Positive_Ratio']
    else:
        category = st.sidebar.selectbox("Category", [
            "GAME", "FAMILY", "TOOLS", "MEDICAL", "BUSINESS", "PRODUCTIVITY", "PERSONALIZATION", "COMMUNICATION", "SPORTS", "LIFESTYLE", "PHOTOGRAPHY", "FINANCE", "SOCIAL", "HEALTH_AND_FITNESS", "SHOPPING", "TRAVEL_AND_LOCAL", "NEWS_AND_MAGAZINES", "BOOKS_AND_REFERENCE", "DATING", "VIDEO_PLAYERS", "MAPS_AND_NAVIGATION", "FOOD_AND_DRINK", "EDUCATION", "ENTERTAINMENT", "AUTO_AND_VEHICLES", "WEATHER", "ART_AND_DESIGN", "HOUSE_AND_HOME", "LIBRARIES_AND_DEMO", "COMICS", "PARENTING", "EVENTS", "BEAUTY"])
        primary_genre = st.sidebar.selectbox("Primary Genre", [
            "Entertainment", "Medical", "Maps & Navigation", "Lifestyle", "Puzzle", "Education", "Tools", "Photography", "Finance", "Social", "Health & Fitness", "Shopping", "Travel & Local", "News & Magazines", "Books & Reference", "Dating", "Video Players", "Food & Drink", "Business", "Productivity", "Personalization", "Communication", "Sports", "Family", "Auto & Vehicles", "Weather", "Art & Design", "House & Home", "Libraries & Demo", "Comics", "Parenting", "Events", "Beauty"])
        size_mb = st.sidebar.slider("App Size (MB)", 1, 500, 20)
        price_usd = st.sidebar.number_input("Price (USD)", 0.0, 100.0, 0.0)
        is_free = 1 if price_usd == 0 else 0
        content_rating = st.sidebar.selectbox("Content Rating", ["Everyone", "Teen", "Mature 17+", "Everyone 10+", "Adults only 18+", "Unrated"])
        reviews = st.sidebar.number_input("Number of Reviews", 0, 1000000, 100)
        avg_sentiment = st.sidebar.slider("Average Review Sentiment", -1.0, 1.0, 0.0)
        sentiment_std = st.sidebar.slider("Sentiment Std Dev", 0.0, 1.0, 0.1)
        review_count = reviews
        avg_subjectivity = st.sidebar.slider("Average Subjectivity", 0.0, 1.0, 0.5)
        subjectivity_std = st.sidebar.slider("Subjectivity Std Dev", 0.0, 1.0, 0.1)
        positive_ratio = st.sidebar.slider("Positive Review Ratio", 0.0, 1.0, 0.7)
        has_rating = 1
        has_reviews = 1 if reviews > 0 else 0

    # Prepare input DataFrame
    input_dict = {
        'Category': category,
        'Primary_Genre': primary_genre,
        'Size_MB': size_mb,
        'Is_Free': is_free,
        'Price_USD': price_usd,
        'Content_Rating': content_rating,
        'Reviews': reviews,
        'Has_Rating': has_rating,
        'Has_Reviews': has_reviews,
        'Avg_Sentiment': avg_sentiment,
        'Sentiment_Std': sentiment_std,
        'Review_Count': review_count,
        'Avg_Subjectivity': avg_subjectivity,
        'Subjectivity_Std': subjectivity_std,
        'Positive_Ratio': positive_ratio
    }
    input_df = pd.DataFrame([input_dict])

    st.header("Prediction Results")
    if st.button("Predict App Success"):
        # Popularity prediction
        pop_pred = models["popularity"].predict(input_df)[0]
        # High/Low rating prediction
        high_rating_pred = models["high_rating"].predict(input_df)[0]
        high_rating_prob = models["high_rating"].predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Installs", f"{int(pop_pred):,}")
            if pop_pred >= 1000000:
                st.success("High popularity potential!")
            elif pop_pred >= 100000:
                st.info("Good popularity potential")
            else:
                st.warning("Moderate popularity potential")
        with col2:
            st.metric("High Rating?", "Yes" if high_rating_pred else "No", delta=f"{high_rating_prob*100:.1f}% prob.")
            if high_rating_pred:
                st.success("Likely to be highly rated!")
            else:
                st.warning("May need improvements for high rating")

        # Debug: Show input features used for prediction
        st.subheader(":mag: Debug - Model Input Features")
        st.write(input_df.T.rename(columns={0: 'Value'}))

        # Recommendations
        st.subheader(":bulb: Recommendations")
        recs = []
        if reviews < 100:
            recs.append("Increase user reviews to boost installs and ratings.")
        if avg_sentiment < 0.2:
            recs.append("Focus on improving user sentiment through better UX and support.")
        if size_mb > 100:
            recs.append("Large app size may affect downloads. Consider optimization.")
        if not high_rating_pred:
            recs.append("Encourage positive reviews and address user feedback.")
        if is_free == 0 and price_usd > 5.0:
            recs.append("Consider a more competitive pricing strategy.")
        if not recs:
            recs.append("Your app is well-positioned! Focus on marketing and scaling.")
        for rec in recs:
            st.write(f"- {rec}")

        # Feature importance (for popularity)
        st.subheader("Top Features Impacting Popularity Prediction")
        importances = None
        try:
            importances = models["popularity"].named_steps["classifier"].feature_importances_
            preprocessor = models["popularity"].named_steps["preprocessor"]
            # Get feature names
            num_features = [
                'Size_MB', 'Is_Free', 'Price_USD', 'Reviews', 'Has_Rating', 'Has_Reviews',
                'Avg_Sentiment', 'Sentiment_Std', 'Review_Count', 'Avg_Subjectivity', 'Subjectivity_Std', 'Positive_Ratio'
            ]
            cat_features = ['Category', 'Primary_Genre', 'Content_Rating']
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
            feature_names = num_features + list(cat_feature_names)
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            st.bar_chart(feat_imp_df.set_index('feature'))
        except Exception as e:
            st.info("Feature importance not available for this model.")

# PDF report generation helper
from fpdf import FPDF

def generate_pdf_report(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Google Play Store Apps Model Report", ln=True, align='C')
    pdf.ln(10)
    # Model Performance
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Model Performance:", ln=True)
    pdf.set_font("Arial", size=11)
    for target, perf in report.get('model_performance', {}).items():
        pdf.cell(0, 8, f"{target}", ln=True)
        for k, v in perf.items():
            pdf.cell(0, 8, f"   {k}: {v}", ln=True)
    pdf.ln(5)
    # Feature Importance
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Feature Importance:", ln=True)
    pdf.set_font("Arial", size=11)
    for target, features in report.get('feature_importance', {}).items():
        pdf.cell(0, 8, f"{target}", ln=True)
        for feat in features:
            pdf.cell(0, 8, f"   {feat['feature']}: {feat['importance']:.4f}", ln=True)
    pdf.ln(5)
    # Sample Predictions
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Sample Predictions:", ln=True)
    pdf.set_font("Arial", size=11)
    for target, preds in report.get('predictions_sample', {}).items():
        pdf.cell(0, 8, f"{target}: {preds}", ln=True)
    return pdf.output(dest='S').encode('latin1')

def show_reports():
    """Show reports page"""
    st.header("📋 Reports")
    try:
        with open('results/comprehensive_report.json', 'r') as f:
            report = json.load(f)
        st.subheader("📊 Model Performance Report")
        for target_name, performance in report['model_performance'].items():
            with st.expander(f"Model: {target_name}"):
                if performance['target_type'] == 'regression':
                    st.write(f"**R² Score:** {performance['r2']:.3f}")
                    st.write(f"**MSE:** {performance['mse']:.3f}")
                else:
                    st.write(f"**Accuracy:** {performance['accuracy']:.3f}")
        # Feature importance
        if 'feature_importance' in report:
            st.subheader("🔍 Feature Importance")
            for target_name, features in report['feature_importance'].items():
                with st.expander(f"Feature Importance: {target_name}"):
                    importance_df = pd.DataFrame(features)
                    st.dataframe(importance_df)
    except FileNotFoundError:
        st.warning("Reports not found. Please run the analysis pipeline first.")
    # Download section
    st.subheader("📥 Download Reports")
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists('results/comprehensive_report.json'):
            with open('results/comprehensive_report.json', 'r') as f:
                st.download_button(
                    label="Download Model Report (JSON)",
                    data=f.read(),
                    file_name="model_report.json",
                    mime="application/json"
                )
    with col2:
        if os.path.exists('results/recommendations.json'):
            with open('results/recommendations.json', 'r') as f:
                st.download_button(
                    label="Download Recommendations (JSON)",
                    data=f.read(),
                    file_name="recommendations.json",
                    mime="application/json"
                )
    with col3:
        if os.path.exists('results/comprehensive_report.json'):
            with open('results/comprehensive_report.json', 'r') as f:
                report = json.load(f)
                pdf_bytes = generate_pdf_report(report)
                st.download_button(
                    label="Download Model Report (PDF)",
                    data=pdf_bytes,
                    file_name="model_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main() 