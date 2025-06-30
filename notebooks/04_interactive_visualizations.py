"""
Google Play Store Apps: Advanced Interactive Visualizations
- Interactive category breakdowns (ratings, installs, sentiment)
- Time series (ratings over time, update frequency)
- Correlation heatmap
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
apps = pd.read_csv('../data/cleaned/googleplaystore_cleaned.csv')
reviews = pd.read_csv('../data/cleaned/googleplaystore_user_reviews_cleaned.csv')

# Category breakdown: Average rating by category
cat_rating = apps.groupby('Category')['Rating'].mean().sort_values(ascending=False)
fig = px.bar(cat_rating, x=cat_rating.index, y=cat_rating.values, title='Average Rating by Category', labels={'x':'Category', 'y':'Average Rating'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Category breakdown: Total installs by category
cat_installs = apps.groupby('Category')['Installs_Numeric'].sum().sort_values(ascending=False)
fig = px.bar(cat_installs, x=cat_installs.index, y=cat_installs.values, title='Total Installs by Category', labels={'x':'Category', 'y':'Total Installs'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Sentiment by category (merge reviews and apps)
merged = reviews.merge(apps[['App', 'Category']], on='App', how='left')
sent_cat = merged.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
sent_cat = sent_cat.loc[sent_cat.sum(axis=1) > 100]  # Only categories with >100 reviews
fig = px.bar(sent_cat, barmode='stack', title='Sentiment by Category')
fig.show()

# Time series: Average rating over time
apps['Last_Updated_Date'] = pd.to_datetime(apps['Last_Updated_Date'])
date_rating = apps.groupby('Last_Updated_Date')['Rating'].mean().dropna()
fig = px.line(date_rating.rolling(30).mean(), title='Average App Rating Over Time (30-day rolling mean)', labels={'value':'Average Rating', 'Last_Updated_Date':'Date'})
fig.show()

# Update frequency vs. rating
update_counts = apps['App'].value_counts()
apps['Update_Count'] = apps['App'].map(update_counts)
fig = px.scatter(apps, x='Update_Count', y='Rating', title='App Update Frequency vs. Rating', labels={'Update_Count':'Number of Updates', 'Rating':'Rating'}, opacity=0.3)
fig.show()

# Correlation heatmap (matplotlib/seaborn for static, Plotly for interactive)
numeric_cols = ['Rating', 'Reviews', 'Installs_Numeric', 'Size_MB', 'Price_Numeric']
corr = apps[numeric_cols].corr()
fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap (Apps Data)')
fig.show() 