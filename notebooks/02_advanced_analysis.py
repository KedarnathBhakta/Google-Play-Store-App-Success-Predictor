"""
Google Play Store Apps: Advanced Analysis
This script covers:
1. Sentiment Analysis of User Reviews (Advanced)
2. Time Series Analysis of App Updates and Ratings
3. Predictive Modeling: Predicting App Ratings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Sentiment Analysis of User Reviews (Advanced)
print("\n=== 1. Sentiment Analysis of User Reviews ===\n")
reviews = pd.read_csv('../data/cleaned/googleplaystore_user_reviews_cleaned.csv')
apps = pd.read_csv('../data/cleaned/googleplaystore_cleaned.csv')

# Sentiment distribution
sentiment_counts = reviews['Sentiment'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution in User Reviews')
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.tight_layout()
plt.show()

# Sentiment by app category (merge with apps)
merged = reviews.merge(apps[['App', 'Category']], on='App', how='left')
cat_sent = merged.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
cat_sent = cat_sent[cat_sent.sum(axis=1) > 100]  # Only show categories with >100 reviews
cat_sent.plot(kind='bar', stacked=True, figsize=(12,6), colormap='viridis')
plt.title('Sentiment by App Category')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()

# Word clouds for positive and negative reviews
positive_reviews = ' '.join(reviews[reviews['Sentiment']=='Positive']['Translated_Review'].dropna().astype(str))
negative_reviews = ' '.join(reviews[reviews['Sentiment']=='Negative']['Translated_Review'].dropna().astype(str))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(WordCloud(width=400, height=300, background_color='white').generate(positive_reviews), interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')
plt.subplot(1,2,2)
plt.imshow(WordCloud(width=400, height=300, background_color='white').generate(negative_reviews), interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.tight_layout()
plt.show()

# 2. Time Series Analysis of App Updates and Ratings
print("\n=== 2. Time Series Analysis of App Updates and Ratings ===\n")
apps['Last_Updated_Date'] = pd.to_datetime(apps['Last_Updated_Date'])
# Average rating by update date
date_rating = apps.groupby('Last_Updated_Date')['Rating'].mean().dropna()
plt.figure(figsize=(12,5))
date_rating.rolling(30).mean().plot()
plt.title('Average App Rating Over Time (30-day rolling mean)')
plt.ylabel('Average Rating')
plt.xlabel('Last Updated Date')
plt.tight_layout()
plt.show()
# Impact of update frequency on rating
update_counts = apps['App'].value_counts()
apps['Update_Count'] = apps['App'].map(update_counts)
plt.figure(figsize=(8,5))
sns.scatterplot(x='Update_Count', y='Rating', data=apps, alpha=0.3)
plt.title('App Update Frequency vs. Rating')
plt.xlabel('Number of Updates')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

# 3. Predictive Modeling: Predicting App Ratings
print("\n=== 3. Predictive Modeling: Predicting App Ratings ===\n")
# Prepare features
model_data = apps[['Rating', 'Category', 'Price_Numeric', 'Installs_Numeric', 'Size_MB']]
model_data = model_data.dropna()
# Encode category
model_data = pd.get_dummies(model_data, columns=['Category'], drop_first=True)
X = model_data.drop('Rating', axis=1)
y = model_data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.3f}')
print(f'R^2: {r2:.3f}')
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs. Predicted App Ratings')
plt.plot([1,5],[1,5],'--',color='red')
plt.tight_layout()
plt.show() 