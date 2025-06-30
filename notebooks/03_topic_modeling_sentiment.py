"""
Google Play Store Apps: Topic Modeling & Aspect-Based Sentiment Analysis

- Extracts main topics from user reviews using LDA (scikit-learn, CPU-friendly)
- Identifies key aspects (e.g., 'ads', 'UI', 'performance') and their sentiment
- Visualizes top words per topic and aspect sentiment
- (Optional) Can be extended to GPU with cuML or RAPIDS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Load cleaned reviews data
reviews = pd.read_csv('../data/cleaned/googleplaystore_user_reviews_cleaned.csv')

# Preprocess text (simple, can be extended)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

reviews['clean_review'] = reviews['Translated_Review'].fillna('').apply(preprocess)

# Topic Modeling with LDA
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

n_topics = 5
n_top_words = 10
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(reviews['clean_review'])
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
lda.fit(X)

print("\n=== Top Words per Topic (LDA) ===\n")
print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words)

# Visualize topics with word clouds
for topic_idx, topic in enumerate(lda.components_):
    plt.figure()
    word_freq = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]}
    wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic #{topic_idx+1} Word Cloud')
    plt.show()

# Aspect-Based Sentiment (simple keyword-based)
aspects = ['ads', 'ui', 'interface', 'performance', 'update', 'bug', 'crash', 'design', 'feature', 'price']
def extract_aspects(text):
    found = [aspect for aspect in aspects if aspect in text]
    return found if found else None

reviews['aspects'] = reviews['clean_review'].apply(extract_aspects)

# Explode aspects for analysis
aspect_df = reviews.explode('aspects').dropna(subset=['aspects'])

# Sentiment by aspect
aspect_sentiment = aspect_df.groupby(['aspects', 'Sentiment']).size().unstack(fill_value=0)
print("\n=== Aspect-Based Sentiment Counts ===\n")
print(aspect_sentiment)

# Visualize aspect sentiment
aspect_sentiment.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title('Sentiment by Aspect (Keyword-based)')
plt.ylabel('Number of Reviews')
plt.xlabel('Aspect')
plt.tight_layout()
plt.show() 