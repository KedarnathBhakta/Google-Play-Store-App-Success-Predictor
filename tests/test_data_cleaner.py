import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
from data_cleaner import DataCleaner

def test_clean_apps_data():
    raw = pd.DataFrame({
        'App': ['A', 'A'],
        'Category': ['GAME', 'GAME'],
        'Rating': [4.5, 6.0],
        'Reviews': ['100', 'bad'],
        'Size': ['19M', 'Varies with device'],
        'Installs': ['1,000+', '5,000+'],
        'Type': ['Free', 'Free'],
        'Price': ['0', '$0.99'],
        'Content Rating': ['Everyone', 'Everyone'],
        'Genres': ['Action', 'Action'],
        'Last Updated': ['January 1, 2020', 'February 1, 2020'],
        'Current Ver': ['1.0', '1.0'],
        'Android Ver': ['4.0 and up', 'Varies with device']
    })
    cleaner = DataCleaner()
    cleaned = cleaner.clean_apps_data(raw)
    assert cleaned.shape[0] == 1  # Duplicates removed
    assert cleaned['Rating'].iloc[0] == 4.5
    assert cleaned['Size_MB'].iloc[0] == 19.0
    assert cleaned['Installs_Numeric'].iloc[0] == 1000.0
    assert cleaned['Price_Numeric'].iloc[0] == 0.0
    assert pd.to_datetime(cleaned['Last_Updated_Date'].iloc[0]) == pd.Timestamp('2020-01-01')
    assert cleaned['Android_Min_Version'].iloc[0] == 4.0

def test_clean_reviews_data():
    raw = pd.DataFrame({
        'App': ['A', 'B'],
        'Translated_Review': ['Good', 'Bad'],
        'Sentiment': ['Positive', None],
        'Sentiment_Polarity': [0.5, None],
        'Sentiment_Subjectivity': [0.7, None]
    })
    cleaner = DataCleaner()
    cleaned = cleaner.clean_reviews_data(raw)
    assert cleaned.shape[0] == 1
    assert cleaned['Sentiment'].iloc[0] == 'Positive' 