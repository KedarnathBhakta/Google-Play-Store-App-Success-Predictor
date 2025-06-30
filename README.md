📱 Google Play Store Apps - Exploratory Data Analysis


📖 Project Overview
This project performs a comprehensive exploratory data analysis of Google Play Store apps data to uncover insights about app performance, user preferences, and market trends.

---

💼 For Recruiters
- Technical Depth: Modular, production-ready code with unit tests, type hints, and docstrings. Efficient for both CPU and GPU workloads.
- Business Acumen: Actionable insights, market trends, and recommendations for app developers and marketers.
- Professional Presentation: Interactive visualizations, clean code, and a clear narrative from data to business value.
- Scalable & Extensible: Easily add new analyses, dashboards, or machine learning models.

---

🚀 Quickstart

1. Clone the repository:
   git clone <repository-url>
   cd google-play-store-eda

2. Install dependencies:
   pip install -r requirements.txt

3. Run the analysis:
   # Run all notebooks in order
   jupyter notebook notebooks/
   # Or run the main script
   python main_analysis.py

4. Run unit tests:
   pytest tests/

---

⚡ CPU/GPU Compatibility & Efficiency
- All data processing uses vectorized operations for speed.
- For large datasets, chunking and efficient memory use are implemented.
- The codebase is compatible with both CPU and GPU (can be extended with RAPIDS/cuDF for GPU acceleration).
- Designed to avoid CPU overload—suitable for both integrated and dedicated CPUs.

---

📊 Dataset Description
The project uses two main datasets:

1️⃣ Google Play Store Apps Dataset (googleplaystore.csv)
- Size: 10,841 rows × 13 columns
- Features:
  - App: App name
  - Category: App category
  - Rating: User rating (1-5)
  - Reviews: Number of reviews
  - Size: App size
  - Installs: Number of installs
  - Type: Free or Paid
  - Price: App price
  - Content Rating: Age group
  - Genres: App genres
  - Last Updated: Last update date
  - Current Ver: Current version
  - Android Ver: Required Android version

2️⃣ User Reviews Dataset (googleplaystore_user_reviews.csv)
- Size: 64,296 rows × 5 columns
- Features:
  - App: App name
  - Translated_Review: User review text
  - Sentiment: Positive, Negative, or Neutral
  - Sentiment_Polarity: Sentiment score (-1 to 1)
  - Sentiment_Subjectivity: Subjectivity score (0 to 1)

---

PROJECT STRUCTURE

📦 Google Play Store App Success Predictor
├── README.md
├── requirements.txt
├── data
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_analysis_apps.ipynb
│   ├── 04_analysis_reviews.ipynb
│   └── 05_final_insights.ipynb
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── visualizer.py
│   └── analyzer.py
├── reports
│   ├── figures
│   └── final_report.md
├── results
│   └── insights_summary.md

---

🔍 Key Analysis Areas

1️⃣ App Performance Analysis
- Rating distribution and trends
- Review count analysis
- Installation patterns
- Size vs Performance correlation

2️⃣ Category Analysis
- Most popular categories
- Category-wise ratings
- Category-wise installations
- Category performance comparison

3️⃣ Pricing Analysis
- Free vs Paid apps comparison
- Price distribution
- Price vs Rating correlation
- Price vs Installation correlation

4️⃣ User Sentiment Analysis
- Overall sentiment distribution
- Sentiment by category
- Sentiment polarity trends
- Review sentiment vs app performance

5️⃣ Content Rating Analysis
- Age group preferences
- Content rating vs performance
- Safety and appropriateness trends

---

🛠️ Technologies Used
- Python 3.8+
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- Matplotlib: Basic plotting
- Seaborn: Statistical data visualization
- Plotly: Interactive visualizations
- Scikit-learn: Machine learning utilities
- NLTK: Natural language processing (for sentiment analysis)

---

⚙️ Installation and Setup

1. Clone the repository:
   git clone <repository-url>
   cd google-play-store-eda

2. Install dependencies:
   pip install -r requirements.txt

3. Run the analysis:
   # Run all notebooks in order
   jupyter notebook notebooks/

---

🌟 Key Findings

App Performance Insights
- Top Performing Categories: Games, Education, and Productivity apps show highest ratings
- Installation Trends: Free apps dominate the market with 93% of apps being free
- Size Impact: Smaller apps (<50MB) tend to have higher installation rates

User Sentiment Insights
- Overall Sentiment: 64% positive, 24% neutral, 12% negative
- Category Sentiment: Educational and productivity apps receive most positive feedback
- Price Impact: Free apps receive slightly more positive sentiment than paid apps

Market Trends
- Category Growth: Gaming and social media categories show highest growth
- Pricing Strategy: Most successful apps are free with in-app purchases
- User Preferences: Users prefer apps with regular updates and good user interface

---

🤝 Contributing
Feel free to contribute to this project by:
- Adding new analysis methods
- Improving visualizations
- Enhancing data cleaning processes
- Adding new insights

---

📝 License
This project is licensed under the MIT License.

---

📬 Contact
For questions or suggestions, please open an issue in the repository. 
Or contact LinkedIn (https://www.linkedin.com/in/kedarnath-bhakta-394964250/)

