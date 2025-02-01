import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Setup
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the sentiment analysis function
def get_sentiment(review):
    sentiment_score = sia.polarity_scores(review)
    if sentiment_score['compound'] > 0.1:
        return 'positive'
    elif sentiment_score['compound'] < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Create a title for the dashboard
st.title("Customer Feedback Analysis Dashboard")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Show data preview
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Data Cleaning
    df_cleaned = df.dropna(subset=['Review Text', 'Department Name'])
    df_cleaned['Sentiment'] = df_cleaned['Review Text'].apply(get_sentiment)
    
    departments = df_cleaned['Department Name'].unique()
    
    # Sentiment Analysis Section
    st.subheader("Sentiment Analysis")
    sentiment_dist = df_cleaned['Sentiment'].value_counts()
    st.write(sentiment_dist)
    
    # Word Cloud Section
    st.subheader("Word Cloud by Department")
    selected_dept = st.selectbox("Select Department", departments)
    
    dept_reviews = df_cleaned[df_cleaned['Department Name'] == selected_dept]['Review Text']
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(dept_reviews))
    
    st.image(wordcloud.to_array(), use_column_width=True)
    
    # Sentiment Distribution Pie Chart
    sentiment_counts = df_cleaned[df_cleaned['Department Name'] == selected_dept]['Sentiment'].value_counts()
    labels = sentiment_counts.index
    sizes = sentiment_counts.values
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title(f"Sentiment Distribution for {selected_dept} Department")
    
    st.pyplot(fig)
    
    # Topic Extraction (Optional)
    st.subheader("Topic Modeling (KMeans Clustering)")
    num_topics = st.slider("Select Number of Topics", 1, 5, 3)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(dept_reviews)
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(X)
    
    terms = vectorizer.get_feature_names_out()
    for topic_num, cluster_center in enumerate(kmeans.cluster_centers_):
        topic_words = [terms[i] for i in cluster_center.argsort()[:-11:-1]]
        st.write(f"Topic #{topic_num + 1}: {', '.join(topic_words)}")
    
    # Show scatter plot for sentiment vs. review length
    df_cleaned['Review Length'] = df_cleaned['Review Text'].apply(len)
    sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
    df_cleaned['Sentiment Score'] = df_cleaned['Sentiment'].map(sentiment_mapping)
    
    fig, ax = plt.subplots()
    ax.scatter(df_cleaned['Review Length'], df_cleaned['Sentiment Score'], alpha=0.5)
    ax.set_title('Sentiment Distribution by Review Length')
    ax.set_xlabel('Review Length (Characters)')
    ax.set_ylabel('Sentiment Score')
    
    st.pyplot(fig)
    
    # Radar Chart (Sentiment Distribution per Department)
    sentiment_dist = df_cleaned.groupby(['Department Name', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_dist = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0)
    
    for dept in departments:
        values = sentiment_dist.loc[dept, ['positive', 'negative', 'neutral']].values
        categories = ['positive', 'negative', 'neutral']
        num_vars = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_title(f"Sentiment Distribution for {dept} Department", size=15, color='blue', y=1.1)

        st.pyplot(fig)
    
    # **Dynamic Sentiment Analysis** - Filter by Sentiment
    st.subheader("Filter Reviews by Sentiment")
    sentiment_filter = st.selectbox("Select Sentiment", ['All', 'Positive', 'Negative', 'Neutral'])
    
    if sentiment_filter != 'All':
        filtered_df = df_cleaned[df_cleaned['Sentiment'] == sentiment_filter.lower()]
    else:
        filtered_df = df_cleaned
    
    st.write(f"Reviews with {sentiment_filter} sentiment:")
    st.dataframe(filtered_df[['Review Text', 'Sentiment']])
    
    # **Review Length Distribution** - Histogram of Review Lengths
    st.subheader("Review Length Distribution")
    
    fig, ax = plt.subplots()
    ax.hist(df_cleaned['Review Length'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Review Lengths')
    ax.set_xlabel('Review Length (Characters)')
    ax.set_ylabel('Frequency')
    
    st.pyplot(fig)
