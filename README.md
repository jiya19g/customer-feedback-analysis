# Customer Feedback Analysis Dashboard

## Overview
This is a Customer Feedback Analysis Dashboard built using Python, Streamlit, and various NLP techniques. The dashboard allows users to upload a CSV file containing customer feedback and provides various analytical insights, such as sentiment analysis, word clouds, sentiment distribution by department, and topic modeling. Users can interact with the dashboard to visualize sentiment trends, word clouds, and the distribution of review lengths.

## Features
- **Sentiment Analysis**: Automatically classifies reviews as positive, negative, or neutral using VADER Sentiment Analysis.
- **Word Cloud by Department**: Generates a word cloud of the most frequent terms in reviews for a selected department.
- **Sentiment Distribution**: Displays a pie chart showing the distribution of sentiments (positive, negative, neutral) for reviews in the selected department.
- **Topic Modeling**: Uses KMeans clustering for topic extraction from the reviews, showing the most frequent terms for each topic.
- **Review Length vs Sentiment**: A scatter plot showing sentiment scores against review lengths.
- **Radar Chart**: Visualizes sentiment distribution for each department in a radar chart.
- **Dynamic Sentiment Filtering**: Allows users to filter reviews based on sentiment (Positive, Negative, Neutral).
- **Review Length Distribution**: Displays a histogram of review lengths (characters).

## Installation

### Prerequisites
You need to have the following Python packages installed:
- `pandas`
- `nltk`
- `matplotlib`
- `wordcloud`
- `streamlit`
- `sklearn`
- `numpy`

You can install these packages using pip:

```bash
pip install pandas nltk matplotlib wordcloud streamlit scikit-learn numpy
