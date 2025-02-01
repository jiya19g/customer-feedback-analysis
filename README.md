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

### Setting Up a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

1. **Create a Virtual Environment**:
   - Navigate to your project folder in the terminal.
   - Run the following command to create a virtual environment (replace `env` with your preferred name for the environment):

     ```bash
     python -m venv env
     ```

2. **Activate the Virtual Environment**:
   - On **Windows**, run:

     ```bash
     .\env\Scripts\activate
     ```

   - On **macOS/Linux**, run:

     ```bash
     source env/bin/activate
     ```

3. **Install the Required Packages**:
   Once the virtual environment is activated, you can install the necessary dependencies by running:

   ```bash
   pip install pandas nltk matplotlib wordcloud streamlit scikit-learn numpy

### Download the Required NLTK Datasets
The application requires some NLTK datasets (VADER lexicon and stopwords). These are downloaded automatically when you run the application, but make sure you have an internet connection.

To manually download the required datasets, run the following Python code:

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```
## Usage

### Running the Dashboard
To run the Streamlit dashboard, follow these steps:

1. Save the script as `app.py`.
2. Open a terminal and navigate to the directory where the script is saved.
3. Run the following command:

   ```bash
   streamlit run app.py
   ```
