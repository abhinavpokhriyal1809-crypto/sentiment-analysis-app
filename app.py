import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, scores

st.title("😊 Sentiment Analysis App")

text = st.text_area("Enter your text")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, scores = analyze_sentiment(text)

        st.write("Sentiment:", sentiment)
        st.write(scores)