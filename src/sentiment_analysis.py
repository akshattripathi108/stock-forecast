
"""
sentiment_analysis.py
Compute sentiment score for raw texts using TextBlob or VADER as fallback.
Input: CSV with columns [Date, username, content]
Output: CSV with added 'sentiment' column (float between -1 and 1)
"""
import argparse
import pandas as pd
import numpy as np

def textblob_sentiment(text):
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def vader_sentiment(text):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except Exception:
        return 0.0

def compute_sentiments(df):
    sentiments = []
    # prefer VADER if available
    use_vader = False
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        use_vader = True
    except Exception:
        use_vader = False
    for txt in df["content"].astype(str).fillna(""):
        if use_vader:
            sentiments.append(vader_sentiment(txt))
        else:
            sentiments.append(textblob_sentiment(txt))
    df["sentiment"] = sentiments
    return df

def process(input_csv, output_csv):
    df = pd.read_csv(input_csv, parse_dates=["Date"])
    df = compute_sentiments(df)
    df.to_csv(output_csv, index=False)
    print(f"Wrote processed tweets with sentiment to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/sample_tweets.csv")
    parser.add_argument("--out", dest="out", default="data/sample_tweets_processed.csv")
    args = parser.parse_args()
    process(args.inp, args.out)
