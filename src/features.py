
"""
features.py
Feature engineering: create lag features, rolling windows, and merge with sentiment.
"""
import pandas as pd
import numpy as np

def create_lag_features(df, cols=["Close"], lags=5):
    df_sorted = df.sort_values("Date").copy()
    for col in cols:
        for lag in range(1, lags+1):
            df_sorted[f"{col}_lag_{lag}"] = df_sorted[col].shift(lag)
    return df_sorted

def rolling_features(df, col="Close", windows=[3,7,14]):
    df_sorted = df.sort_values("Date").copy()
    for w in windows:
        df_sorted[f"{col}_roll_mean_{w}"] = df_sorted[col].rolling(window=w).mean()
        df_sorted[f"{col}_roll_std_{w}"] = df_sorted[col].rolling(window=w).std()
    return df_sorted

def merge_sentiment(df_prices, df_sentiment):
    # df_sentiment: Date, sentiment (numeric)
    df_prices["Date"] = pd.to_datetime(df_prices["Date"])
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
    daily_sent = df_sentiment.groupby(pd.Grouper(key="Date", freq="D")).agg({"sentiment":"mean"}).reset_index()
    merged = pd.merge(df_prices, daily_sent, on="Date", how="left")
    merged["sentiment"].fillna(0.0, inplace=True)
    return merged
