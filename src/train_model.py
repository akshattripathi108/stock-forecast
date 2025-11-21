
"""
train_model.py
Train a simple scikit-learn model on engineered features.
"""
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import create_lag_features, rolling_features, merge_sentiment

def load_data(path_prices, path_sentiment=None):
    df = pd.read_csv(path_prices, parse_dates=["Date"])
    if path_sentiment:
        ds = pd.read_csv(path_sentiment, parse_dates=["Date"])
        df = merge_sentiment(df, ds)
    else:
        df["sentiment"] = 0.0
    df = create_lag_features(df, cols=["Close"], lags=5)
    df = rolling_features(df, col="Close", windows=[3,7,14])
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=["Date","Open","High","Low","Close","Adj Close","Volume"])
    y = df["Close"]
    return X, y

def train(args):
    X, y = load_data(args.data, args.sentiment)
    tscv = TimeSeriesSplit(n_splits=3)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    joblib.dump(model, args.out)
    print(f"Trained and saved model to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_stock.csv")
    parser.add_argument("--sentiment", default="data/sample_tweets_processed.csv")
    parser.add_argument("--out", default="models/model.pkl")
    args = parser.parse_args()
    train(args)
