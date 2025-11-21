
"""
predict.py
Load a trained model and predict the next month's daily closing prices.
This script demonstrates a naive rolling-prediction approach using the last available rows.
"""
import argparse
import pandas as pd
import joblib
import numpy as np
from datetime import timedelta

from features import create_lag_features, rolling_features, merge_sentiment

def predict(args):
    model = joblib.load(args.model)
    history = pd.read_csv(args.history, parse_dates=["Date"])
    # if sentiment provided, load processed sentiments
    if args.sentiment:
        ds = pd.read_csv(args.sentiment, parse_dates=["Date"])
        history = merge_sentiment(history, ds)
    else:
        history["sentiment"] = 0.0
    # We'll build features from history and then do rolling predictions for next N days
    last_date = history["Date"].max()
    N = args.days
    preds = []
    hist = history.copy().sort_values("Date").reset_index(drop=True)
    for i in range(N):
        # create feature row from hist
        temp = create_lag_features(hist, cols=["Close"], lags=5)
        temp = rolling_features(temp, col="Close", windows=[3,7,14])
        row = temp.dropna().iloc[-1]
        X = row.drop(labels=["Date","Open","High","Low","Close","Adj Close","Volume"], errors="ignore").to_frame().T
        y_pred = model.predict(X)[0]
        next_date = last_date + timedelta(days=1)
        preds.append({"Date": next_date.strftime("%Y-%m-%d"), "PredictedClose": float(y_pred)})
        # append predicted row into hist with Close = y_pred to allow next-step lags
        new_row = {"Date": next_date, "Open": y_pred, "High": y_pred, "Low": y_pred, "Close": y_pred, "Adj Close": y_pred, "Volume": 0, "sentiment": 0.0}
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date
    out_df = pd.DataFrame(preds)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} predictions to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", default="data/sample_stock.csv")
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()
    predict(args)
