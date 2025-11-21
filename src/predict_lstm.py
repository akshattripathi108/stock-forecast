
"""
predict_lstm.py
Load LSTM model and scaler, and perform rolling predictions for the next N days.
Saves predictions to predictions_lstm.csv
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

def load_meta():
    import json
    with open("models/lstm_meta.json", "r") as f:
        return json.load(f)

def predict(args):
    model = load_model(args.model)
    scaler = joblib.load(args.scaler)
    meta = load_meta()
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    history = pd.read_csv(args.history, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if args.sentiment:
        ds = pd.read_csv(args.sentiment, parse_dates=["Date"])
        history = pd.merge(history, ds[["Date","sentiment"]], on="Date", how="left")
        history["sentiment"].fillna(0.0, inplace=True)
    else:
        history["sentiment"] = 0.0
    # build features same as training
    history["close_pct_change"] = history["Close"].pct_change().fillna(0.0)
    history["sma_5"] = history["Close"].rolling(5).mean().fillna(method="bfill")
    feature_df = history[["Close","close_pct_change","sma_5","sentiment"]].fillna(0.0)
    scaled = scaler.transform(feature_df)
    numeric = scaled
    preds = []
    last_date = history["Date"].max()
    seq = numeric[-seq_len:].copy()
    for i in range(args.days):
        X = np.expand_dims(seq, axis=0)  # shape (1, seq_len, features)
        y_pred = model.predict(X)[0][0]
        next_date = last_date + timedelta(days=1)
        preds.append({"Date": next_date.strftime("%Y-%m-%d"), "PredictedClose": float(y_pred)})
        # append predicted row into seq: we need to create a new feature row
        # approximate pct_change and sma_5 using previous values
        prev_close = seq[-1,0]
        new_close = y_pred
        new_pct = (new_close - prev_close) / (abs(prev_close) + 1e-9)
        sma5 = float(np.concatenate([seq[:,0], [new_close]])[-5:].mean())
        sentiment = 0.0
        new_row = np.array([new_close, new_pct, sma5, sentiment])
        seq = np.vstack([seq[1:], new_row])
        last_date = next_date
    out_df = pd.DataFrame(preds)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} LSTM predictions to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lstm_model.h5")
    parser.add_argument("--scaler", default="models/scaler.pkl")
    parser.add_argument("--history", default="data/sample_stock.csv")
    parser.add_argument("--sentiment", default="data/sample_tweets_processed.csv")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", default="predictions_lstm.csv")
    args = parser.parse_args()
    predict(args)
