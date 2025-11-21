
"""
train_lstm.py
Train an LSTM model for next-day closing price prediction using historical prices + daily sentiment.
Saves model to models/lstm_model.h5 and scaler to models/scaler.pkl
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from features import create_lag_features, rolling_features, merge_sentiment

def create_sequences(data, target_col="Close", seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][target_col])
    return np.array(X), np.array(y)

def load_and_prepare(path_prices, path_sentiment=None, seq_len=20):
    df = pd.read_csv(path_prices, parse_dates=["Date"])
    if path_sentiment:
        ds = pd.read_csv(path_sentiment, parse_dates=["Date"])
        df = merge_sentiment(df, ds)
    else:
        df["sentiment"] = 0.0
    # Sort & basic features
    df = df.sort_values("Date").reset_index(drop=True)
    # Use Close and sentiment plus simple moving average as features
    df["close_pct_change"] = df["Close"].pct_change().fillna(0.0)
    df["sma_5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
    # select features to scale
    feature_cols = ["Close", "close_pct_change", "sma_5", "sentiment"]
    df_feat = df[feature_cols].copy().fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_feat)
    # build sequences
    data_seq = []
    for i in range(len(scaled)):
        row = {col: scaled[i, idx] for idx, col in enumerate(feature_cols)}
        data_seq.append(row)
    # convert to structured numeric array
    numeric = []
    for r in data_seq:
        numeric.append([r[c] for c in feature_cols])
    numeric = np.array(numeric)
    X, y = create_sequences(numeric, target_col=None, seq_len=seq_len)
    # Since target_col=None in numeric arrays, get original unscaled close as y_raw
    y_raw = df["Close"].values[seq_len:]
    return X, y_raw, scaler, feature_cols

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train(args):
    seq_len = args.seq_len
    X, y, scaler, feature_cols = load_and_prepare(args.data, args.sentiment, seq_len=seq_len)
    # train-test split (time-series)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/lstm_model.h5", save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpoint, es])
    joblib.dump(scaler, "models/scaler.pkl")
    # Save a small metadata file
    import json
    with open("models/lstm_meta.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "seq_len": seq_len}, f)
    print("LSTM training complete. Model and scaler saved to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_stock.csv")
    parser.add_argument("--sentiment", default="data/sample_tweets_processed.csv")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    train(args)
