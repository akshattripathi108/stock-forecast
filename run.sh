#!/usr/bin/env bash
# Entrypoint: process sentiment, train LSTM, predict next 30 days using LSTM
set -e
python -u src/sentiment_analysis.py --in data/sample_tweets.csv --out data/sample_tweets_processed.csv || true
python -u src/train_lstm.py --data data/sample_stock.csv --sentiment data/sample_tweets_processed.csv --seq_len 10 --epochs 5 --batch_size 8
python -u src/predict_lstm.py --model models/lstm_model.h5 --scaler models/scaler.pkl --history data/sample_stock.csv --sentiment data/sample_tweets_processed.csv --days 30 --out predictions_lstm.csv
echo "Done. LSTM predictions saved to predictions_lstm.csv"
