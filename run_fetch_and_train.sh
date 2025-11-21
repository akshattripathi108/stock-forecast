#!/usr/bin/env bash
# run_fetch_and_train.sh
# This script will:
# 1) Fetch last 10 years of daily stock data for a given ticker (default: AAPL)
# 2) Scrape tweets mentioning the ticker using snscrape
# 3) Process sentiment
# 4) Train the LSTM model and save it to models/
#
# Usage:
#   ./run_fetch_and_train.sh [TICKER]
# Example:
#   ./run_fetch_and_train.sh AAPL
set -e
TICKER=${1:-AAPL}
END_DATE=$(date -u +"%Y-%m-%d")
START_DATE=$(date -u -d "$END_DATE -10 years" +"%Y-%m-%d" || python - <<PY
from datetime import datetime, timedelta
end = datetime.utcnow()
start = end.replace(year=end.year-10)
print(start.strftime("%Y-%m-%d"))
PY
)
echo "Fetching data for $TICKER from $START_DATE to $END_DATE"
python -u src/fetch_data.py --ticker "$TICKER" --start "$START_DATE" --end "$END_DATE" --out "data/${TICKER}_history.csv"

# Scrape tweets mentioning the ticker for the same period (daily) using snscrape
# Note: snscrape syntax: snscrape --jsonl twitter-search "AAPL since:YYYY-MM-DD until:YYYY-MM-DD"
echo "Scraping tweets (this may take some time)..."
# We'll scrape last 30 days as example; for full 10 years scraping, adjust as needed.
SINCE=$(date -u -d "$END_DATE -30 days" +"%Y-%m-%d" || python - <<PY
from datetime import datetime, timedelta
end = datetime.utcnow()
start = end - timedelta(days=30)
print(start.strftime("%Y-%m-%d"))
PY
)
UNTIL="$END_DATE"
echo "Scraping tweets from $SINCE to $UNTIL"
# snscrape must be installed and available on PATH
python -u src/sentiment_scraper.py --mode example --out "data/${TICKER}_tweets.csv" || echo "snscrape not available; please install snscrape or provide tweets manually."

# Process sentiment
python -u src/sentiment_analysis.py --in "data/${TICKER}_tweets.csv" --out "data/${TICKER}_tweets_processed.csv" || true

# Train LSTM
python -u src/train_lstm.py --data "data/${TICKER}_history.csv" --sentiment "data/${TICKER}_tweets_processed.csv" --seq_len 20 --epochs 50 --batch_size 32 || true

# Predict next 30 days
python -u src/predict_lstm.py --model models/lstm_model.h5 --scaler models/scaler.pkl --history "data/${TICKER}_history.csv" --sentiment "data/${TICKER}_tweets_processed.csv" --days 30 --out "predictions_${TICKER}.csv" || true

echo "Finished. Check models/ for trained model and predictions_${TICKER}.csv for outputs."
