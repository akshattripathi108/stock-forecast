
# Stock Market Analysis & Sentiment Prediction — Plug & Play Project

## Overview
This project performs stock market analysis and predicts the next month's closing prices for a chosen stock.
It also performs sentiment analysis on social media posts (e.g., Twitter) related to the stock market and
incorporates sentiment as a feature into the forecasting pipeline.

This repository is packaged to be plug-and-play. You can run it either with Docker (recommended) or locally with Python.

---

## Structure
```
stock_market_analysis_project/
├─ data/
│  ├─ sample_stock.csv            # sample historical stock data (daily)
│  └─ sample_tweets.csv           # sample tweets with sentiment labels
├─ notebooks/
│  └─ analysis_notebook.md        # high-level walkthrough (markdown)
├─ src/
│  ├─ fetch_data.py               # fetch stock data using yfinance (requires internet)
│  ├─ sentiment_scraper.py        # scrape tweets using snscrape or Twitter API (placeholders)
│  ├─ features.py                 # feature engineering (lags, rolling means, sentiment merge)
│  ├─ train_model.py              # training pipeline (sklearn)
│  ├─ predict.py                  # load model and predict next month's prices
│  └─ utils.py                    # helper utilities
├─ models/
│  └─ (empty)                     # trained model pickle will be saved here
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ run.sh
└─ README.md
```

## Quick start (Docker - recommended)
1. Install Docker and Docker Compose on your machine.
2. From the project root run:
   ```bash
   docker compose up --build
   ```
3. This will build an image that can run the training pipeline and serve simple predictions (if you adapt the scripts).

## Quick start (Local - without Docker)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\\Scripts\\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Inspect `data/sample_stock.csv` and optionally replace with your own CSV (format: Date,Open,High,Low,Close,Adj Close,Volume).
4. Run training:
   ```bash
   python src/train_model.py --data data/sample_stock.csv --out models/model.pkl
   ```
5. Predict next month's prices:
   ```bash
   python src/predict.py --model models/model.pkl --history data/sample_stock.csv --out predictions.csv
   ```

## Notes & Customization
- **Stock data**: `src/fetch_data.py` uses `yfinance`. You can fetch any ticker and save to `data/`.
- **Sentiment**: `src/sentiment_scraper.py` includes a snscrape example and placeholders for Twitter API usage. For large-scale scraping, follow platform terms.
- **Model choices**: The pipeline uses a classical ML model (sklearn) on lag features. For better performance, consider time-series models (ARIMA, Prophet, LSTM) or ensembling.
- **Privacy & API keys**: Do not commit production API keys. Use `.env` or environment variables.

## Files to inspect first
- `src/features.py` — how features are built (lags, rolling means, sentiment merge)
- `src/train_model.py` — training flow
- `src/predict.py` — prediction & output format

---

If you want, I can:
- Train a model on real stock data and include the trained model in the ZIP (requires internet to fetch data).
- Add a simple Flask app to serve predictions.
- Replace sample data with a specific ticker's historical data (tell me the ticker and date range).


## Automated fetch & train


A helper script `run_fetch_and_train.sh` is included to fetch the last 10 years of data for a ticker,
scrape tweets (example mode with sample tweets if snscrape not available), process sentiment,
train the LSTM, and output predictions.

Important: This zipper **does not** include a pre-trained model because this environment cannot access the internet
or perform the heavy training. Run the script locally (or in Docker) to fetch data and train the LSTM.

Steps (local machine with internet):
1. Install dependencies: `pip install -r requirements.txt`
2. Make script executable: `chmod +x run_fetch_and_train.sh`
3. Run: `./run_fetch_and_train.sh AAPL`
4. After completion, check `models/lstm_model.h5` and `predictions_AAPL.csv`.

If you prefer, upload your historical stock CSV and tweets CSV here and I can train the model using those files and return a ZIP with the trained model.


## Frontend

A static Bootstrap frontend is included under `frontend/index.html`. It is a demo dashboard that
loads embedded prediction & sentiment data and allows downloading the predictions as CSV. To view it,
open `frontend/index.html` in your browser. For a dynamic dashboard, integrate it with a Flask/Express API endpoint that serves `predictions_lstm.csv`.



## Flask API & Deployment

A Flask API is included at `src/app.py`. It serves:
- `GET /api/predictions?ticker=XXX&start=YYYY-MM-DD&end=YYYY-MM-DD` — returns predictions JSON.
- `GET /api/sentiment?ticker=XXX&start=YYYY-MM-DD&end=YYYY-MM-DD` — returns aggregated sentiment.
- `GET /` — serves the frontend `frontend/index.html`.

### Authentication
The API uses a simple API key. Set `API_KEY` in a `.env` file (or environment variable).
There is a `.env.example` included. Example:
```
API_KEY=your_secret_api_key
PORT=8080
```
Requests must include header `x-api-key: your_secret_api_key` or `?api_key=your_secret_api_key`.

### Run locally (development)
1. Create a virtualenv and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `API_KEY`.
3. Run:
   ```
   python -m src.app
   ```
4. Open `http://localhost:8080/` in your browser.

### Run with Docker
1. Build and run:
   ```
   docker compose up --build
   ```
2. Visit `http://localhost:8080/`. The API key is set via `docker-compose.yml` (default `changeme`).

### Notes
- The frontend will call the API endpoints at `/api/...`. If you deploy behind a reverse proxy, ensure paths are preserved.
- For production, consider using HTTPS, stronger auth (OAuth or JWT), rate-limiting, and proper logging.


## Models included

The trained RandomForest models and prediction CSVs have been added to the project root:

- rf_sp500.pkl
- scaler_sp500.pkl
- predictions_sp500.csv
- rf_nifty50.pkl
- scaler_nifty50.pkl
- predictions_nifty50.csv

Also packaged ZIP: final_models_predictions_rf_fixed.zip
