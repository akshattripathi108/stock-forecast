
# Analysis Walkthrough (high-level)

1. **Data collection**
   - Stock price history: use `yfinance` or any CSV with `Date,Open,High,Low,Close,Adj Close,Volume`.
   - Social media posts: use `snscrape` or Twitter API to collect tweets mentioning the stock or market terms.

2. **Preprocessing**
   - Parse dates; sort by date; forward/backward fill missing values.
   - Compute daily returns; create lag features (t-1, t-2, ...); rolling means and volatilities.
   - Clean tweets and compute sentiment scores (using a pre-trained sentiment model or `TextBlob` / `VADER`).

3. **Feature engineering**
   - Merge daily sentiment (average sentiment per day) with stock data.
   - Use lagged sentiment features (t-1, t-3).

4. **Modeling**
   - Use scikit-learn pipelines: scaler, feature selector, regressor (RandomForest / GradientBoosting).
   - Train on historical data up to a cutoff date; evaluate on a holdout set.

5. **Prediction**
   - Predict daily values for the next month (or aggregate to monthly).
   - Save predictions to `predictions.csv`.

6. **Extensions**
   - Replace regressor with LSTM for sequence prediction.
   - Add technical indicators (MACD, RSI).
   - Use more advanced NLP (transformer-based sentiment models).
