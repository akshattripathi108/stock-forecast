
"""
fetch_data.py
Fetch historical stock data using yfinance and save to CSV.
"""
import argparse
import yfinance as yf
import pandas as pd

def fetch(ticker, start, end, out):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", default="data/stock.csv")
    args = parser.parse_args()
    fetch(args.ticker, args.start, args.end, args.out)
