
from flask import Flask, request, jsonify, send_from_directory, abort
from functools import wraps
import os
import pandas as pd
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

API_KEY = os.getenv("API_KEY", "d4o0ejhr01qk2nuecia0d4o0ejhr01qk2nueciag")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
PRED_DIR = os.path.join(os.path.dirname(__file__), "..")
SENT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if not key or key != API_KEY:
            return jsonify({"error":"unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/api/predictions")
@require_api_key
def api_predictions():
    ticker = request.args.get("ticker", "AAPL")
    # try several possible filenames
    candidates = [
        os.path.join(PRED_DIR, f"predictions_{ticker}.csv"),
        os.path.join(PRED_DIR, "predictions_lstm.csv"),
        os.path.join(PRED_DIR, "predictions.csv"),
        os.path.join(PRED_DIR, "data", "predictions_lstm.csv"),
    ]
    df = None
    for c in candidates:
        if os.path.exists(c):
            try:
                df = pd.read_csv(c, parse_dates=["Date"])
                break
            except Exception:
                continue
    if df is None:
        # return empty array
        return jsonify([])
    # allow date filtering
    start = request.args.get("start")
    end = request.args.get("end")
    if start:
        df = df[df["Date"] >= start]
    if end:
        df = df[df["Date"] <= end]
    return df.to_dict(orient="records")

@app.route("/api/sentiment")
@require_api_key
def api_sentiment():
    ticker = request.args.get("ticker", None)
    # look for processed sentiment files
    candidates = [
        os.path.join(SENT_DIR, f"{ticker}_tweets_processed.csv") if ticker else None,
        os.path.join(SENT_DIR, "sample_tweets_processed.csv"),
        os.path.join(SENT_DIR, "data_sample_tweets_processed.csv"),
    ]
    df = None
    for c in candidates:
        if c and os.path.exists(c):
            try:
                df = pd.read_csv(c, parse_dates=["Date"])
                break
            except Exception:
                continue
    if df is None:
        return jsonify([])
    # aggregate daily mean sentiment
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    daily = df.groupby("Date").agg({"sentiment":"mean", "content":"count"}).reset_index().rename(columns={"content":"count"})
    # date filtering
    start = request.args.get("start")
    end = request.args.get("end")
    if start:
        daily = daily[daily["Date"] >= pd.to_datetime(start).date()]
    if end:
        daily = daily[daily["Date"] <= pd.to_datetime(end).date()]
    return daily.to_dict(orient="records")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
