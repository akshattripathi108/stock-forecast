
"""
sentiment_scraper.py
Two modes:
- snscrape mode: requires snscrape to be installed (no API key).
- twitter_api mode: requires Twitter API credentials.

Outputs a CSV: Date,username,text
"""
import argparse
import subprocess
import pandas as pd
import datetime as dt
import sys

def snscrape_search(query, since, until, max_results, out):
    # Requires snscrape installed and available on PATH
    cmd = ["snscrape", "--jsonl", "twitter-search", f"{query} since:{since} until:{until}"] 
    # We will collect up to max_results lines
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("snscrape not found. Install via `pip install snscrape` or use the Twitter API mode.")
        sys.exit(1)
    tweets = []
    for i, line in enumerate(proc.stdout):
        if i >= max_results:
            break
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tweets.append({"date": obj.get("date"), "username": obj.get("user", {}).get("username"), "content": obj.get("content")})
    df = pd.DataFrame(tweets)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} tweets to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="example")
    parser.add_argument("--out", default="data/sample_tweets.csv")
    args = parser.parse_args()
    if args.mode == "example":
        # write sample file
        import pandas as pd
        df = pd.DataFrame([{"date":"2024-01-01","username":"user1","content":"Stock market is doing great! Bulls are back."},
                           {"date":"2024-01-02","username":"user2","content":"I think AAPL will go down, worried about earnings."}])
        df.to_csv(args.out, index=False)
        print("Wrote sample tweets.")
    else:
        print("Other modes not implemented in this packaged script.")
