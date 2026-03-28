import requests
import time
import csv
import sys
import re
import pandas as pd
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG
# =========================
SUBREDDITS = ["stocks", "wallstreetbets"]
CHUNK_DAYS = 1
SLEEP = 0.3
MAX_RETRIES = 5

# =========================
# CLI DATE HANDLING
# =========================
if len(sys.argv) >= 3:
    START_DATE = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    END_DATE = datetime.strptime(sys.argv[2], "%Y-%m-%d")
else:
    START_DATE = datetime(2018, 1, 1)
    END_DATE = datetime.now(timezone.utc).replace(tzinfo=None)

OUTPUT_FILE = sys.argv[3] if len(sys.argv) >= 4 else "reddit_firehose.csv"

print(f"RUNNING FROM {START_DATE} TO {END_DATE} → {OUTPUT_FILE}")

# =========================
# LOAD S&P 500 DATA (DO NOT MODIFY FILE)
# =========================
sp500_df = pd.read_csv("data/sp500.csv")

# Build mappings
ticker_to_name = dict(zip(sp500_df["ticker"], sp500_df["name"]))

name_to_ticker = {}

for _, row in sp500_df.iterrows():
    name = row["name"].lower()
    ticker = row["ticker"]

    # full name
    name_to_ticker[name] = ticker

    # simplified name (remove suffixes)
    simple = re.sub(r"inc\\.?|corp\\.?|corporation|company|co\\.?|ltd\\.?|plc", "", name)
    simple = re.sub(r"[^a-z0-9 ]", " ", simple)
    simple = re.sub(r"\\s+", " ", simple).strip()

    if simple:
        name_to_ticker[simple] = ticker

ALL_TICKERS = set(ticker_to_name.keys())

BLACKLIST = {"IT", "ALL", "ON", "ARE", "FOR", "NOW", "CAN"}

# =========================
# HELPERS
# =========================
def to_epoch(dt):
    return int(dt.timestamp())


def fetch_data(subreddit, after, before):
    url = "https://api.pullpush.io/reddit/search/submission/"

    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "size": 500,
        "sort": "asc",
        "sort_type": "created_utc"
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 200:
                return r.json().get("data", [])

            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Sleeping {wait}s")
                time.sleep(wait)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2 ** attempt)

    return []


def extract_tickers(text):
    text_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text_lower = text_clean.lower()

    found = set()

    # 1. ticker match
    candidates = re.findall(r"\b[A-Z]{2,5}\b", text)
    for c in candidates:
        if c in ALL_TICKERS and c not in BLACKLIST:
            found.add(c)

    # 2. company name match
    for name, ticker in name_to_ticker.items():
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            found.add(ticker)

    return list(found)


def normalize(post):
    title = post.get("title", "") or ""
    selftext = post.get("selftext", "") or ""
    full_text = (title + " " + selftext).strip()

    tickers = extract_tickers(full_text)

    return {
        "timestamp": post.get("created_utc"),
        "full_text": full_text,
        "subreddit": post.get("subreddit"),
        "tickers": ",".join(tickers)
    }

# =========================
# MAIN
# =========================
def run():
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "full_text", "subreddit", "tickers"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for subreddit in SUBREDDITS:
            current = START_DATE

            while True:
                if current >= END_DATE:
                    break

                next_time = current + timedelta(days=CHUNK_DAYS)
                if next_time > END_DATE:
                    next_time = END_DATE

                print(f"[{subreddit}] {current} -> {next_time}")

                after = to_epoch(current)
                before = to_epoch(next_time)

                data = fetch_data(subreddit, after, before)

                if not data:
                    print("No data returned")
                    current = next_time
                    continue

                while data:
                    batch_size = len(data)
                    print(f"Writing batch of {batch_size}")

                    for post in data:
                        writer.writerow(normalize(post))

                    last_ts = data[-1]["created_utc"]

                    if last_ts <= after or last_ts >= before:
                        break

                    after = last_ts
                    data = fetch_data(subreddit, after, before)
                    time.sleep(SLEEP)

                current = next_time


if __name__ == "__main__":
    run()

# =========================
# USAGE
# =========================
"""
Test:
python3 reddithist.py 2018-01-01 2018-01-02 test.csv

Full:
nohup python3 -u reddithist.py 2018-01-01 2022-01-01 out1.csv > log1.txt 2>&1 &
nohup python3 -u reddithist.py 2022-01-01 2026-03-01 out2.csv > log2.txt 2>&1 &
"""
