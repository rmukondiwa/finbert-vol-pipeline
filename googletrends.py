#!/usr/bin/env python3
"""
Clean Google Trends downloader using a user-provided stocklist.csv

- No Wikipedia
- No market cap logic
- Uses search_name + ticker
- Minimal + robust
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pytrends.request import TrendReq


# -----------------------------
# Paths
# -----------------------------

OUTPUT_DIR = Path("data")
RAW_DIR = OUTPUT_DIR / "raw"
LOG_DIR = OUTPUT_DIR / "logs"

STOCKLIST_FILE = OUTPUT_DIR / "stocklist.csv"
DOWNLOAD_LOG_FILE = LOG_DIR / "download_log.csv"
RUN_LOG_FILE = LOG_DIR / "run.log"


# -----------------------------
# Config
# -----------------------------

START_DATE = "2018-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

HL = "en-US"
TZ = 360
GEO = "US"
CHUNK_DAYS = 90  # bigger chunks = fewer requests (weekly data anyway)

SKIP_EXISTING = True
REQUEST_SLEEP_RANGE = (1, 3)  # aggressive but usually safe
TICKER_SLEEP_RANGE = (3, 6)
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 15


# -----------------------------
# Logging
# -----------------------------

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trends")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(RUN_LOG_FILE)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logging()


# -----------------------------
# Utils
# -----------------------------

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def sleep_random(bounds: Tuple[int, int]):
    s = random.uniform(*bounds)
    logger.info(f"Sleeping {s:.1f}s")
    time.sleep(s)


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def make_chunks(start: str, end: str, days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=days), end)
        chunks.append((cur, nxt))
        cur = nxt

    return chunks


def append_log(row: dict):
    df = pd.DataFrame([row])
    if DOWNLOAD_LOG_FILE.exists():
        df.to_csv(DOWNLOAD_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(DOWNLOAD_LOG_FILE, index=False)


# -----------------------------
# Data
# -----------------------------

def load_stocklist() -> pd.DataFrame:
    df = pd.read_csv(STOCKLIST_FILE)

    if "ticker" not in df.columns or "search_name" not in df.columns:
        raise ValueError("stocklist.csv must have: ticker, search_name")

    logger.info(f"Loaded {len(df)} tickers")
    return df


# -----------------------------
# Pytrends
# -----------------------------

# Global rate limiter (shared across threads)
from threading import Lock
_last_request_ts = 0.0
_rate_lock = Lock()
MIN_INTERVAL = 1.2  # seconds between requests globally (tune 1.0–2.0)

def throttle():
    global _last_request_ts
    with _rate_lock:
        now = time.time()
        wait = MIN_INTERVAL - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()

def build_pytrends():
    return TrendReq(hl=HL, tz=TZ)


def query_chunk(pytrends, search_name, ticker, start_dt, end_dt):
    timeframe = f"{start_dt.strftime('%Y-%m-%d')} {end_dt.strftime('%Y-%m-%d')}"

    pytrends.build_payload(
        kw_list=[search_name],
        timeframe=timeframe,
        geo=GEO,
    )

    df = pytrends.interest_over_time()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"date": "timestamp"})
    df["ticker"] = ticker
    df["search_name"] = search_name
    return df


def query_with_retries(pytrends, search_name, ticker, start_dt, end_dt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Query {ticker} {start_dt.date()} -> {end_dt.date()} (attempt {attempt})")
            # jitter + global throttle to avoid synchronized bursts
            time.sleep(random.uniform(0.2, 0.8))
            throttle()
            df = query_chunk(pytrends, search_name, ticker, start_dt, end_dt)
            time.sleep(random.uniform(*REQUEST_SLEEP_RANGE))
            return df

        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg
            wait = (BACKOFF_BASE_SECONDS * attempt) * (2 if is_429 else 1)
            logger.warning(f"Retry {ticker} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
            pytrends = build_pytrends()

    # Do not kill the whole ticker on one bad chunk; return empty and continue
    logger.error(f"Giving up chunk for {ticker} {start_dt} -> {end_dt}")
    return pd.DataFrame()


# -----------------------------
# Main download
# -----------------------------

def download_one(row):
    ticker = row["ticker"]
    search_name = row["search_name"]

    outfile = RAW_DIR / f"{sanitize_ticker(ticker)}.csv"

    if SKIP_EXISTING and outfile.exists():
        logger.info(f"Skipping {ticker}")
        return True, "skipped", 0

    chunks = make_chunks(START_DATE, END_DATE, CHUNK_DAYS)
    pytrends = build_pytrends()

    parts = []

    for i, (s, e) in enumerate(chunks, 1):
        logger.info(f"{ticker} chunk {i}/{len(chunks)}")
        df = query_with_retries(pytrends, search_name, ticker, s, e)
        if not df.empty:
            parts.append(df)
        else:
            logger.warning(f"Empty/failed chunk {ticker} {s.date()}->{e.date()}")

    if not parts:
        return False, "empty", 0

    df = pd.concat(parts).drop_duplicates(subset=["timestamp"])
    df.to_csv(outfile, index=False)

    logger.info(f"Saved {ticker} ({len(df)} rows)")
    sleep_random(TICKER_SLEEP_RANGE)

    return True, "success", len(df)


# -----------------------------
# Run
# -----------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 3  # reduced to lower 429 risk (tune 2–4)  # tune 3–6 depending on throttling


def process_row(row):
    ticker = row["ticker"]
    start = pd.Timestamp.utcnow()

    try:
        ok, status, n = download_one(row)
        end = pd.Timestamp.utcnow()

        append_log({
            "ticker": ticker,
            "status": status,
            "rows": n,
            "start": start,
            "end": end,
        })

        return ticker, status, n

    except Exception as e:
        logger.exception(f"FAIL {ticker} {e}")
        return ticker, "error", 0


def main():
    ensure_dirs()

    logger.info("START")

    df = load_stocklist()

    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _, row in df.iterrows():
            futures.append(executor.submit(process_row, row))

        for i, future in enumerate(as_completed(futures), 1):
            ticker, status, n = future.result()
            logger.info(f"[{i}/{len(futures)}] DONE {ticker} | {status} | rows={n}")

    logger.info("END")


if __name__ == "__main__":
    main()
