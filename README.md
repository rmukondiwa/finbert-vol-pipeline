# Volatility Sentiment Strategy

A data pipeline that collects retail investor sentiment from Reddit and Google Trends, scores it with FinBERT, and produces daily trading signals that feed into a QuantConnect strategy.

The core idea: instead of using sentiment as a directional signal ("buy because people are bullish"), we use **sentiment intensity** as a regime indicator — high retail attention triggers a momentum + long straddle strategy, while low attention triggers pairs trading + iron condor.

---

## How It Works

```
Reddit / Google Trends
        │
        ▼
  Data Collection          ingestion/
  ─────────────
  Pull historical posts from r/stocks and r/wallstreetbets.
  Tag each post with S&P 500 ticker mentions.
  Collect Google search interest per ticker.
        │
        ▼
  FinBERT Scoring          sentiment/finbert_scorer.py
  ───────────────
  Send Reddit posts to a Modal cloud GPU.
  FinBERT classifies each post: positive / negative / neutral + confidence.
  Results returned and saved locally.
        │
        ▼
  Aggregation              sentiment/aggregator.py
  ───────────
  Convert post-level scores to daily signals per ticker.
  Compute confidence-weighted sentiment score.
  Compute mention volume z-score (how much above normal is today's chatter?).
  Flag sentiment shocks where z-score > 2.0.
        │
        ▼
  Signal Export            sentiment/signal_exporter.py
  ─────────────
  Format signals to QuantConnect schema.
  Push CSV to QC Object Store.
        │
        ▼
  QuantConnect
  ────────────
  SentimentSignal(PythonData) reads signals each data cycle.
  High shock → momentum + long straddle.
  Low shock  → pairs trading + iron condor.
```

---

## Project Structure

```
volatility-sentiment/
  ingestion/
    sp500.py              Scrape S&P 500 ticker list from Wikipedia → data/reference/sp500.csv
    reddithist.py         Pull historical Reddit posts via PullPush API → data/reddit/
    extractreddit.py      Incremental Reddit collection
    googletrends.py       Pull Google Trends per ticker → data/google_trends/
    parquet.py            Merge Reddit CSVs into a single parquet

  sentiment/
    finbert_scorer.py     Score Reddit posts with FinBERT on Modal GPU
    aggregator.py         Aggregate to daily (date, ticker) sentiment signals
    signal_exporter.py    Format and push signals to QuantConnect Object Store

  data/
    reddit/               Raw and scored Reddit data (parquet)
    google_trends/        Per-ticker Google Trends CSVs
    reference/            sp500.csv, stocklist.csv
    signals/              Daily signal output (CSV)
    logs/                 Run logs

  tests/
    test.py               Quick data inspection

  requirements.txt
```

---

## Signal Schema

The final output at `data/signals/qc_signals.csv`:

| Column | Description |
|---|---|
| `timestamp` | Unix epoch (seconds) |
| `ticker` | S&P 500 ticker symbol |
| `sentiment_score` | Confidence-weighted mean sentiment in [-1, +1] |
| `volume_zscore` | 30-day rolling z-score of mention count |
| `shock_flag` | 1 if volume_zscore ≥ 2.0 (high attention regime) |
| `source` | Data source (e.g. `reddit`) |

---

## Setup

```bash
pip install -r requirements.txt
modal setup   # authenticate with Modal for cloud GPU scoring
```

## Running the Pipeline

```bash
# 1. One-time: generate ticker reference file
python3 ingestion/sp500.py

# 2. Collect Reddit data (runs as two background jobs, 2018-2022 and 2022-present)
nohup python3 -u ingestion/reddithist.py 2018-01-01 2022-01-01 data/reddit/out1.csv > data/logs/log1.txt 2>&1 &
nohup python3 -u ingestion/reddithist.py 2022-01-01 2026-01-01 data/reddit/out2.csv > data/logs/log2.txt 2>&1 &

# 3. Merge into a single parquet
python3 ingestion/parquet.py

# 4. Score with FinBERT on Modal GPU
modal run sentiment/finbert_scorer.py

# 5. Aggregate to daily signals
python3 sentiment/aggregator.py

# 6. Export to QuantConnect
python3 sentiment/signal_exporter.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `PullPush API` | Reddit historical data |
| `pytrends` | Google Trends data |
| `FinBERT (ProsusAI)` | Financial sentiment NLP model |
| `Modal` | Cloud GPU inference (T4) via RPC |
| `pandas` | Data manipulation |
| `QuantConnect` | Backtesting and live trading |
