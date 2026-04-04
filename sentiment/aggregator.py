import logging
import pandas as pd
from pathlib import Path

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# PATHS
# =========================
INPUT = Path("data/reddit/reddit_scored.parquet")
OUTPUT_DIR = Path("data/signals")
OUTPUT = OUTPUT_DIR / "daily_signals.csv"

# =========================
# CONFIG
# =========================
# Rolling window for volume z-score (trading days)
# 30 days = ~1.5 months of history to establish "normal" mention volume
ROLLING_WINDOW = 30

# A post is a "sentiment shock" if mention volume is 2 standard deviations
# above its rolling mean — i.e. unusually high retail attention
SHOCK_THRESHOLD = 2.0

# =========================
# SENTIMENT DIRECTION MAP
# =========================
# Convert FinBERT label to a numeric direction:
# positive → +1, negative → -1, neutral → 0
# We then multiply by confidence to get a weighted score.
# e.g. "positive" with confidence 0.94 → score of +0.94
#      "negative" with confidence 0.80 → score of -0.80
#      "neutral"  with confidence 0.70 → score of  0.00
DIRECTION = {"positive": 1, "negative": -1, "neutral": 0}


def load_and_clean(path: Path) -> pd.DataFrame:
    logger.info(f"Loading {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} scored posts")

    # Drop posts with no ticker mentions
    df = df[df["tickers"].notna() & (df["tickers"] != "None") & (df["tickers"] != "")]
    logger.info(f"{len(df)} posts remain after dropping ticker-less rows")

    # Convert Unix epoch timestamp → date
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

    return df


def explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
    # Each post can mention multiple tickers e.g. "AAPL,MSFT,TSLA"
    # We split on comma and explode so each (post, ticker) is its own row.
    # This means one post contributes a sentiment signal to each ticker it mentions.
    df = df.copy()
    df["ticker"] = df["tickers"].str.split(",")
    df = df.explode("ticker")
    df["ticker"] = df["ticker"].str.strip()
    df = df[df["ticker"] != ""]
    logger.info(f"{len(df)} rows after exploding tickers (one row per post-ticker pair)")
    return df


def compute_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    # Map label → direction and multiply by confidence
    # This gives a score in [-1, +1] that captures both direction and conviction
    df["score"] = df["sentiment"].map(DIRECTION) * df["confidence"]
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Group by date + ticker and compute:
    # - sentiment_score: mean of confidence-weighted scores across all posts
    # - mention_count: how many posts mentioned this ticker today
    logger.info("Aggregating to daily (date, ticker) level...")

    daily = (
        df.groupby(["date", "ticker"])
        .agg(
            sentiment_score=("score", "mean"),
            mention_count=("score", "count"),
        )
        .reset_index()
    )

    logger.info(f"{len(daily)} (date, ticker) rows after aggregation")
    return daily


def compute_volume_zscore(daily: pd.DataFrame) -> pd.DataFrame:
    # For each ticker, compute a rolling z-score of its daily mention count.
    # z = (today's count - rolling mean) / rolling std
    #
    # This tells us: is today's mention volume unusually high relative to
    # the past 30 days? A z-score of 2.0 means 2 standard deviations above normal.
    #
    # We sort by date first so the rolling window is chronologically correct.
    logger.info(f"Computing {ROLLING_WINDOW}-day rolling volume z-score per ticker...")

    daily = daily.sort_values(["ticker", "date"])

    rolling = (
        daily.groupby("ticker")["mention_count"]
        .transform(lambda x: (x - x.rolling(ROLLING_WINDOW, min_periods=3).mean())
                              / x.rolling(ROLLING_WINDOW, min_periods=3).std())
    )

    daily["volume_zscore"] = rolling.fillna(0)

    return daily


def apply_shock_flag(daily: pd.DataFrame) -> pd.DataFrame:
    # A "sentiment shock" fires when volume z-score exceeds the threshold.
    # This is the regime signal — high shock → momentum + long straddle strategy.
    daily["shock_flag"] = (daily["volume_zscore"] >= SHOCK_THRESHOLD).astype(int)

    shock_count = daily["shock_flag"].sum()
    logger.info(f"Shock flag fired on {shock_count} ({shock_count/len(daily)*100:.1f}%) of (date, ticker) rows")

    return daily


def save(daily: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Final schema matches what the signal exporter will push to QuantConnect:
    # (date, ticker, sentiment_score, volume_zscore, shock_flag, source)
    daily["source"] = "reddit"
    daily = daily[["date", "ticker", "sentiment_score", "volume_zscore", "shock_flag", "source"]]

    daily.to_csv(OUTPUT, index=False)
    logger.info(f"Saved daily signals to {OUTPUT}")
    logger.info(f"\nSample output:\n{daily.head(10).to_string()}")


def main():
    df = load_and_clean(INPUT)
    df = explode_tickers(df)
    df = compute_sentiment_score(df)
    daily = aggregate_daily(df)
    daily = compute_volume_zscore(daily)
    daily = apply_shock_flag(daily)
    save(daily)


if __name__ == "__main__":
    main()
