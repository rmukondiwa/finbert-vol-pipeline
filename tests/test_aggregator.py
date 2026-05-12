import pytest
import pandas as pd
from datetime import date, timedelta

from sentiment.aggregator import (
    explode_tickers,
    compute_sentiment_score,
    aggregate_daily,
    compute_volume_zscore,
    apply_shock_flag,
    SHOCK_THRESHOLD,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_scored_df(**overrides):
    """Minimal DataFrame matching finbert_scorer output schema."""
    base = {
        "timestamp": [1700000000, 1700000001],
        "full_text": ["AAPL is great", "TSLA moon"],
        "subreddit": ["stocks", "wallstreetbets"],
        "tickers": ["AAPL", "TSLA"],
        "sentiment": ["positive", "positive"],
        "confidence": [0.9, 0.8],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def make_daily_df(mention_counts, ticker="AAPL"):
    """Daily-aggregated DataFrame for z-score / shock-flag tests."""
    start = date(2024, 1, 1)
    n = len(mention_counts)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(n)],
        "ticker": [ticker] * n,
        "sentiment_score": [0.0] * n,
        "mention_count": mention_counts,
    })


# ── explode_tickers ───────────────────────────────────────────────────────────

class TestExplodeTickers:
    def test_single_ticker_produces_one_row(self):
        df = make_scored_df(tickers=["AAPL"], timestamp=[1], full_text=["x"],
                            subreddit=["stocks"], sentiment=["positive"], confidence=[0.9])
        result = explode_tickers(df)
        assert list(result["ticker"]) == ["AAPL"]

    def test_multiple_tickers_expand_to_one_row_each(self):
        df = make_scored_df(tickers=["AAPL,MSFT"], timestamp=[1], full_text=["x"],
                            subreddit=["stocks"], sentiment=["positive"], confidence=[0.9])
        result = explode_tickers(df)
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

    def test_empty_ticker_row_is_dropped(self):
        df = make_scored_df(tickers=["", "TSLA"])
        result = explode_tickers(df)
        assert "" not in result["ticker"].values
        assert "TSLA" in result["ticker"].values

    def test_whitespace_around_tickers_is_stripped(self):
        df = make_scored_df(tickers=[" AAPL , MSFT "], timestamp=[1], full_text=["x"],
                            subreddit=["stocks"], sentiment=["positive"], confidence=[0.9])
        result = explode_tickers(df)
        assert all(t == t.strip() for t in result["ticker"])

    def test_three_tickers_expand_correctly(self):
        df = make_scored_df(tickers=["AAPL,MSFT,NVDA"], timestamp=[1], full_text=["x"],
                            subreddit=["stocks"], sentiment=["positive"], confidence=[0.9])
        result = explode_tickers(df)
        assert len(result) == 3
        assert set(result["ticker"]) == {"AAPL", "MSFT", "NVDA"}


# ── compute_sentiment_score ───────────────────────────────────────────────────

class TestComputeSentimentScore:
    def _df(self, labels, confidences):
        return pd.DataFrame({
            "tickers": ["X"] * len(labels),
            "sentiment": labels,
            "confidence": confidences,
        })

    def test_positive_label_gives_positive_score(self):
        result = compute_sentiment_score(self._df(["positive"], [0.9]))
        assert pytest.approx(result["score"].iloc[0]) == 0.9

    def test_negative_label_gives_negative_score(self):
        result = compute_sentiment_score(self._df(["negative"], [0.8]))
        assert pytest.approx(result["score"].iloc[0]) == -0.8

    def test_neutral_label_scores_zero_regardless_of_confidence(self):
        result = compute_sentiment_score(self._df(["neutral"], [0.99]))
        assert result["score"].iloc[0] == 0.0

    def test_high_confidence_produces_larger_magnitude(self):
        low = compute_sentiment_score(self._df(["positive"], [0.5]))["score"].iloc[0]
        high = compute_sentiment_score(self._df(["positive"], [0.95]))["score"].iloc[0]
        assert high > low

    def test_mixed_labels(self):
        result = compute_sentiment_score(
            self._df(["positive", "negative", "neutral"], [1.0, 1.0, 1.0])
        )
        assert list(result["score"]) == [1.0, -1.0, 0.0]

    def test_score_bounded_within_minus_one_to_one(self):
        result = compute_sentiment_score(
            self._df(["positive", "negative"], [1.0, 1.0])
        )
        assert result["score"].between(-1.0, 1.0).all()


# ── aggregate_daily ───────────────────────────────────────────────────────────

class TestAggregateDaily:
    def _base_df(self):
        return pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "score": [0.9, 0.5, -0.3],
        })

    def test_two_posts_same_day_same_ticker_averaged(self):
        result = aggregate_daily(self._base_df())
        row = result[(result["date"] == date(2024, 1, 1)) & (result["ticker"] == "AAPL")]
        assert pytest.approx(row["sentiment_score"].iloc[0]) == 0.7

    def test_mention_count_equals_number_of_posts(self):
        result = aggregate_daily(self._base_df())
        row = result[(result["date"] == date(2024, 1, 1)) & (result["ticker"] == "AAPL")]
        assert row["mention_count"].iloc[0] == 2

    def test_different_days_produce_separate_rows(self):
        result = aggregate_daily(self._base_df())
        aapl = result[result["ticker"] == "AAPL"]
        assert len(aapl) == 2

    def test_different_tickers_same_day_produce_separate_rows(self):
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "ticker": ["AAPL", "MSFT"],
            "score": [0.9, -0.5],
        })
        result = aggregate_daily(df)
        assert len(result) == 2

    def test_single_negative_post(self):
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "ticker": ["TSLA"],
            "score": [-0.7],
        })
        result = aggregate_daily(df)
        assert pytest.approx(result["sentiment_score"].iloc[0]) == -0.7
        assert result["mention_count"].iloc[0] == 1


# ── compute_volume_zscore ─────────────────────────────────────────────────────

class TestComputeVolumeZscore:
    def test_insufficient_data_fills_zero(self):
        # fewer than min_periods=3 → NaN → filled 0
        df = make_daily_df([10, 20])
        result = compute_volume_zscore(df)
        assert (result["volume_zscore"] == 0).all()

    def test_spike_produces_zscore_above_threshold(self):
        # 29 stable days then a large spike
        df = make_daily_df([10] * 29 + [100])
        result = compute_volume_zscore(df)
        assert result["volume_zscore"].iloc[-1] > SHOCK_THRESHOLD

    def test_constant_series_stays_at_zero(self):
        # std=0 → NaN → filled 0
        df = make_daily_df([10] * 35)
        result = compute_volume_zscore(df)
        assert (result["volume_zscore"] == 0).all()

    def test_tickers_computed_independently(self):
        aapl = make_daily_df([10] * 29 + [100], ticker="AAPL")
        msft = make_daily_df([5] * 30, ticker="MSFT")   # flat → z-score stays 0
        df = pd.concat([aapl, msft]).reset_index(drop=True)
        result = compute_volume_zscore(df)

        aapl_z = result[result["ticker"] == "AAPL"]["volume_zscore"].iloc[-1]
        msft_z = result[result["ticker"] == "MSFT"]["volume_zscore"].iloc[-1]

        assert aapl_z > SHOCK_THRESHOLD
        assert msft_z == 0.0

    def test_output_column_exists(self):
        df = make_daily_df([10] * 5)
        result = compute_volume_zscore(df)
        assert "volume_zscore" in result.columns


# ── apply_shock_flag ──────────────────────────────────────────────────────────

class TestApplyShockFlag:
    def _df(self, zscores):
        return pd.DataFrame({"volume_zscore": zscores})

    def test_above_threshold_is_flagged(self):
        result = apply_shock_flag(self._df([SHOCK_THRESHOLD + 0.1]))
        assert result["shock_flag"].iloc[0] == 1

    def test_exactly_at_threshold_is_flagged(self):
        result = apply_shock_flag(self._df([SHOCK_THRESHOLD]))
        assert result["shock_flag"].iloc[0] == 1

    def test_below_threshold_not_flagged(self):
        result = apply_shock_flag(self._df([SHOCK_THRESHOLD - 0.01]))
        assert result["shock_flag"].iloc[0] == 0

    def test_zero_zscore_not_flagged(self):
        result = apply_shock_flag(self._df([0.0]))
        assert result["shock_flag"].iloc[0] == 0

    def test_shock_flag_column_is_integer(self):
        result = apply_shock_flag(self._df([3.0, 1.0]))
        assert result["shock_flag"].dtype in ["int32", "int64", int]

    def test_mixed_flags(self):
        result = apply_shock_flag(self._df([3.0, 0.5, SHOCK_THRESHOLD]))
        assert list(result["shock_flag"]) == [1, 0, 1]
