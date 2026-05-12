import sys
import importlib
import pytest
import pandas as pd
from datetime import date


# ── googletrends pure functions ───────────────────────────────────────────────
# Import only the pure utility functions — avoids triggering pytrends network calls.

from ingestion.googletrends import make_chunks, sanitize_ticker


class TestMakeChunks:
    def test_single_chunk_when_range_fits(self):
        chunks = make_chunks("2024-01-01", "2024-03-01", days=90)
        assert len(chunks) == 1

    def test_multiple_chunks_for_long_range(self):
        chunks = make_chunks("2020-01-01", "2021-01-01", days=90)
        assert len(chunks) > 1

    def test_last_chunk_does_not_exceed_end(self):
        chunks = make_chunks("2024-01-01", "2024-06-01", days=90)
        end = pd.Timestamp("2024-06-01")
        assert chunks[-1][1] <= end

    def test_chunks_are_contiguous(self):
        chunks = make_chunks("2024-01-01", "2024-12-01", days=90)
        for i in range(1, len(chunks)):
            assert chunks[i][0] == chunks[i - 1][1]

    def test_start_of_first_chunk_matches_input(self):
        chunks = make_chunks("2024-03-15", "2024-09-15", days=90)
        assert chunks[0][0] == pd.Timestamp("2024-03-15")


class TestSanitizeTicker:
    def test_dot_replaced_with_hyphen(self):
        assert sanitize_ticker("BRK.B") == "BRK-B"

    def test_no_dot_unchanged(self):
        assert sanitize_ticker("AAPL") == "AAPL"

    def test_multiple_dots_all_replaced(self):
        assert sanitize_ticker("A.B.C") == "A-B-C"


# ── reddithist ticker extraction ──────────────────────────────────────────────
# reddithist.py has module-level sys.argv parsing and CSV loading.
# We patch sys.argv before import so it takes the safe else-branch defaults.
# Tests must be run from the project root so data/reference/sp500.csv resolves.

@pytest.fixture(scope="module")
def rh():
    original = sys.argv[:]
    sys.argv = ["reddithist.py"]
    try:
        if "reddithist" in sys.modules:
            return sys.modules["reddithist"]
        return importlib.import_module("reddithist")
    finally:
        sys.argv = original


class TestExtractTickers:
    def test_explicit_ticker_found(self, rh):
        result = rh.extract_tickers("AAPL is going to the moon")
        assert "AAPL" in result

    def test_blacklisted_ticker_excluded(self, rh):
        # "IT" is Gartner's ticker but is in BLACKLIST to avoid false positives
        result = rh.extract_tickers("IT is definitely not a ticker here")
        assert "IT" not in result

    def test_empty_string_returns_empty(self, rh):
        assert rh.extract_tickers("") == []

    def test_no_ticker_text_returns_empty(self, rh):
        result = rh.extract_tickers("just some random words with no finance")
        assert result == []

    def test_multiple_tickers_all_found(self, rh):
        result = rh.extract_tickers("AAPL TSLA MSFT looking bullish")
        assert "AAPL" in result
        assert "TSLA" in result
        assert "MSFT" in result

    def test_company_name_maps_to_ticker(self, rh):
        # name_to_ticker stores "apple inc" (simplified), so the full form must appear
        result = rh.extract_tickers("Apple Inc is releasing a new product")
        assert "AAPL" in result

    def test_result_is_list(self, rh):
        result = rh.extract_tickers("AAPL")
        assert isinstance(result, list)

    def test_no_duplicate_tickers(self, rh):
        result = rh.extract_tickers("AAPL AAPL buy more AAPL")
        assert len(result) == len(set(result))
