import hashlib
import logging
import time
import requests
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
INPUT = Path("data/signals/daily_signals.csv")
OUTPUT = Path("data/signals/qc_signals.csv")

# =========================
# QC CREDENTIALS (fill in when ready)
# =========================
QC_USER_ID = "YOUR_USER_ID"
QC_API_TOKEN = "YOUR_API_TOKEN"
QC_OBJECT_STORE_KEY = "sentiment/daily_signals.csv"  # path inside QC Object Store


# =========================
# STEP 1: LOAD + FORMAT FOR QC
# =========================
# QuantConnect's PythonData reader expects:
# - timestamp as a Unix epoch integer (seconds)
# - all signal columns flat in one row per (date, ticker)
def load_and_format(path: Path) -> pd.DataFrame:
    logger.info(f"Loading {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} daily signal rows")

    # Convert date string back to Unix epoch so QC can parse it
    df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9

    # Final QC schema — matches SentimentSignal(PythonData) on the QC side
    df = df[["timestamp", "ticker", "sentiment_score", "volume_zscore", "shock_flag", "source"]]

    logger.info(f"Formatted to QC schema: {list(df.columns)}")
    logger.info(f"\nSample:\n{df.head(5).to_string()}")

    return df


# =========================
# STEP 2: SAVE LOCALLY
# =========================
def save_locally(df: pd.DataFrame):
    df.to_csv(OUTPUT, index=False)
    logger.info(f"Saved QC-formatted signals to {OUTPUT}")


# =========================
# STEP 3: PUSH TO QC OBJECT STORE
# =========================
# QC Object Store uses Basic Auth with a HMAC-SHA256 signed timestamp.
# The API endpoint accepts a key (file path in QC) and the file contents.
# Once uploaded, QC's SentimentSignal(PythonData) reads it each data cycle.
def push_to_qc(df: pd.DataFrame):
    # ── AUTH ──────────────────────────────────────────────────────────
    # QC signs requests with: hash = SHA256(userId:apiToken:timestamp)
    ts = str(int(time.time()))
    hash_bytes = hashlib.sha256(f"{QC_USER_ID}:{QC_API_TOKEN}:{ts}".encode()).hexdigest()
    auth = (QC_USER_ID, f"{hash_bytes}:{ts}")
    # ──────────────────────────────────────────────────────────────────

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    logger.info(f"Pushing {len(csv_bytes) / 1024:.1f} KB to QC Object Store at key: {QC_OBJECT_STORE_KEY}")

    # TODO: fill in when QC credentials are ready
    # response = requests.post(
    #     "https://www.quantconnect.com/api/v2/object/set",
    #     auth=auth,
    #     data={"objectKey": QC_OBJECT_STORE_KEY},
    #     files={"objectData": ("signals.csv", csv_bytes, "text/csv")},
    # )
    # response.raise_for_status()
    # logger.info(f"QC Object Store response: {response.json()}")

    logger.info("TODO: QC push commented out — add credentials to activate")


# =========================
# MAIN
# =========================
def main():
    df = load_and_format(INPUT)
    save_locally(df)
    push_to_qc(df)
    logger.info("Signal export complete")


if __name__ == "__main__":
    main()
