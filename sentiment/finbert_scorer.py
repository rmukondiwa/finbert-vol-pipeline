import logging
import modal
import pandas as pd
import pickle
import time
from pathlib import Path

# =========================
# LOGGING
# =========================
# Logs from the local entrypoint show in your terminal.
# Logs from the remote function show in your terminal AND on the Modal dashboard.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# PATHS
# =========================
INPUT = Path("data/reddit/reddit_full.parquet")
OUTPUT = Path("data/reddit/reddit_scored.parquet")

# =========================
# MODAL APP
# =========================
# Just telling the cloud environment what we need to run our code:
# - debian_slim: lightweight Linux base image
# - pip_install: packages available inside the container
app = modal.App("finbert-scorer")

image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "pandas")
)

# =========================
# REMOTE FUNCTION
# =========================
# This function runs in the cloud on Modal's infra via RPC.
# pickle is Python's built-in serialization format — it converts
# any Python object (list, dict, dataframe) into bytes for transport.
@app.function(image=image, gpu="T4", timeout=600)
def score_texts(texts: list[str]) -> list[dict]:
    import logging
    from transformers import pipeline

    # This logger runs inside the Modal container — its output
    # streams back to the terminal in real time via Modal's log transport
    remote_logger = logging.getLogger("modal.remote")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    remote_logger.info(f"[REMOTE] Received {len(texts)} texts after deserialization")
    remote_logger.info("[REMOTE] Loading FinBERT onto GPU...")

    finbert = pipeline(
        "text-classification",  # telling transformers we want to classify text
        model="ProsusAI/finbert",  # the specific FinBERT model trained on financial text
        device=0,           # 0 means use GPU
        truncation=True,    # clip posts longer than 512 tokens so we don't exceed model limits
        max_length=512,     # specifying max_length to be safe, even though truncation handles it
    )

    remote_logger.info("[REMOTE] FinBERT loaded. Running inference...")

    # batch_size=64: 64 posts processed simultaneously on the GPU
    t0 = time.time()
    results = finbert(texts, batch_size=64)
    elapsed = time.time() - t0

    remote_logger.info(f"[REMOTE] Inference complete in {elapsed:.1f}s ({len(texts)/elapsed:.0f} posts/sec)")
    remote_logger.info(f"[REMOTE] Serializing {len(results)} results to send back to client...")

    return results


# =========================
# LOCAL ENTRYPOINT
# =========================
# This runs on your laptop and orchestrates everything:
# 1. Load the parquet
# 2. Serialize texts and send to Modal (RPC call)
# 3. Receive and deserialize results
# 4. Attach results back to the dataframe
# 5. Save scored parquet locally
@app.local_entrypoint()
def main():
    logger.info(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    logger.info(f"Loaded {len(df)} posts")

    # get the list of posts to score, filling any missing values with empty strings
    texts = df["full_text"].fillna("").tolist()

    # ── RPC STUB: CLIENT SIDE ──────────────────────────────────────────
    # Before .remote() sends the data, Python pickles (serializes) the
    # texts list into raw bytes. We measure this to show what's happening.
    payload_bytes = len(pickle.dumps(texts))
    logger.info(f"[LOCAL] Serialized payload size: {payload_bytes / 1024:.1f} KB")
    logger.info(f"[LOCAL] Calling score_texts.remote() — sending payload to Modal GPU worker...")
    # ──────────────────────────────────────────────────────────────────

    start = time.time()
    results = score_texts.remote(texts)  # RPC call — blocks until remote function returns
    elapsed = time.time() - start

    # ── RPC STUB: CLIENT SIDE (response) ──────────────────────────────
    # Modal deserializes the returned bytes back into a Python list of dicts
    response_bytes = len(pickle.dumps(results))
    logger.info(f"[LOCAL] Received deserialized response: {response_bytes / 1024:.1f} KB")
    logger.info(f"[LOCAL] Round-trip RPC time: {elapsed:.1f}s ({len(texts)/elapsed:.0f} posts/sec)")
    # ──────────────────────────────────────────────────────────────────

    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [r["score"] for r in results]

    df.to_parquet(OUTPUT, index=False)
    logger.info(f"Saved scored parquet to {OUTPUT}")
    logger.info("\nSentiment distribution:\n" + df["sentiment"].value_counts().to_string())
