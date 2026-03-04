import os
import re
import json
import time
from pathlib import Path
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import requests

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "financial_documents")

# Ollama — local embedding model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# nomic-embed-text → 768-dim; mxbai-embed-large → 1024-dim
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

EMBEDDING_BATCH_SIZE = 20    # keep small for local Ollama
MIN_TEXT_LEN = 10             # characters — skip trivially short texts
MAX_TEXT_LEN = 2000           # characters — safe within nomic-embed-text context

# How long to wait for Ollama to become ready (seconds, 0 = wait forever)
OLLAMA_WARMUP_TIMEOUT = int(os.getenv("OLLAMA_WARMUP_TIMEOUT", "600"))
OLLAMA_WARMUP_INTERVAL = 5   # seconds between probes

_NULL_SENTINELS = {"", "nan", "none", "null", "n/a", "na", "undefined"}


# ─────────────────────────────────────────────────────────────────────────────
def wait_for_ollama() -> bool:
    """
    Probe Ollama /api/tags until it responds or timeout expires.
    Returns True if ready, raises RuntimeError if not available within timeout.
    This causes the Airflow task to FAIL and trigger its retry policy
    instead of silently producing payload-only (vector-less) points.
    """
    deadline = time.time() + OLLAMA_WARMUP_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=5,
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(EMBEDDING_MODEL in m for m in models):
                    print(
                        f"[Qdrant Loader] Ollama ready — "
                        f"{EMBEDDING_MODEL} available (probe #{attempt})"
                    )
                    return True
                else:
                    print(
                        f"[Qdrant Loader] Ollama up but {EMBEDDING_MODEL} "
                        f"not loaded yet (available: {models}) — waiting..."
                    )
        except Exception as e:
            print(f"[Qdrant Loader] Ollama probe #{attempt} failed: {e} — retrying...")
        time.sleep(OLLAMA_WARMUP_INTERVAL)

    raise RuntimeError(
        f"[Qdrant Loader] Ollama not available after {OLLAMA_WARMUP_TIMEOUT}s. "
        f"Ensure Ollama is running with `{EMBEDDING_MODEL}` pulled. "
        f"Task will retry per DAG retry policy."
    )


# ─────────────────────────────────────────────────────────────────────────────
def sanitise_text(raw: str) -> str | None:
    """
    Clean raw text from EODHD news/sentiment payloads before embedding.
    Returns None if the result is too short to be useful.

    Steps:
      1. Strip HTML tags  (e.g. <p>, <strong>, &amp;)
      2. Strip bare URLs  (http://... or www...)
      3. Collapse whitespace
      4. Remove non-printable / control characters
      5. Truncate to MAX_TEXT_LEN characters
      6. Reject if below MIN_TEXT_LEN
    """
    text = raw.strip()
    if text.lower() in _NULL_SENTINELS:
        return None

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )
    # Strip bare URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    # Remove control characters (keep CJK and standard unicode)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate at word boundary
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN].rsplit(" ", 1)[0]

    return text if len(text) >= MIN_TEXT_LEN else None


# ─────────────────────────────────────────────────────────────────────────────
def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"[Qdrant Loader] Created collection: {QDRANT_COLLECTION}")
    else:
        print(f"[Qdrant Loader] Collection exists: {QDRANT_COLLECTION}")


def _embed_single(text: str) -> list[float] | None:
    """Embed one pre-sanitised text. Returns None on failure.

    Uses the modern Ollama /api/embed endpoint (Ollama >= 0.1.26).
    Falls back to legacy /api/embeddings for older Ollama versions.
    """
    # Try modern endpoint first
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=None,
        )
        if resp.status_code == 404:
            raise requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) > 0 and embeddings[0]:
            return embeddings[0]
        print(f"[Qdrant Loader] /api/embed returned empty embeddings — trying legacy endpoint")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            pass  # fall through to legacy endpoint
        else:
            print(f"[Qdrant Loader] Embedding failed (modern endpoint): {e}")
            return None
    except Exception as e:
        print(f"[Qdrant Loader] Embedding failed (modern endpoint): {e}")
        # Fall through to legacy endpoint

    # Legacy fallback for Ollama < 0.1.26
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=None,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
        if embedding:
            return embedding
        print(f"[Qdrant Loader] Legacy /api/embeddings returned empty vector")
        return None
    except Exception as e:
        print(f"[Qdrant Loader] Embedding failed (legacy endpoint): {e}")
        return None


def get_embeddings(
    texts: list[str],
) -> tuple[list[list[float]], list[int]]:
    """
    Sanitise + embed texts in batches.
    Returns (vectors, valid_idx) — only successfully embedded entries.
    """
    embeddings: list[list[float]] = []
    valid_idx: list[int] = []
    skipped_sanitise = 0
    skipped_embed = 0

    # Pre-sanitise — track which original indices survive
    clean: list[tuple[int, str]] = []
    for i, raw in enumerate(texts):
        cleaned = sanitise_text(raw)
        if cleaned:
            clean.append((i, cleaned))
        else:
            skipped_sanitise += 1

    for batch_start in range(0, len(clean), EMBEDDING_BATCH_SIZE):
        batch = clean[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        batch_ok = 0
        for orig_idx, text in batch:
            emb = _embed_single(text)
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(orig_idx)
                batch_ok += 1
            else:
                skipped_embed += 1
        print(
            f"[Qdrant Loader] Embedded batch "
            f"{batch_start // EMBEDDING_BATCH_SIZE + 1} "
            f"({batch_ok}/{len(batch)} ok)"
        )

    total_skipped = skipped_sanitise + skipped_embed
    if total_skipped:
        print(
            f"[Qdrant Loader] Skipped {total_skipped} texts "
            f"({skipped_sanitise} sanitise, {skipped_embed} embed error)"
        )
    return embeddings, valid_idx


# ─────────────────────────────────────────────────────────────────────────────
def load_qdrant_for_agent_ticker(agent_name: str, ticker_symbol: str) -> int:
    """
    Main entry point called by the Airflow DAG.
    Probes Ollama first — raises RuntimeError (triggering task retry) if
    Ollama is not available, so vectors are never silently skipped.
    """
    # ── 1. Ensure Ollama is alive before doing any work ────────────────────
    wait_for_ollama()   # raises RuntimeError → Airflow marks task FAILED → retries

    agent_dir = BASE_ETL_DIR / agent_name / ticker_symbol
    metadata_path = agent_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[Qdrant Loader] No metadata.json for {agent_name}/{ticker_symbol}")
        return 0

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    client = get_client()
    ensure_collection(client)

    all_payloads: list[dict] = []
    all_texts: list[str] = []
    all_ids: list[str] = []

    for data_name, info in metadata.items():
        if info.get("storage_destination") != "qdrant_prep":
            continue

        csv_path = agent_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[Qdrant Loader] Missing CSV for {data_name} — skipping")
            continue

        if csv_path.stat().st_size < 4:
            print(f"[Qdrant Loader] Empty file for {data_name} — skipping")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Qdrant Loader] Failed to read {csv_path}: {e}")
            continue

        if df.empty or len(df.columns) == 0:
            print(f"[Qdrant Loader] No usable data in {data_name} — skipping")
            continue

        print(
            f"[Qdrant Loader] Preparing {data_name} "
            f"for {ticker_symbol} ({len(df)} rows)"
        )

        # Pick the best text column for semantic embedding
        text_col = None
        for c in ["content", "text", "body", "summary", "headline", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            df["__text__"] = df.astype(str).agg(" | ".join, axis=1)
            text_col = "__text__"

        for _, row in df.iterrows():
            raw_text = str(row.get(text_col, ""))
            if raw_text.strip().lower() in _NULL_SENTINELS:
                continue
            payload = row.to_dict()
            payload.update({
                "agent_name":    agent_name,
                "ticker_symbol": ticker_symbol,
                "data_name":     data_name,
                "source":        info.get("source", "unknown"),
            })
            all_ids.append(str(uuid.uuid4()))
            all_texts.append(raw_text)
            all_payloads.append(payload)

    if not all_texts:
        print(f"[Qdrant Loader] Nothing to embed for {agent_name}/{ticker_symbol}")
        return 0

    vectors, valid_idx = get_embeddings(all_texts)

    if not vectors:
        print(f"[Qdrant Loader] All embeddings failed for {agent_name}/{ticker_symbol}")
        return 0

    points = [
        qmodels.PointStruct(
            id=all_ids[i],
            vector=vectors[k],
            payload=all_payloads[i],
        )
        for k, i in enumerate(valid_idx)
    ]

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
        wait=True,
    )

    n = len(points)
    print(f"[Qdrant Loader] Upserted {n} points for {agent_name}/{ticker_symbol}")
    return n


if __name__ == "__main__":
    import sys
    agent  = sys.argv[1] if len(sys.argv) > 1 else "business_analyst"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    print(load_qdrant_for_agent_ticker(agent, ticker))
