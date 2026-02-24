import os
import re
import json
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

EMBEDDING_BATCH_SIZE = 20   # keep small for local Ollama
MIN_TEXT_LEN = 10            # characters — skip trivially short texts
MAX_TEXT_LEN = 2000          # characters — nomic-embed-text context window ~8192 tokens
                             # 2000 chars ≈ 500 tokens, safe headroom

# Sentinel values that mean "no text"
_NULL_SENTINELS = {"", "nan", "none", "null", "n/a", "na", "undefined"}


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

    # Null-sentinel check
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

    # Remove non-printable / control characters (keep CJK and common unicode)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse runs of whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate — cut at word boundary if possible
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN].rsplit(" ", 1)[0]

    if len(text) < MIN_TEXT_LEN:
        return None

    return text


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
    """
    Embed one pre-sanitised text via Ollama /api/embeddings.
    Returns None on failure — caller drops the point.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"[Qdrant Loader] Embedding failed: {e}")
        return None


def get_embeddings(
    texts: list[str],
) -> tuple[list[list[float]], list[int]]:
    """
    Sanitise then embed texts in batches via Ollama.

    Returns:
        embeddings — valid embedding vectors
        valid_idx  — original indices that were successfully embedded
    """
    embeddings: list[list[float]] = []
    valid_idx: list[int] = []
    skipped_sanitise = 0
    skipped_embed = 0

    # Pre-sanitise all texts; track which originals survive
    clean: list[tuple[int, str]] = []   # (original_index, clean_text)
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

        # Guard: skip 0-byte or header-only files
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
            # Quick null check before sanitise (saves regex overhead)
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
            all_texts.append(raw_text)       # sanitise happens inside get_embeddings
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
    agent = sys.argv[1] if len(sys.argv) > 1 else "business_analyst"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    print(load_qdrant_for_agent_ticker(agent, ticker))
