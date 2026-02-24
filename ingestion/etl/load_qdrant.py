import os
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

EMBEDDING_BATCH_SIZE = 20  # keep small for local Ollama

# Minimum text length — skip embedding tokens that are too short
MIN_TEXT_LEN = 8


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
    Embed one text via Ollama /api/embeddings.
    Returns None on failure so caller can decide to skip or use zero-vector.
    """
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return None  # too short — caller will skip this point entirely
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
    Embed texts in batches via Ollama.

    Returns:
        embeddings  — list of valid embedding vectors
        valid_idx   — indices into the original `texts` list that were
                      successfully embedded (i.e. caller can zip these
                      back to the original ids/payloads lists)
    """
    embeddings: list[list[float]] = []
    valid_idx: list[int] = []
    skipped = 0

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = []
        batch_valid = []

        for j, text in enumerate(batch):
            emb = _embed_single(text)
            if emb is not None:
                batch_embeddings.append(emb)
                batch_valid.append(i + j)
            else:
                skipped += 1

        embeddings.extend(batch_embeddings)
        valid_idx.extend(batch_valid)
        print(
            f"[Qdrant Loader] Embedded batch "
            f"{i // EMBEDDING_BATCH_SIZE + 1} "
            f"({len(batch_valid)}/{len(batch)} ok)"
        )

    if skipped:
        print(f"[Qdrant Loader] Skipped {skipped} texts (too short / embed error)")
    return embeddings, valid_idx


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
        dest = info.get("storage_destination")
        if dest != "qdrant_prep":
            continue

        csv_path = agent_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[Qdrant Loader] Missing CSV for {data_name} — skipping")
            continue

        # Guard: skip truly empty files (0 bytes or header-only)
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

        # Determine which column to use as the embedding text
        text_col = None
        for c in ["content", "text", "body", "summary", "headline", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            # Fallback: concatenate all columns
            df["__text__"] = df.astype(str).agg(" | ".join, axis=1)
            text_col = "__text__"

        for _, row in df.iterrows():
            raw_text = str(row.get(text_col, ""))
            # Skip rows where the text is NaN/None/empty string
            if raw_text.strip().lower() in ("", "nan", "none", "null"):
                continue

            payload = row.to_dict()
            payload.update(
                {
                    "agent_name": agent_name,
                    "ticker_symbol": ticker_symbol,
                    "data_name": data_name,
                    "source": info.get("source", "unknown"),
                }
            )
            all_ids.append(str(uuid.uuid4()))
            all_texts.append(raw_text)
            all_payloads.append(payload)

    if not all_texts:
        print(f"[Qdrant Loader] Nothing to embed for {agent_name}/{ticker_symbol}")
        return 0

    # Embed — returns only successfully embedded vectors + their original indices
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

    print(
        f"[Qdrant Loader] Upserted {len(points)} points "
        f"for {agent_name}/{ticker_symbol}"
    )
    return len(points)


if __name__ == "__main__":
    print(load_qdrant_for_agent_ticker("business_analyst", "AAPL"))
