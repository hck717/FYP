import os
import json
from pathlib import Path
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import requests

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "fyp-qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "agentic_analyst_docs")

# Ollama instead of OpenAI
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# nomic-embed-text → 768-dim; if using mxbai-embed-large → 1024-dim
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

EMBEDDING_BATCH_SIZE = 20  # Ollama is local, keep batches small


def get_client():
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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via local Ollama — no API key needed."""
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]

        for text in batch:
            text = text.strip() or "N/A"
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    timeout=30,
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"[Qdrant Loader] Embedding failed for text snippet: {e}")
                # Fallback: zero vector so the batch doesn't crash
                all_embeddings.append([0.0] * EMBEDDING_DIM)

        print(
            f"[Qdrant Loader] Embedded batch "
            f"{i // EMBEDDING_BATCH_SIZE + 1} ({len(batch)} texts)"
        )

    return all_embeddings


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

    payloads = []
    texts = []
    ids = []

    for data_name, info in metadata.items():
        dest = info.get("storage_destination")
        if dest != "qdrant_prep":
            continue

        csv_path = agent_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[Qdrant Loader] Missing CSV for {data_name} at {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Qdrant Loader] Failed to read {csv_path}: {e}")
            continue

        if df.empty:
            continue

        print(f"[Qdrant Loader] Preparing {data_name} for {ticker_symbol} ({len(df)} rows)")

        text_col = None
        for c in ["text", "content", "body", "headline", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            df["__text__"] = df.astype(str).agg(" | ".join, axis=1)
            text_col = "__text__"

        for _, row in df.iterrows():
            text = str(row[text_col])
            payload = row.to_dict()
            payload.update(
                {
                    "agent_name": agent_name,
                    "ticker_symbol": ticker_symbol,
                    "data_name": data_name,
                    "source": info.get("source", "unknown"),
                }
            )
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(text)
            payloads.append(payload)

    if not texts:
        print(f"[Qdrant Loader] Nothing to upsert for {agent_name}/{ticker_symbol}")
        return 0

    vectors = get_embeddings(texts)

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ],
    )

    print(f"[Qdrant Loader] Upserted {len(ids)} points for {agent_name}/{ticker_symbol}")
    return len(ids)


if __name__ == "__main__":
    print(load_qdrant_for_agent_ticker("business_analyst", "AAPL"))
