import os
import json
from pathlib import Path
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "fyp-qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "agentic_analyst_docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1536"))


def get_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )


def get_embeddings(texts):
    import numpy as np
    return [np.zeros(EMBEDDING_DIM).tolist() for _ in texts]


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
