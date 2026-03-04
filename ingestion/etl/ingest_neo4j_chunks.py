#!/usr/bin/env python3
"""Standalone Neo4j Chunk ingestion pipeline.

For each of the 5 supported tickers (AAPL, MSFT, GOOGL, TSLA, NVDA):
  1. Reads company_profile.csv (Description field) and financial_news.csv headlines
     from ingestion/etl/agent_data/business_analyst/{TICKER}/
  2. Calls qwen2.5:7b via Ollama to synthesise 5 analytical text chunks covering:
       - competitive_moat   (business model & competitive advantages)
       - risk_factors       (key risks from news + profile)
       - mda                (management discussion & analysis narrative)
       - earnings           (revenue/earnings outlook and recent performance)
       - news               (recent news summary and market developments)
  3. Embeds each chunk with all-MiniLM-L6-v2 (384-dim) via sentence_transformers
  4. Upserts into Neo4j as :Chunk nodes linked (Company)-[:HAS_CHUNK]->(Chunk)

The vector index `chunk_embedding` on :Chunk.embedding (384-dim) must already
exist in Neo4j — it is created by docker/init-neo4j.cypher during docker-compose up.

Usage:
    python ingestion/etl/ingest_neo4j_chunks.py [TICKER ...]

    # All 5 tickers:
    python ingestion/etl/ingest_neo4j_chunks.py

    # Single ticker:
    python ingestion/etl/ingest_neo4j_chunks.py AAPL
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ── Project paths ─────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.business_analyst.config import BusinessAnalystConfig
from agents.business_analyst.tools import LocalEmbeddingClient, Neo4jConnector

# ── Constants ─────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

# LLM config — use qwen2.5:7b (faster) with deepseek-r1:8b as fallback
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL   = os.getenv("CHUNK_SYNTH_MODEL", "qwen2.5:7b")
FALLBACK_MODEL  = "deepseek-r1:8b"
LLM_TIMEOUT     = None  # no timeout — let the request complete

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Data directory
AGENT_DATA_DIR = (
    Path(os.getenv("BASE_ETL_DIR", str(_REPO_ROOT / "ingestion" / "etl" / "agent_data")))
    / "business_analyst"
)

# Filing date stamped on synthesised chunks (today, since they are synthetic)
TODAY = date.today().isoformat()

# Section definitions: (section_key, human_label, synthesis_task)
SECTIONS = [
    (
        "competitive_moat",
        "Competitive Moat & Business Model",
        (
            "Write a detailed analyst note on {ticker}'s competitive moat and business model. "
            "Focus on: what the company sells, its primary revenue drivers, sustainable "
            "competitive advantages (brand, ecosystem, switching costs, scale), key markets "
            "served, and strategic positioning vs competitors. "
            "Use only the provided company profile and news. Do NOT invent financial figures. "
            "Length: 3-5 paragraphs."
        ),
    ),
    (
        "risk_factors",
        "Risk Factors",
        (
            "Write an analyst risk assessment for {ticker}. "
            "Identify the top 5-7 material risks from the company profile and recent news: "
            "macro/regulatory risks, competitive threats, product/technology risks, "
            "supply-chain or geopolitical risks, and valuation/market risks. "
            "For each risk, briefly explain why it is material. "
            "Use only the provided information. Do NOT invent data. "
            "Length: 4-6 paragraphs."
        ),
    ),
    (
        "mda",
        "Management Discussion & Analysis",
        (
            "Write a Management Discussion & Analysis (MD&A) style narrative for {ticker}. "
            "Cover: recent operational performance trends, key business segment dynamics, "
            "margin and profitability commentary, capital allocation approach (buybacks, dividends, "
            "R&D investment), and management's strategic priorities as reflected in news. "
            "Use only the provided company profile and recent news. Do NOT invent specific revenue "
            "or earnings figures beyond what is stated. Length: 4-6 paragraphs."
        ),
    ),
    (
        "earnings",
        "Earnings & Revenue Outlook",
        (
            "Write an earnings and revenue outlook note for {ticker}. "
            "Summarise: recent revenue/earnings trends visible in the news, analyst consensus "
            "commentary, guidance signals, key growth drivers and headwinds for the next 1-2 "
            "quarters, and any noteworthy beats or misses mentioned. "
            "Use only the provided news headlines and summaries. "
            "Do NOT fabricate specific EPS or revenue numbers. Length: 3-5 paragraphs."
        ),
    ),
    (
        "news",
        "Recent News & Market Developments",
        (
            "Write a structured news digest for {ticker} covering the most significant recent "
            "developments. Organise by theme: product launches, regulatory/legal events, "
            "M&A activity, executive changes, macro/geopolitical impacts, and analyst actions. "
            "Highlight why each development matters to an equity investor. "
            "Base this entirely on the provided news headlines/summaries. Length: 4-6 paragraphs."
        ),
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ollama_model(model: str) -> bool:
    """Return True if model is available in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False


def _pick_model() -> str:
    """Return PRIMARY_MODEL if available, else FALLBACK_MODEL, else raise."""
    if _check_ollama_model(PRIMARY_MODEL):
        print(f"[Ingest] Using LLM: {PRIMARY_MODEL}")
        return PRIMARY_MODEL
    print(f"[Ingest] {PRIMARY_MODEL} not found — trying fallback {FALLBACK_MODEL}")
    if _check_ollama_model(FALLBACK_MODEL):
        print(f"[Ingest] Using LLM: {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    raise RuntimeError(
        f"Neither {PRIMARY_MODEL} nor {FALLBACK_MODEL} is available in Ollama at {OLLAMA_BASE_URL}. "
        f"Run: ollama pull {PRIMARY_MODEL}"
    )


def _ollama_generate(model: str, prompt: str, max_retries: int = 2) -> str:
    """Call Ollama /api/generate and return the full response text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            last_exc = exc
            print(f"[Ingest] LLM call attempt {attempt} failed: {exc}")
            if attempt <= max_retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"LLM generation failed after {max_retries + 1} attempts: {last_exc}")


def _load_profile(ticker: str) -> Dict[str, Any]:
    """Load company profile data for a ticker. Returns dict with key fields."""
    path = AGENT_DATA_DIR / ticker / "company_profile.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {
        "name":         row.get("Name", ticker),
        "sector":       row.get("Sector", ""),
        "industry":     row.get("Industry", ""),
        "description":  row.get("Description", ""),
        "employees":    row.get("FullTimeEmployees", ""),
        "pe_ratio":     row.get("Highlights_PERatio", ""),
        "profit_margin":row.get("Highlights_ProfitMargin", ""),
        "market_cap":   row.get("Highlights_MarketCapitalization", ""),
    }


def _load_news_headlines(ticker: str, max_items: int = 40) -> List[str]:
    """Load news article titles + first 200 chars of content."""
    path = AGENT_DATA_DIR / ticker / "financial_news.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    items = []
    for _, row in df.head(max_items).iterrows():
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        snippet = content[:200].replace("\n", " ") if content and content != "nan" else ""
        if title and title != "nan":
            entry = f"- {title}"
            if snippet:
                entry += f": {snippet}"
            items.append(entry)
    return items


def _load_executives(ticker: str) -> str:
    """Return a brief executives summary string."""
    path = AGENT_DATA_DIR / ticker / "key_executives.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    lines = []
    for _, row in df.iterrows():
        name  = str(row.get("name", "")).strip()
        title = str(row.get("title", "")).strip()
        pay   = row.get("pay", "")
        if name and name != "nan" and title and title != "nan":
            pay_str = f" (pay: ${pay:,.0f})" if pay and str(pay) != "nan" else ""
            lines.append(f"  - {name}, {title}{pay_str}")
    return "\n".join(lines)


def _build_prompt(
    ticker: str,
    section_task: str,
    profile: Dict[str, Any],
    news_headlines: List[str],
    executives_summary: str,
) -> str:
    """Construct the LLM prompt for a given section synthesis task."""
    desc = (profile.get("description") or "")[:3000]
    news_block = "\n".join(news_headlines) if news_headlines else "(no news available)"
    exec_block = executives_summary if executives_summary else "(not available)"

    task = section_task.format(ticker=ticker)

    prompt = textwrap.dedent(f"""
    You are a senior equity research analyst. Your task:

    {task}

    === COMPANY PROFILE: {ticker} ===
    Name: {profile.get('name', ticker)}
    Sector: {profile.get('sector', '')}
    Industry: {profile.get('industry', '')}
    Employees: {profile.get('employees', '')}
    Business Description:
    {desc}

    === KEY EXECUTIVES ===
    {exec_block}

    === RECENT NEWS HEADLINES (up to 40 most recent) ===
    {news_block}

    === INSTRUCTIONS ===
    - Write in professional analyst prose (no bullet lists, no headers within your response).
    - Do NOT use phrases like "Based on the provided information" or "According to the context".
    - Do NOT hallucinate specific financial metrics not present above.
    - Be specific about {ticker}'s actual products, markets, and competitive dynamics.
    - Output ONLY the analytical text. No preamble, no postamble.
    """).strip()

    return prompt


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that deepseek-r1 emits."""
    import re
    # Remove think blocks (possibly multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def synthesise_chunks(ticker: str, model: str) -> List[Dict[str, Any]]:
    """Synthesise all section chunks for a single ticker. Returns list of chunk dicts."""
    print(f"\n[Ingest] === {ticker} ===")
    profile    = _load_profile(ticker)
    headlines  = _load_news_headlines(ticker, max_items=40)
    executives = _load_executives(ticker)

    if not profile.get("description"):
        print(f"[Ingest] WARNING: No company profile description for {ticker}")

    chunks: List[Dict[str, Any]] = []

    for section_key, section_label, section_task in SECTIONS:
        print(f"[Ingest]   Synthesising section: {section_key} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            prompt = _build_prompt(ticker, section_task, profile, headlines, executives)
            raw_text = _ollama_generate(model, prompt)
            text = _strip_thinking_tags(raw_text)
            if len(text) < 100:
                print(f"WARN (too short: {len(text)} chars) — skipping")
                continue
            elapsed = time.time() - t0
            print(f"OK ({len(text)} chars, {elapsed:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        chunk_id = f"neo4j::{ticker}::{section_key}::synth_0"
        chunks.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "section":     section_key,
            "filing_date": TODAY,
            "ticker":      ticker,
        })

    print(f"[Ingest] {ticker}: synthesised {len(chunks)}/{len(SECTIONS)} sections")
    return chunks


def embed_and_ingest(
    ticker: str,
    chunks: List[Dict[str, Any]],
    neo4j: Neo4jConnector,
    embedding_client: LocalEmbeddingClient,
) -> int:
    """Embed chunks and upsert into Neo4j. Returns number of chunks ingested."""
    if not chunks:
        return 0

    print(f"[Ingest]   Embedding {len(chunks)} chunks for {ticker} ...")
    embedded: List[Dict[str, Any]] = []
    for chunk in chunks:
        try:
            vec = embedding_client.embed(chunk["text"])
            row = dict(chunk)
            row["embedding"] = vec
            embedded.append(row)
        except Exception as exc:
            print(f"[Ingest]   WARNING: embedding failed for {chunk['chunk_id']}: {exc}")

    if not embedded:
        print(f"[Ingest]   ERROR: all embeddings failed for {ticker}")
        return 0

    print(f"[Ingest]   Upserting {len(embedded)} chunks to Neo4j ...")
    n = neo4j.insert_chunks(ticker, embedded)
    print(f"[Ingest]   Neo4j: {n} chunks upserted for {ticker}")
    return n


def verify_neo4j_chunks(neo4j: Neo4jConnector, tickers: List[str]) -> None:
    """Print chunk counts per ticker from Neo4j."""
    print("\n[Ingest] === Neo4j chunk verification ===")
    cypher = """
    MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk)
    RETURN count(ch) AS n
    """
    with neo4j.driver.session(database=None) as session:
        for ticker in tickers:
            result = session.run(cypher, ticker=ticker).single()
            n = result["n"] if result else 0
            print(f"[Ingest]   {ticker}: {n} Chunk nodes in Neo4j")


def ensure_vector_index(neo4j: Neo4jConnector, dimension: int = 384) -> None:
    """Create the chunk_embedding vector index if it does not exist."""
    check_cypher = """
    SHOW INDEXES
    YIELD name, type
    WHERE name = 'chunk_embedding'
    RETURN name
    """
    create_cypher = f"""
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {dimension},
        `vector.similarity_function`: 'cosine'
    }}}}
    """
    with neo4j.driver.session(database=None) as session:
        result = session.run(check_cypher).data()
        if result:
            print(f"[Ingest] Vector index 'chunk_embedding' already exists ({dimension}-dim)")
        else:
            print(f"[Ingest] Creating vector index 'chunk_embedding' ({dimension}-dim) ...")
            session.run(create_cypher)
            print("[Ingest] Vector index created.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(tickers: Optional[List[str]] = None) -> None:
    tickers = tickers or TICKERS
    print(f"[Ingest] Starting Neo4j Chunk ingestion for: {', '.join(tickers)}")
    print(f"[Ingest] Data directory: {AGENT_DATA_DIR}")
    print(f"[Ingest] Embedding model: {EMBEDDING_MODEL}")

    # Validate data directory
    if not AGENT_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Agent data directory not found: {AGENT_DATA_DIR}\n"
            "Set BASE_ETL_DIR env var or run from the repo root."
        )

    # Pick LLM
    model = _pick_model()

    # Initialise clients
    config = BusinessAnalystConfig()
    neo4j  = Neo4jConnector(config)
    emb    = LocalEmbeddingClient(EMBEDDING_MODEL)

    # Warm up embedding model
    print(f"[Ingest] Loading embedding model: {EMBEDDING_MODEL} ...")
    try:
        _ = emb.embed("warm up")
        print(f"[Ingest] Embedding model ready.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}\n"
            "Ensure sentence-transformers is installed: pip install sentence-transformers"
        ) from exc

    # Ensure vector index exists
    ensure_vector_index(neo4j, dimension=384)

    # Process each ticker
    total_chunks = 0
    for ticker in tickers:
        data_dir = AGENT_DATA_DIR / ticker
        if not data_dir.exists():
            print(f"[Ingest] WARNING: No data directory for {ticker} at {data_dir} — skipping")
            continue

        chunks = synthesise_chunks(ticker, model)
        n = embed_and_ingest(ticker, chunks, neo4j, emb)
        total_chunks += n

    # Verify
    verify_neo4j_chunks(neo4j, tickers)
    neo4j.close()

    print(f"\n[Ingest] Done. Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    if requested:
        unknown = [t for t in requested if t not in TICKERS]
        if unknown:
            print(f"[Ingest] Unknown tickers: {unknown}. Supported: {TICKERS}")
            sys.exit(1)
    main(requested)
