# ingestion/etl/load_neo4j.py
import os
import json
import hashlib
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

# FMP data_names that contain strategy / narrative content and should be
# loaded as (Company)-[:HAS_STRATEGY]->(Strategy) nodes.
# Matches actual data_name values used by the FMP scraper in
# dag_fmp_ingestion_unified.py AGENT_CONFIGS.
_STRATEGY_DATA_NAMES = {
    "strategy", "narrative", "mda", "md&a",        # legacy / EODHD names
    "management_discussion", "company_outlook",      # FMP actual names
    "company_notes", "business_description",         # FMP actual names
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _load_company_profile(tx, ticker_symbol, row):
    props = {k: v for k, v in row.items() if pd.notna(v)}
    props["ticker"] = ticker_symbol
    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        SET c += $props
        """,
        ticker=ticker_symbol,
        props=props,
    )


def _load_risk_factor(tx, ticker_symbol, row):
    """
    FIX 1: Relationship direction corrected.
    BEFORE: (Risk)-[:AFFECTS]->(Company)  — wrong direction
    AFTER:  (Company)-[:FACES_RISK]->(Risk) — matches fetch_graph_facts() traversal:
            MATCH (c:Company {ticker})-[r]->(n)
            WHERE type(r) IN ['HAS_STRATEGY','FACES_RISK','OFFERS_PRODUCT']
    """
    desc    = row.get("risk", row.get("description", "")) or ""
    risk_id = row.get("id") or f"{ticker_symbol}:{hash(desc)}"
    props   = {k: v for k, v in row.items() if pd.notna(v)}
    props["risk_id"] = risk_id
    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        MERGE (r:Risk {risk_id: $risk_id})
        SET r += $props
        MERGE (c)-[:FACES_RISK]->(r)
        """,
        ticker=ticker_symbol,
        risk_id=risk_id,
        props=props,
    )


def _load_strategy(tx, ticker_symbol, row, label="Strategy"):
    """
    FIX 2: Relationship direction corrected + renamed.
    BEFORE: (Strategy)-[:APPLIES_TO]->(Company)  — wrong direction + wrong name
    AFTER:  (Company)-[:HAS_STRATEGY]->(Strategy) — matches fetch_graph_facts():
            WHERE type(r) IN ['HAS_STRATEGY','FACES_RISK','OFFERS_PRODUCT']
    """
    text     = row.get("text") or row.get("narrative") or row.get("description") or ""
    strat_id = row.get("id") or f"{ticker_symbol}:{hash(text)}"
    props    = {k: v for k, v in row.items() if pd.notna(v)}
    props["strategy_id"] = strat_id
    tx.run(
        f"""
        MERGE (c:Company {{ticker: $ticker}})
        MERGE (s:{label} {{strategy_id: $strategy_id}})
        SET s += $props
        MERGE (c)-[:HAS_STRATEGY]->(s)
        """,
        ticker=ticker_symbol,
        strategy_id=strat_id,
        props=props,
    )


def _load_fact(tx, ticker_symbol, data_name, row):
    """
    Uses MERGE with deterministic fact_id to prevent duplicate Fact nodes.
    Relationship: (Company)-[:HAS_FACT]->(Fact) — consistent with outbound traversal.
    """
    props   = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
    fact_id = hashlib.md5(
        f"{ticker_symbol}:{data_name}:{json.dumps(props, sort_keys=True)}".encode()
    ).hexdigest()
    props["fact_id"] = fact_id
    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        MERGE (f:Fact {fact_id: $fact_id})
        SET f += $props
        SET f.data_name = $data_name
        MERGE (c)-[:HAS_FACT]->(f)
        """,
        ticker=ticker_symbol,
        fact_id=fact_id,
        data_name=data_name,
        props=props,
    )


def _load_peer_relationships(session, ticker_symbol: str, peers_csv_path: Path) -> int:
    """Create (Company)-[:COMPETES_WITH]->(Company) edges from stock_peers CSV.

    FMP /stock_peers returns: [{"symbol": "AAPL", "peersList": ["MSFT", "GOOGL", ...]}]
    The CSV will have columns 'symbol' and 'peersList' (as string of JSON array).

    Both business_analyst/tools.py (fetch_graph_facts) and
    financial_modelling/tools.py (Neo4jPeerSelector.get_peers) query
    COMPETES_WITH relationships — this function creates them so Neo4j peer
    resolution works without falling back to the static map.
    """
    if not peers_csv_path.exists():
        return 0

    try:
        df = pd.read_csv(peers_csv_path)
    except Exception as e:
        print(f"[Neo4j Loader] Failed to read stock_peers CSV: {e}")
        return 0

    if df.empty:
        return 0

    # FMP returns list of objects; each row has 'symbol' and 'peersList'
    # peersList may be a JSON-array string like '["MSFT","GOOGL"]'
    count = 0
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", ticker_symbol)).strip().upper() or ticker_symbol
        peers_raw = row.get("peersList", "[]")
        try:
            if isinstance(peers_raw, str):
                peers = json.loads(peers_raw)
            elif isinstance(peers_raw, list):
                peers = peers_raw
            else:
                peers = []
        except Exception:
            peers = []

        for peer in peers:
            peer = str(peer).strip().upper()
            if not peer or peer == symbol:
                continue
            session.run(
                """
                MERGE (c:Company {ticker: $ticker})
                MERGE (p:Company {ticker: $peer})
                MERGE (c)-[:COMPETES_WITH]->(p)
                MERGE (p)-[:COMPETES_WITH]->(c)
                """,
                ticker=symbol,
                peer=peer,
            )
            count += 1

    if count:
        print(f"[Neo4j Loader] Created {count} COMPETES_WITH edges for {ticker_symbol}")
    return count


def load_neo4j_for_agent_ticker(agent_name: str, ticker_symbol: str) -> int:
    agent_dir     = BASE_ETL_DIR / agent_name / ticker_symbol
    metadata_path = agent_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[Neo4j Loader] No metadata.json for {agent_name}/{ticker_symbol}")
        return 0

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    count  = 0
    driver = get_driver()

    with driver.session() as session:
        for data_name, info in metadata.items():
            dest = info.get("storage_destination")
            if dest != "neo4j":
                continue

            csv_path = agent_dir / f"{data_name}.csv"
            if not csv_path.exists():
                print(f"[Neo4j Loader] Missing CSV for {data_name} at {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[Neo4j Loader] Failed to read {csv_path}: {e}")
                continue

            if df.empty:
                continue

            print(f"[Neo4j Loader] Loading {data_name} for {ticker_symbol} ({len(df)} rows)")

            if "profile" in data_name:
                for _, row in df.iterrows():
                    session.execute_write(_load_company_profile, ticker_symbol, row.to_dict())
                    count += 1

            elif "risk" in data_name:
                # FIX 1: now creates (Company)-[:FACES_RISK]->(Risk)
                for _, row in df.iterrows():
                    session.execute_write(_load_risk_factor, ticker_symbol, row.to_dict())
                    count += 1

            elif any(k in data_name for k in _STRATEGY_DATA_NAMES):
                # FIX 2: now creates (Company)-[:HAS_STRATEGY]->(Strategy)
                for _, row in df.iterrows():
                    session.execute_write(
                        _load_strategy,
                        ticker_symbol,
                        row.to_dict(),
                        label="Strategy",
                    )
                    count += 1

            else:
                for _, row in df.iterrows():
                    session.execute_write(_load_fact, ticker_symbol, data_name, row)
                    count += 1

        # ── COMPETES_WITH peer relationships ──────────────────────────────────
        # Load from stock_peers CSV if present (FMP financial_modeling agent).
        # This enables Neo4j peer resolution in Neo4jPeerSelector.get_peers().
        peers_csv = agent_dir / "stock_peers.csv"
        count += _load_peer_relationships(session, ticker_symbol, peers_csv)

    driver.close()
    return count


if __name__ == "__main__":
    print(load_neo4j_for_agent_ticker("business_analyst", "AAPL"))
