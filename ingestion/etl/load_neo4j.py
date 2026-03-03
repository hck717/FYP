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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme_neo4j_password")


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

            elif any(k in data_name for k in ["strategy", "narrative", "mda", "md&a"]):
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

    driver.close()
    return count


if __name__ == "__main__":
    print(load_neo4j_for_agent_ticker("business_analyst", "AAPL"))
