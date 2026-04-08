# Local Data Storage

This directory stores local runtime data and mounted volumes used by the stack.

## Typical Contents

- `data/raw/` - raw downloaded/generated files (git-ignored)
- `data/logs/` - local logs (git-ignored)
- `data/postgres_data/` - PostgreSQL persistent volume
- `data/neo4j_data/` - Neo4j persistent volume
- `data/neo4j_logs/` - Neo4j logs
- `data/neo4j_import/` - Neo4j import directory
- `data/neo4j_plugins/` - Neo4j plugins directory
- `data/textual data/` - local PDF/textual sources used by textual ingestion

## Notes

- Do not commit sensitive data or large datasets.
- Most runtime contents are generated and can be rebuilt from ingestion flows.

## Documentation Metadata

- Last updated: 2026-04-08
