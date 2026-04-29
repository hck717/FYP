# Documentation Index

Central index for repository documentation, grouped by audience and use case.

## Start Here

- Project overview and quick start: `README.md`
- Troubleshooting: `TROUBLESHOOTING.md`

## Architecture and Runtime

- Orchestration graph and runtime behavior: `orchestration/README.md`
- Agent catalog and responsibilities: `agents/README.md`
- Ingestion pipeline overview: `ingestion/README.md`
- Airflow DAG details: `ingestion/dags/README_eodhd_dag.md`

## Agent-Specific Docs

- Business Analyst: `agents/business_analyst/README.md`
- Quant Fundamental: `agents/quant_fundamental/README.md`
- Financial Modelling: `agents/financial_modelling/README.md`
- Web Search: `agents/web_search/README.md`

## Testing and Quality

- Test suite guide: `tests/README.md`

## Operations and Environment

- Local data directory notes: `data/README.md`
- Deployment and Hosting (Cloud/ngrok): `docs/DEPLOYMENT.md`

## Archive and Reference

- Legacy architecture setup guide: `docs/archive/legacy_architecture_setup.md`
- Sample outputs from agents: `docs/example_outputs/`
- Developer notes: `ingestion/notes/`

## Documentation Governance

Suggested maintenance routine for docs drift control:

1. Validate CLI commands against current parser implementations.
2. Validate env var names against `config.py` in each module.
3. Validate architecture diagrams against `orchestration/graph.py`.
4. Update each touched README metadata date after changes.
5. Keep historical references clearly labeled to avoid confusion.

Last updated: 2026-04-08
