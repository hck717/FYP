# Testing Suite

Test suite for orchestration, agent integrations, and prompt/output behaviors.

## Structure

```text
tests/
├── integration/
│   ├── test_infra.py
│   ├── test_agent_db.py
│   ├── test_graph_nodes.py
│   ├── test_rlaif_memory.py
│   └── test_multi_ticker.py
├── prompts/
│   ├── test_planner_prompts.py
│   ├── test_ba_prompts.py
│   ├── test_qf_prompts.py
│   ├── test_fm_prompts.py
│   ├── test_ws_prompts.py
│   ├── test_sr_prompts.py
│   ├── test_macro_prompts.py
│   ├── test_insider_news_prompts.py
│   ├── test_summarizer_prompts.py
│   ├── test_hallucination_guard.py
│   ├── test_citation_accuracy.py
│   └── test_orchestration_prompt_telemetry.py
├── conftest.py
└── metrics.py
```

## Run Commands

From repo root:

```bash
pytest tests/ -v
```

Integration only:

```bash
pytest tests/integration/ -v -m integration --timeout=120
```

Prompt tests only:

```bash
pytest tests/prompts/ -v -m prompt --timeout=60
```

Unit-style subset (skip integration):

```bash
pytest tests/ -v -m "not integration"
```

## Markers

Configured in `pytest.ini`:

- `integration`
- `prompt`
- `unit`

## Dependencies and Environment

Integration tests typically require:

- PostgreSQL
- Neo4j
- Ollama (for embedding/local model paths)
- DeepSeek API access for DeepSeek-dependent flows

Core test env keys usually needed:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `OLLAMA_BASE_URL`
- `DEEPSEEK_API_KEY`

## Notes

- The orchestration runtime is planner -> parallel agents -> summarizer -> post_processing.
- Keep tests aligned with current node names and graph behavior in `orchestration/graph.py` and `orchestration/nodes.py`.
- Prompt telemetry outputs are written under `tests/outputs/` where relevant.

## Docs Validation Checklist

- Confirm marker names against `pytest.ini`
- Confirm test file paths still exist under `tests/`
- Confirm orchestration flow statement matches `orchestration/graph.py`
- Confirm required service dependencies remain accurate for integration tests

## Documentation Metadata

- Last updated: 2026-04-08
