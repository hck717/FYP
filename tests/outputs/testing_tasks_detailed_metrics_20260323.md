# Testing Tasks Detailed Metrics Report

## Scope

This report consolidates all current artifacts in `tests/outputs` after cleanup and presents detailed metrics for each testing task.

Active source artifacts used:
- `tests/outputs/pytest_all_with_metrics_20260323_171441.log`
- `tests/outputs/test_metrics.jsonl`
- `tests/outputs/test_metrics_summary.md`

Cleanup performed:
- Removed outdated/duplicate logs and summaries.

## Final Test Outcome

- Collected: **140**
- Passed: **140**
- Failed: **0**
- Skipped: **0**
- Errors: **0**
- Total wall time: **386,735.417 ms** (about 6m 26s)

## Global Metrics

- Metrics records: **155**
- Total measured duration (sum): **386,478.565 ms**
- Average record duration: **2,493.41 ms**
- Total captured output snapshot size: **1,854 bytes**
- Average reported latency: **25.799 ms**
- Maximum reported latency: **311.495 ms**
- Average connection time: **3.225 ms**
- Maximum connection time: **6.765 ms**
- Total reported data amount units: **803**

## Per Testing Task (By Test File)

| Testing task | Tests | Pass | Fail | Total duration (ms) | Avg/test (ms) |
|---|---:|---:|---:|---:|---:|
| `tests/integration/test_agent_db.py` | 14 | 14 | 0 | 363,489.890 | 25,963.564 |
| `tests/integration/test_graph_nodes.py` | 16 | 16 | 0 | 105.472 | 6.592 |
| `tests/integration/test_infra.py` | 8 | 8 | 0 | 374.889 | 46.861 |
| `tests/integration/test_multi_ticker.py` | 13 | 13 | 0 | 0.546 | 0.042 |
| `tests/integration/test_rlaif_memory.py` | 11 | 11 | 0 | 22,503.580 | 2,045.780 |
| `tests/prompts/test_ba_prompts.py` | 12 | 12 | 0 | 0.619 | 0.052 |
| `tests/prompts/test_citation_accuracy.py` | 16 | 16 | 0 | 0.601 | 0.038 |
| `tests/prompts/test_fm_prompts.py` | 1 | 1 | 0 | 0.108 | 0.108 |
| `tests/prompts/test_hallucination_guard.py` | 11 | 11 | 0 | 0.426 | 0.039 |
| `tests/prompts/test_insider_news_prompts.py` | 1 | 1 | 0 | 0.091 | 0.091 |
| `tests/prompts/test_macro_prompts.py` | 1 | 1 | 0 | 0.094 | 0.094 |
| `tests/prompts/test_orchestration_prompt_telemetry.py` | 1 | 1 | 0 | 0.134 | 0.134 |
| `tests/prompts/test_planner_prompts.py` | 20 | 20 | 0 | 0.980 | 0.049 |
| `tests/prompts/test_qf_prompts.py` | 1 | 1 | 0 | 0.119 | 0.119 |
| `tests/prompts/test_sr_prompts.py` | 1 | 1 | 0 | 0.207 | 0.207 |
| `tests/prompts/test_summarizer_prompts.py` | 12 | 12 | 0 | 0.589 | 0.049 |
| `tests/prompts/test_ws_prompts.py` | 1 | 1 | 0 | 0.220 | 0.220 |

## Infrastructure Quantified Details

### PostgreSQL
- `postgres_tables`:
  - connection time: **6.765 ms**
  - data amount: **6 tables**
  - data type: `list[str]`
  - snapshot size: **118 bytes**
- `postgres_feedback_tables`:
  - connection time: **5.977 ms**
  - data amount: **3 tables**
  - data type: `list[str]`
  - snapshot size: **51 bytes**

### Neo4j
- `neo4j_connection`:
  - connection time: **0.091 ms**
  - query latency: **5.527 ms**
  - data amount: **1 row**
  - data type: `int`
  - snapshot size: **1 byte**
- `neo4j_vector_index`:
  - connection time: **0.067 ms**
  - query latency: **3.098 ms**
  - data amount: **1 index**
  - data type: `list[str]`
  - snapshot size: **19 bytes**

### Ollama
- `ollama_embed`:
  - API latency: **311.495 ms**
  - data amount: **768 embedding cells**
  - data type: `embedding_matrix`
  - snapshot size: **8 bytes**
- `ollama_models`:
  - API latency: **2.495 ms**
  - data amount: **2 models**
  - data type: `list[str]`
  - snapshot size: **40 bytes**

### DeepSeek and Health
- `deepseek_api_not_configured`:
  - data type: `bool`
  - data amount: **0**
  - snapshot size: **5 bytes**
- `all_services_health`:
  - healthy services: **3/3**
  - data amount: **3 services**
  - data type: `dict`
  - snapshot size: **49 bytes**

## Prompt-Agent Quantified Details

### Business Analyst (BA)
- Covered by `tests/prompts/test_ba_prompts.py` (12 tests)
- Total BA prompt task time: **0.619 ms**

### Quant Fundamental (QF)
- `prompt_qf`:
  - latency: **0.004 ms**
  - data amount: **1**
  - data type: `dict`
  - output snapshot size: **150 bytes**

### Financial Modelling (FM)
- `prompt_fm`:
  - latency: **0.001 ms**
  - data amount: **1**
  - data type: `dict`
  - output snapshot size: **136 bytes**

### Web Search (WS)
- `prompt_ws`:
  - latency: **0.001 ms**
  - data amount: **1**
  - data type: `list[dict]`
  - output snapshot size: **184 bytes**

### Stock Research (SR)
- `prompt_sr`:
  - latency: **0.002 ms**
  - data amount: **2**
  - data type: `list[dict]`
  - output snapshot size: **238 bytes**

### Macro
- `prompt_macro`:
  - latency: **0.011 ms**
  - data amount: **3**
  - data type: `list[str]`
  - output snapshot size: **181 bytes**

### Insider News
- `prompt_insider_news`:
  - latency: **0.002 ms**
  - data amount: **4**
  - data type: `dict`
  - output snapshot size: **219 bytes**

### Orchestration Prompt (Whole System)
- `prompt_orchestration`:
  - latency: **0.006 ms** (mocked telemetry mode)
  - data amount: **7** output groups
  - data type: `orchestration_state`
  - output snapshot size: **455 bytes**
  - reported output payload bytes in metrics: **665 bytes**

## Slowest Individual Tests (Top 10)

1. `tests/integration/test_agent_db.py::test_insider_news_postgres_data` — **170,253.659 ms**
2. `tests/integration/test_agent_db.py::test_macro_pg_neo4j_data` — **131,464.429 ms**
3. `tests/integration/test_agent_db.py::test_sr_chunk_search` — **27,101.403 ms**
4. `tests/integration/test_agent_db.py::test_ws_perplexity_response` — **19,187.807 ms**
5. `tests/integration/test_rlaif_memory.py::test_track_multiple_agent_failures` — **10,779.800 ms**
6. `tests/integration/test_agent_db.py::test_ba_neo4j_retrieval` — **9,590.376 ms**
7. `tests/integration/test_rlaif_memory.py::test_episodic_memory_query` — **7,230.286 ms**
8. `tests/integration/test_agent_db.py::test_ba_full_pipeline` — **5,409.536 ms**
9. `tests/integration/test_rlaif_memory.py::test_episodic_hints_loaded` — **4,031.086 ms**
10. `tests/integration/test_infra.py::test_ollama_embed` — **312.276 ms**

## Active Files Kept in `tests/outputs`

- `pytest_all_with_metrics_20260323_171441.log`
- `test_metrics.jsonl`
- `test_metrics_summary.md`
- `testing_tasks_detailed_metrics_20260323.md`
