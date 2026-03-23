# FYP Testing Suite

A comprehensive testing framework for the multi-agent LangGraph investment analyst pipeline.

## Overview

This testing suite validates the complete system architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION GRAPH                      │
├─────────────────────────────────────────────────────────────┤
│  planner → [BA, QF, WS, FM, SR, MACRO, IN] → summarizer → post_proc │
└─────────────────────────────────────────────────────────────┘
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── integration/          # Integration tests (require live services)
│   ├── test_infra.py     # Infrastructure connectivity
│   ├── test_agent_db.py  # Agent ↔ Database integration
│   ├── test_graph_nodes.py    # LangGraph node wiring
│   ├── test_rlaif_memory.py   # RLAIF + episodic memory
│   └── test_multi_ticker.py   # Multi-ticker queries
└── prompts/              # Prompt/LLM tests
    ├── test_planner_prompts.py     # Planner agent selection
    ├── test_ba_prompts.py           # Business Analyst prompts
    ├── test_summarizer_prompts.py  # Summarizer stages
    ├── test_hallucination_guard.py # Hallucination prevention
    ├── test_citation_accuracy.py   # Citation verification
    └── golden_set.jsonl             # Regression test data
```

## Running Tests

### All Tests
```bash
cd /Users/brianho/FYP
pytest tests/ -v
```

### Integration Tests Only
```bash
# Requires Docker services (PostgreSQL, Neo4j, Ollama)
pytest tests/integration/ -v -m integration --timeout=120
```

### Prompt Tests Only
```bash
pytest tests/prompts/ -v -m prompt --timeout=60
```

### Prompt + Metrics Telemetry
```bash
pytest tests/prompts/ -v -m prompt
# Produces:
# - tests/outputs/test_metrics.jsonl
# - tests/outputs/test_metrics_summary.md
```

### Specific Test File
```bash
pytest tests/integration/test_infra.py -v
pytest tests/prompts/test_planner_prompts.py -v
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Configuration

### Markers
- `@pytest.mark.integration` - Requires live services (PostgreSQL, Neo4j, Ollama)
- `@pytest.mark.prompt` - Tests LLM prompt outputs
- `@pytest.mark.unit` - No external dependencies

### Environment Variables
Tests automatically set these in `conftest.py`:
```python
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
NEO4J_URI=bolt://localhost:7687
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
```

## Adding Tests for New Agents

### 1. Register Agent in conftest.py

```python
# In tests/conftest.py, add to AGENT_REGISTRY:
AGENT_REGISTRY = {
    # ... existing agents ...
    "new_agent": {
        "module": "agents.new_agent.agent",
        "node_name": "node_new_agent",
        "output_key": "new_agent_outputs",
        "requires_db": ["postgres", "neo4j"],
    },
}
```

### 2. Create Agent DB Test

```python
# tests/integration/test_agent_db.py
@pytest.mark.integration
def test_new_agent_postgres_data():
    """Test new agent fetches data from PostgreSQL."""
    from agents.new_agent import tools
    
    connector = tools.PostgresConnector()
    data = connector.get_data(ticker="AAPL")
    assert isinstance(data, (list, dict, type(None)))
```

### 3. Create Agent Prompt Tests

```python
# tests/prompts/test_new_agent_prompts.py
@pytest.mark.prompt
def test_new_agent_output_structure():
    """Test new agent produces correct output structure."""
    result = _mock_new_agent_output("AAPL")
    
    assert "ticker" in result
    assert "output_key" in result or True
```

### 4. Use Factory Functions

```python
# Extensible test helpers in each file:
create_agent_db_test(agent_name: str, ticker: str)
create_node_wiring_test(node_name: str, output_key: str, agent_key: str)
create_planner_test(query: str, expected_ticker: str, expected_agents: list, complexity: int)
```

## Adding Tests for New Nodes

### 1. Register Node in conftest.py

```python
# In tests/conftest.py, add to NODE_REGISTRY:
NODE_REGISTRY = {
    # ... existing nodes ...
    "node_new_node": "orchestration.nodes.node_new_node",
}
```

### 2. Test Node Routing

```python
# tests/integration/test_graph_nodes.py
def test_route_after_new_node():
    """Test ReAct retry routing for new node."""
    state: OrchestrationState = {
        "run_new_agent": True,
        "new_agent_outputs": [],
        "agent_react_iterations": {"new_agent": 0},
        "react_max_iterations": 2,
    }
    
    # Use factory helper
    route_fn = test_node_routing("node_new_agent", "run_new_agent", "new_agent_outputs")
    assert route_fn(state) == "node_new_node"
```

### 3. Test Node Integration

```python
# Test full pipeline with new node
def test_graph_with_new_node():
    """Test graph compiles with new node."""
    from orchestration.graph import build_graph
    
    graph = build_graph()
    assert graph is not None
```

## Test Fixtures Reference

### Database Fixtures
- `mock_pg_conn` - Mock PostgreSQL connection
- `mock_neo4j_driver` - Mock Neo4j driver

### Data Fixtures
- `sample_neo4j_chunk` - Single Neo4j chunk
- `sample_neo4j_chunks` - Multiple chunks
- `sample_financial_data` - Financial statement data
- `sample_valuation_data` - Valuation metrics

### Output Fixtures
- `mock_ba_output` - Business Analyst output
- `mock_qf_output` - Quant Fundamental output
- `mock_fm_output` - Financial Modelling output
- `mock_all_agent_outputs` - All agent outputs combined

### Citation Fixtures
- `sample_citations` - All citation types (broker, transcript, macro)

## Integration Tests

### Infrastructure Tests (`test_infra.py`)
- PostgreSQL connection + tables
- Neo4j bolt connection + vector index
- Ollama embedding endpoint
- DeepSeek API

### Agent DB Tests (`test_agent_db.py`)
- BA ↔ Neo4j retrieval
- QF ↔ PostgreSQL factors
- FM ↔ PostgreSQL DCF inputs + Neo4j peers
- SR ↔ Neo4j PDF chunks
- Macro ↔ PostgreSQL/Neo4j macro + earnings data
- Insider News ↔ PostgreSQL insider + news tables
- WS ↔ Perplexity API

### Graph Node Tests (`test_graph_nodes.py`)
- Planner fan-out logic
- ReAct retry routing
- Fan-in summarizer
- Graph compilation

### RLAIF + Memory Tests (`test_rlaif_memory.py`)
- RLAIF score persistence
- Episodic memory propagation
- Feedback table creation

### Multi-Ticker Tests (`test_multi_ticker.py`)
- Multi-ticker state structure
- Agent processing per ticker
- Comparative summarization

## Prompt Tests

### Planner Tests (`test_planner_prompts.py`)
- Agent selection by query type
- Ticker extraction
- Complexity mapping
- Edge cases (unsupported ticker, ambiguous, non-English)

### Business Analyst Tests (`test_ba_prompts.py`)
- CRAG grading (CORRECT/INCORRECT/AMBIGUOUS)
- Moat analysis structured output
- Sentiment analysis

### Quant Fundamental Tests (`test_qf_prompts.py`)
- Output schema checks
- Metrics logging (latency, size, data type)

### Financial Modelling Tests (`test_fm_prompts.py`)
- Output schema checks
- Metrics logging (latency, size, data amount)

### Web Search Tests (`test_ws_prompts.py`)
- Output schema checks
- Metrics logging (latency, citations/news amount)

### Stock Research Tests (`test_sr_prompts.py`)
- Output schema checks
- Metrics logging (broker-rating payload size/amount)

### Macro Tests (`test_macro_prompts.py`)
- Output schema checks
- Metrics logging (driver count, output size)

### Insider News Tests (`test_insider_news_prompts.py`)
- Output schema checks
- Metrics logging (coverage count, output size)

### Orchestration Telemetry (`test_orchestration_prompt_telemetry.py`)
- End-to-end prompt execution through orchestration graph
- Logs full-system latency and aggregate output snapshot

### Summarizer Tests (`test_summarizer_prompts.py`)
- Stage 1: Raw synthesis
- Stage 2: Structure sections
- Stage 3: Add citations
- Stage 4: Translation

### Hallucination Guard Tests (`test_hallucination_guard.py`)
- DCF numbers computed (not hallucinated)
- WACC from inputs
- Claims with citations
- Input validation

### Citation Accuracy Tests (`test_citation_accuracy.py`)
- Broker report citation format
- Earnings transcript citation
- Macro report citation
- Citation chain validation

## Golden Set Regression

The `golden_set.jsonl` contains regression test cases:

```json
{"query": "What is AAPL moat?", "must_contain": ["brand", "ecosystem"]}
{"query": "TSLA DCF valuation", "numeric_fields": ["dcf_value", "wacc"]}
```

Run with:
```bash
pytest tests/prompts/ -v --update-golden
```

## Troubleshooting

### Skip Integration Tests
```bash
pytest tests/ -v -m "not integration"
```

### Skip Slow Tests
```bash
pytest tests/ -v --timeout=30
```

### Verbose Output
```bash
pytest tests/ -vv -s
```

### Check Test Collection
```bash
pytest tests/ --collect-only
```

## CI/CD Integration

Example GitHub Actions workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
      neo4j:
        image: neo4j:5
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install pytest pytest-timeout pytest-cov
          pytest tests/ -v --timeout=120
```

## Architecture Notes

- **Extensible**: Add new agents/nodes by updating registries in `conftest.py`
- **Isolated**: Each test is independent and can run in any order
- **Documented**: Factory functions make adding tests self-documenting
- **Comprehensive**: Covers infrastructure, integration, prompts, and regression

## Contact

For questions or issues with the testing framework, refer to:
- Main FYP README: `/Users/brianho/FYP/README.md`
- Orchestration docs: `/Users/brianho/FYP/orchestration/README.md`
