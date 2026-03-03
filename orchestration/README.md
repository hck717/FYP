# Orchestration Layer

The orchestration layer coordinates the four specialised agents (Business Analyst, Quantitative Fundamental, Financial Modelling, Web Search) via a LangGraph `StateGraph`. It implements a **Global Plan-and-Execute / Local ReAct** architecture.

---

## Architecture

```
user_query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_planner  (llama3.2:latest)                                    │
│  ├── classify query intent + complexity (1/2/3)                     │
│  ├── resolve ticker symbol(s) from natural-language input           │
│  ├── select which agents to invoke (run_* flags)                    │
│  └── run data_availability.check_all() — ping all backends once     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_parallel_agents  (ThreadPoolExecutor)                         │
│  ├── Business Analyst  ─── run_full_analysis(ticker)                │
│  ├── Quant Fundamental ─── run_full_analysis(ticker)                │
│  ├── Financial Modelling── run_full_analysis(ticker)                │
│  └── Web Search        ─── run_web_search_agent({...})              │
│                                                                     │
│  Wall-clock time = max(T_BA, T_QF, T_FM, T_WS)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_react_check                                                   │
│  ├── if gaps (enabled agents with no output) AND iterations left    │
│  │       → loop back to node_parallel_agents (retry gap agents)     │
│  └── if no gaps OR iteration cap reached → proceed to summarizer   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴────────────┐
                    │ loop if gaps remain   │
                    ▼                       │
           [parallel_agents]────────────────┘
                    │ all done
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_summarizer  (deepseek-r1:8b)                                  │
│  ├── receives all 4 agent outputs                                   │
│  ├── writes 11-section buy-side research note                       │
│  └── injects [N] inline citation numbers                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                              END
                    final_summary + citations
```

---

## Files

| File | Description |
|---|---|
| `graph.py` | Builds and compiles the LangGraph `StateGraph`. Exposes `run()` and `stream()` as the public API. |
| `nodes.py` | All 8 LangGraph node functions: `node_planner`, `node_parallel_agents`, `node_business_analyst`, `node_quant_fundamental`, `node_financial_modelling`, `node_web_search`, `node_react_check`, `node_summarizer`. |
| `state.py` | `OrchestrationState` TypedDict — the shared state schema flowing between all nodes. |
| `llm.py` | `plan_query()` (llama3.2:latest) and `summarise_results()` (deepseek-r1:8b) plus the massive system prompts for both. |
| `citations.py` | `build_citation_block()` extracts all `qdrant::` tokens from agent outputs and builds a numbered `[N]` reference list. `inject_inline_numbers()` replaces tokens with `[N]` numbers in the final summary. |
| `data_availability.py` | `check_all()` concurrently pings Neo4j, Qdrant, PostgreSQL, and Ollama once per request and returns a readiness report. Used by `node_planner` to detect backend outages before dispatching agents. |

---

## Graph Variants

### Default — Parallel Graph

```
planner → parallel_agents → react_check ──► (parallel_agents | summarizer) → END
```

All enabled agents run **concurrently** in a `ThreadPoolExecutor` inside `node_parallel_agents`. This is the production default.

### Legacy Sequential Graph (debug)

```
planner → react_dispatch → [BA | QF | WS | FM] (one at a time) → react_check → (react_dispatch | summarizer) → END
```

Enable with:
```bash
ORCHESTRATION_SEQUENTIAL=1
```

Useful for debugging individual agent failures without parallelism noise.

---

## LLM Models

| Node | Model | Why |
|---|---|---|
| `node_planner` | `llama3.2:latest` (local Ollama) | Fast (~3s); reliable structured JSON routing output |
| `node_summarizer` | `deepseek-r1:8b` (local Ollama) | Deep analytical prose; handles 11-section research note at 8K+ tokens |

Both models run locally via Ollama — no cloud calls for orchestration.

---

## State Schema (`state.py`)

```python
class OrchestrationState(TypedDict, total=False):
    # Input
    user_query: str                    # original user message
    session_id: str                    # optional session identifier

    # Planner output
    plan: Optional[Dict[str, Any]]     # structured routing plan
    ticker: Optional[str]              # primary ticker (legacy compat)
    tickers: List[str]                 # all resolved tickers (multi-ticker support)

    # Agent dispatch flags (set by planner)
    run_business_analyst: bool
    run_quant_fundamental: bool
    run_web_search: bool
    run_financial_modelling: bool

    # ReAct loop control
    react_steps: List[Dict]            # [{tool, input, observation}, ...]
    react_iteration: int               # current pass index (0-based)
    react_max_iterations: int          # complexity 1→1, 2→2, 3→3

    # Agent outputs (multi-ticker: list of results per ticker)
    business_analyst_outputs: List[Dict]
    quant_fundamental_outputs: List[Dict]
    web_search_outputs: List[Dict]
    financial_modelling_outputs: List[Dict]

    # Legacy single-output aliases → outputs[0]
    business_analyst_output: Optional[Dict]
    quant_fundamental_output: Optional[Dict]
    web_search_output: Optional[Dict]
    financial_modelling_output: Optional[Dict]

    # Data availability (from data_availability.check_all())
    data_availability: Optional[Dict]

    # Errors
    agent_errors: Dict[str, str]       # {agent_name: error_message}

    # Final output
    final_summary: Optional[str]       # DeepSeek research note
    output: Optional[Dict]             # full structured response
```

---

## ReAct Loop Behaviour

The `node_react_check` + `_should_loop` conditional edge implement the ReAct pattern:

| Complexity | `react_max_iterations` | Behaviour |
|---|---|---|
| 1 (simple) | 1 | Single pass — no retry regardless of gaps |
| 2 (moderate) | 2 | One retry if any enabled agent failed |
| 3 (full report) | 3 | Up to two retries on gaps or errors |

**Key rule:** On each retry pass, only agents with **no output yet** or an **error** are re-run. Agents that already produced a successful result are never re-executed.

---

## Public API

### `run(user_query, session_id)` — blocking

```python
from orchestration.graph import run

result = run("What is Apple's competitive moat and current valuation?")

# Available keys in result dict:
result["final_summary"]               # str — DeepSeek research note
result["ticker"]                      # str — "AAPL"
result["tickers"]                     # list[str] — ["AAPL"]
result["plan"]                        # dict — planner routing decision
result["business_analyst_output"]     # dict — BA agent JSON
result["quant_fundamental_output"]    # dict — QF agent JSON
result["financial_modelling_output"]  # dict — FM agent JSON
result["web_search_output"]           # dict — WS agent JSON
result["agent_errors"]                # dict — {agent_name: error_msg}
result["react_steps"]                 # list — ReAct trace
result["react_iteration"]             # int — number of passes used
```

### `stream(user_query, session_id)` — streaming

```python
from orchestration.graph import stream

for node_name, node_output in stream("Compare MSFT vs AAPL"):
    if node_name == "planner":
        print("Routing plan:", node_output.get("plan"))
        print("Tickers:", node_output.get("tickers"))
    elif node_name == "parallel_agents":
        print("Agents pass completed")
    elif node_name == "react_check":
        print("ReAct check:", node_output.get("react_iteration"))
    elif node_name == "summarizer":
        print("Summary:", node_output.get("final_summary", "")[:200])
```

For a **complexity-1** query the UI receives exactly 4 events: `planner → parallel_agents → react_check → summarizer`.
For **complexity-3** the events may repeat: `planner → parallel_agents → react_check → parallel_agents → react_check → summarizer`.

---

## Example Queries

```python
from orchestration.graph import run

# 1. Single ticker — full analysis (complexity 3)
result = run("Full fundamental analysis of NVDA including DCF and competitive moat")

# 2. Simple metric look-up (complexity 1)
result = run("What is Tesla's current P/E ratio?")

# 3. Multi-ticker comparison (complexity 3)
result = run("Compare Apple vs Microsoft cloud strategy and valuation")

# 4. Risk-focused (complexity 2)
result = run("What are the main risks for Google stock right now?")

# 5. Earnings analysis (complexity 2)
result = run("How has AAPL performed vs analyst earnings estimates?")
```

---

## Running Directly

```bash
# Single query from command line
source .venv/bin/activate
python - <<'EOF'
from orchestration.graph import run
result = run("What is Apple's competitive moat?")
print(result["final_summary"])
EOF

# Sequential debug mode
ORCHESTRATION_SEQUENTIAL=1 python - <<'EOF'
from orchestration.graph import run
result = run("AAPL P/E check")
print(result["ticker"], result["agent_errors"])
EOF

# Override LLM models
ORCHESTRATION_PLANNER_MODEL=llama3.2:latest \
ORCHESTRATION_SUMMARIZER_MODEL=deepseek-r1:8b \
python - <<'EOF'
from orchestration.graph import run
print(run("NVDA quick overview")["final_summary"][:500])
EOF
```

---

## Environment Variables

```bash
# LLM model selection
ORCHESTRATION_PLANNER_MODEL=llama3.2:latest     # default
ORCHESTRATION_SUMMARIZER_MODEL=deepseek-r1:8b   # default

# Timeouts (seconds; unset = no cap)
ORCHESTRATION_LLM_TIMEOUT=60                    # planner timeout
ORCHESTRATION_SUMMARIZER_TIMEOUT=1200           # summarizer timeout

# Graph mode
ORCHESTRATION_SEQUENTIAL=0                       # set to 1 for sequential debug

# Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Citation System

Every factual claim in the final summary is cited. The pipeline works as follows:

1. **Agents** produce outputs with inline `[qdrant::TICKER::title-slug]` tokens.
2. **`build_citation_block()`** scans all 4 agent outputs, collects unique tokens, and assigns sequential `[1]`, `[2]`, ... numbers.
3. **`inject_inline_numbers()`** replaces every token in the final summary with its `[N]` number.
4. A **References** block is appended to the bottom of the summary.

---

## Data Availability Check

Before dispatching agents, `node_planner` calls `data_availability.check_all()` which concurrently pings:

| Backend | What is checked |
|---|---|
| PostgreSQL | Connection + row count in `raw_fundamentals` |
| Qdrant | Collection status + vector count |
| Neo4j | Connection + `:Company` node count |
| Ollama | `/api/tags` endpoint — model list |

If a backend is unavailable, the planner can skip the dependent agent and log the gap in `agent_errors` rather than failing the entire pipeline.

---

*Last updated: 2026-03-03 | Author: hck717*
