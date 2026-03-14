# Orchestration Layer

The orchestration layer coordinates the four specialised agents (Business Analyst, Quantitative Fundamental, Financial Modelling, Web Search) via a LangGraph `StateGraph`. It implements a **Global Plan-and-Execute / Local ReAct** architecture with **RLAIF feedback loop** for continuous improvement.

---

## Architecture

```
user_query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_planner  (deepseek-chat)                                      │
│  ├── classify query intent + complexity (1/2/3)                     │
│  ├── resolve ticker symbol(s) from natural-language input           │
│  ├── select which agents to invoke (run_* flags)                    │
│  └── run data_availability.check_all() — ping all backends once    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_parallel_agents  (ThreadPoolExecutor)                          │
│  ├── Business Analyst  ─── run_full_analysis(ticker)                │
│  ├── Quant Fundamental ─── run_full_analysis(ticker)                │
│  ├── Financial Modelling── run_full_analysis(ticker)                │
│  └── Web Search        ─── run_web_search_agent({...})            │
│                                                                      │
│  Wall-clock time = max(T_BA, T_QF, T_FM, T_WS)                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_react_check                                                   │
│  ├── if gaps (enabled agents with no output) AND iterations left    │
│  │       → loop back to node_parallel_agents (retry gap agents)     │
│  └── if no gaps OR iteration cap reached → proceed to summarizer   │
└──────────────────────────────┬──────────────────────────────────────┘
                ┌──────────────┴──────────────┐
                │ loop if gaps remain          │
                ▼                             ▼
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
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_rlaif_scorer  (deepseek-chat as judge)                       │
│  ├── scores report on 5 dimensions (factual accuracy, citations,    │
│  │   analysis depth, structure compliance, language quality)        │
│  ├── identifies which agent caused any issues                       │
│  └── stores scores in rl_feedback table for learning               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_translator                                                    │
│  └── translates to user's preferred language (if not English)       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  node_memory_update                                                 │
│  └── records query for semantic router + episodic memory           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                              END
                     final_summary + citations + rl_feedback_scores
```

---

## Files

| File | Description |
|---|---|
| `graph.py` | Builds and compiles the LangGraph `StateGraph`. Exposes `run()` and `stream()` as the public API. |
| `nodes.py` | All 9 LangGraph node functions: `node_planner`, `node_parallel_agents`, `node_business_analyst`, `node_quant_fundamental`, `node_financial_modelling`, `node_web_search`, `node_react_check`, `node_summarizer`, `node_rlaif_scorer`. |
| `state.py` | `OrchestrationState` TypedDict — the shared state schema flowing between all nodes. |
| `llm.py` | `plan_query()` (deepseek-chat) and `summarise_results()` (deepseek-r1:8b) plus the massive system prompts for both. |
| `citations.py` | `build_citation_block()` extracts all `qdrant::` tokens from agent outputs and builds a numbered `[N]` reference list. `inject_inline_numbers()` replaces tokens with `[N]` numbers in the final summary. |
| `data_availability.py` | `check_all()` concurrently pings Neo4j, Qdrant, PostgreSQL, and Ollama once per request and returns a readiness report. Used by `node_planner` to detect backend outages before dispatching agents. |
| `feedback.py` | RLAIF scoring using DeepSeek Chat API as judge. Stores scores in `rl_feedback`, `user_feedback`, and `prompt_versions` tables. |
| `episodic_memory.py` | Records query failures for semantic similarity lookup to pre-empt known failure patterns. |

---

## Graph Variants

### Default — Parallel Graph

```
planner → parallel_agents → react_check ──► (parallel_agents | summarizer) → rlaif_scorer → translator → memory_update → END
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
| `node_planner` | `deepseek-chat` (API) | Fast; reliable structured JSON routing output |
| `node_summarizer` | `deepseek-r1:8b` (local Ollama) | Deep analytical prose; handles 11-section research note at 8K+ tokens |
| `node_rlaif_scorer` | `deepseek-chat` (API) | Acts as judge to score report quality |

---

## RLAIF Feedback System

### Overview

After each query, the **RLAIF scorer** automatically evaluates the generated report using DeepSeek Chat as a judge. This enables **continuous improvement** of agent prompts based on actual output quality.

### Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Factual Accuracy | 30% | Do all numbers match the original agent outputs? |
| Citation Completeness | 20% | Does every claim have [N] citation? |
| Analysis Depth | 25% | Does it explain WHY numbers matter? |
| Structure Compliance | 15% | Are all 11 sections present in correct order? |
| Language Quality | 10% | Professional tone, no banned words |

### Database Tables

```sql
-- RLAIF feedback (AI-generated scores)
CREATE TABLE rl_feedback (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50),
    user_query TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    factual_accuracy FLOAT,
    citation_completeness FLOAT,
    analysis_depth FLOAT,
    structure_compliance FLOAT,
    language_quality FLOAT,
    overall_score FLOAT,
    strengths JSONB,
    weaknesses JSONB,
    specific_feedback TEXT,
    agent_blamed VARCHAR(50),  -- which agent caused issues
    report_excerpt TEXT,
    ticker VARCHAR(20)
);

-- User feedback (explicit ratings from UI)
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50),
    session_id VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW(),
    helpful BOOLEAN,
    comment TEXT,
    issue_tags JSONB,
    report_version VARCHAR(20)
);

-- Prompt versions (A/B testing)
CREATE TABLE prompt_versions (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50),
    version VARCHAR(20),
    prompt_text TEXT,
    deployed_at TIMESTAMP DEFAULT NOW(),
    deployed_to FLOAT DEFAULT 1.0,
    avg_score_before FLOAT,
    avg_score_after FLOAT,
    improvement_pct FLOAT,
    weaknesses_addressed JSONB
);
```

### Short-term Learning

The system implements daily analysis of low-scoring reports:

1. **Detect patterns** - Query `rl_feedback` for reports with `overall_score < 7.0`
2. **Identify root causes** - Group by `agent_blamed` and common `weaknesses`
3. **Update prompts** - Enhance agent prompts to address specific weaknesses
4. **A/B test** - Deploy new prompts to 20% of traffic, compare scores

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

    # RLAIF Feedback
    rl_feedback_scores: Optional[Dict[str, Any]]  # RLAIF scores from AI judge
    rl_feedback_run_id: Optional[str]  # Unique run ID for this analysis
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
result["ticker"]                    # str — "AAPL"
result["tickers"]                    # list[str] — ["AAPL"]
result["plan"]                      # dict — planner routing decision
result["business_analyst_output"]   # dict — BA agent JSON
result["quant_fundamental_output"]  # dict — QF agent JSON
result["financial_modelling_output"] # dict — FM agent JSON
result["web_search_output"]         # dict — WS agent JSON
result["agent_errors"]              # dict — {agent_name: error_msg}
result["react_steps"]               # list — ReAct trace
result["react_iteration"]           # int — number of passes used

# RLAIF Feedback (new)
result["rl_feedback_scores"]        # dict — AI judge scores
result["rl_feedback_run_id"]        # str — unique run ID
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
    elif node_name == "rlaif_scorer":
        print("RLAIF scores:", node_output.get("rl_feedback_scores"))
```

For a **complexity-1** query the UI receives events: `planner → parallel_agents → react_check → summarizer → rlaif_scorer → translator → memory_update`.
For **complexity-3** the events may repeat the parallel_agents loop.

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

# Access RLAIF scores after any query
print(result["rl_feedback_scores"]["overall_score"])  # e.g., 8.2
print(result["rl_feedback_scores"]["agent_blamed"])    # e.g., "quant_fundamental"
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
print("RLAIF Score:", result["rl_feedback_scores"]["overall_score"])
EOF

# Sequential debug mode
ORCHESTRATION_SEQUENTIAL=1 python - <<'EOF'
from orchestration.graph import run
result = run("AAPL P/E check")
print(result["ticker"], result["agent_errors"])
EOF
```

---

## Environment Variables

```bash
# LLM model selection
ORCHESTRATION_PLANNER_MODEL=deepseek-chat     # default
ORCHESTRATION_SUMMARIZER_MODEL=deepseek-r1:8b # default

# Timeouts (seconds; unset = no cap)
ORCHESTRATION_LLM_TIMEOUT=60                    # planner timeout
ORCHESTRATION_SUMMARIZER_TIMEOUT=1200           # summarizer timeout

# Graph mode
ORCHESTRATION_SEQUENTIAL=0                      # set to 1 for sequential debug

# Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434

# DeepSeek API (for RLAIF scoring)
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
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

## Feedback Functions

### From `orchestration.feedback`:

```python
from orchestration.feedback import (
    score_report_with_rlaif,      # Score a report using AI judge
    store_user_feedback,          # Store explicit user feedback from UI
    get_recent_rl_feedback,      # Get low-scoring reports for analysis
    get_user_feedback_summary,   # Get user feedback statistics
    get_agent_performance_summary, # Get RLAIF scores by agent
    check_low_score_alert,       # Check for runs needing review
)

# Score a report
scores = score_report_with_rlaif(
    run_id="abc123",
    user_query="Analyze AAPL",
    final_summary="The report...",
    agent_outputs={"quant_fundamental_output": {...}},
    ticker="AAPL"
)
# Returns: {overall_score: 8.2, factual_accuracy: 9.0, ...}

# Store user feedback
store_user_feedback(
    run_id="abc123",
    session_id="sess_456",
    helpful=True,
    comment="Great analysis!",
    issue_tags=["Analysis too shallow"]
)

# Get low-scoring reports for learning
low_scores = get_recent_rl_feedback(days=7, min_score=7.0)
```

---

*Last updated: 2026-03-14 | Author: hck717*
