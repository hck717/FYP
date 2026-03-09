# Agents Layer — Overview

The `agents/` directory contains four specialised AI agents. Each agent is an
independent LangGraph pipeline responsible for a distinct class of financial
analysis. The orchestration layer (`orchestration/`) selects which agents to
run, dispatches them in parallel, and synthesises their outputs into a final
research note.

---

## Architecture

```
User Query
    │
    ▼
[Orchestration Planner]  ─  resolves ticker(s), selects agents, sets complexity
    │
    ▼ (parallel — ThreadPoolExecutor)
┌───────┬────────────────┬───────────────────┬──────────────────────┐
│  Web  │    Business    │      Quant        │     Financial        │
│Search │    Analyst     │   Fundamental     │     Modelling        │
│ Agent │     Agent      │      Agent        │       Agent          │
└───────┴────────────────┴───────────────────┴──────────────────────┘
    │           │                 │                    │
    └───────────┴─────────────────┴────────────────────┘
                          │
                          ▼
              [ReAct Check]  ─  retry gap agents if needed
                          │
                          ▼
              [Summarizer]   ─  deepseek-r1:8b narrative
                          │
                          ▼
                  Final Research Note (Markdown)
```

---

## Agents at a Glance

| Agent | Directory | Primary Responsibility | Key Data Sources |
|---|---|---|---|
| **Web Search** | `web_search/` | Live breaking news, sentiment signals | Perplexity sonar-pro API |
| **Business Analyst** | `business_analyst/` | Qualitative moat analysis, earnings calls, broker reports | Neo4j, PostgreSQL, Ollama |
| **Quant Fundamental** | `quant_fundamental/` | Value/quality/momentum factors, Piotroski, Beneish | PostgreSQL |
| **Financial Modelling** | `financial_modelling/` | DCF valuation, peer comps, technicals | PostgreSQL, Neo4j |

---

## Shared Technology Stack

| Component | Role |
|---|---|
| **LangGraph** | Agent state machine / node-based workflow |
| **LangChain** | LLM invocation, prompt management |
| **Ollama** | Local LLM inference (`llama3.2:latest`, `deepseek-r1:8b`) |
| **Neo4j** | Company knowledge graph + vector search (Chunk nodes) |
| **PostgreSQL** | Historical time-series, fundamentals, sentiment trends |
| **pgvector** | Vector embeddings for semantic search (PostgreSQL) |

---

## Quick-Start Commands

Run any agent directly from the repo root (activate `.venv` first):

```bash
# Business Analyst (inside Docker - recommended)
docker exec fyp-airflow-webserver python -m agents.business_analyst.agent --ticker AAPL
docker exec fyp-airflow-webserver python -m agents.business_analyst.agent --ticker MSFT --task "What is Microsoft's competitive moat?"

# Business Analyst (from host)
source .venv/bin/activate
python -m agents.business_analyst.agent --ticker AAPL --task "What is Apple's competitive moat?"

# Quant Fundamental
python -m agents.quant_fundamental.agent --ticker NVDA
python -m agents.quant_fundamental.agent --prompt "Compare MSFT vs AAPL fundamentals"

# Financial Modelling
python -m agents.financial_modelling.agent --ticker TSLA
python -m agents.financial_modelling.agent --ticker GOOGL --log-level DEBUG
```

---

## Configuration

All agents use environment variables for configuration. Key settings:

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` (host) or `http://host.docker.internal:11434` (Docker) |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |

---

## Data Flow

1. **Ingestion** (Airflow DAG): EODHD API → PostgreSQL + Neo4j
2. **Text Processing**: PDF earnings calls + broker reports → Neo4j chunks
3. **Agent Execution**: Query → Retrieval → Analysis → Citation
4. **Synthesis**: All agent outputs → Final research note

---

## Environment

- **Python**: 3.11+
- **Ollama**: Must be running with models pulled
- **Docker**: docker-compose up -d (PostgreSQL, Neo4j, Airflow)
- **No timeout**: Agent requests default to no timeout for quality analysis
