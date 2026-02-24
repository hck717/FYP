# 🌐 Web Search Agent — "The News Desk"

> **Role:** Agent 7 of 7 | Real-Time Intelligence  
> **Status:** ✅ Production-ready | Supervisor-callable | Structured JSON output | 6/6 tests passing  
> **Primary Model:** `sonar-pro` via Perplexity API  
> **Fallback:** DuckDuckGo (no API key required)  
> **Last tested:** 2026-02-24 — live output verified  

---

## 🎯 Agent Responsibility

The Web Search Agent surfaces **"unknown unknowns"** — breaking developments that are NOT yet
captured in the system's local static knowledge base (PostgreSQL / Qdrant / Neo4j).

It is the **only agent with real-time web access**. Every call is stateless and returns one structured JSON object consumed directly by the Supervisor.

| ✅ In Scope | ❌ Out of Scope (handled by other agents) |
|---|---|
| Breaking earnings announcements | Historical price/OHLCV data → Financial Modelling Agent |
| Regulatory investigations / fines / sanctions | Fundamental ratios / DCF → Quant Agent |
| CEO / CFO / Board leadership changes | GDP, CPI, yield curve → Macro Agent |
| M&A announcements, product launches, recalls | 10-K / 10-Q deep-read → Business Analyst Agent |
| Litigation, patent disputes, ESG events | Analyst consensus / estimates → Consensus Agent |
| Events not yet in EODHD / FMP DB | Insider trades / 13F → Insider & Sentiment Agent |
| Competitor earnings surprises (indirect signals) | |

---

## 🧠 Prompt Engineering Stack

| Technique | Implementation | Benefit |
|---|---|---|
| **Step-Back Prompting** | Model reasons about macro/sector/competitor context before answering | Surfaces indirect risks missed by direct queries |
| **HyDE** | Hypothetical ideal article constructed in `agent.py` to prime semantic direction | +15–25% search precision vs. raw query |
| **Freshness Reranking** | `search_recency_filter` param passed to Perplexity; > 30 days labeled `[HISTORICAL]` | Ensures real-time relevance |
| **Hallucination Guard** | Every factual claim requires URL + date; single-source flagged `⚠️ UNCONFIRMED` | Enables Critic Agent NLI verification |
| **Schema Enforcement** | JSON schema injected into every user message | Guarantees Supervisor-parseable output |

---

## 📁 Directory Structure

```
agents/web_search/
├── README.md           # This file — Supervisor and developer reference
├── examples.md         # Real query examples with expected output patterns
├── agent.py            # LangGraph node entrypoint + runtime logic
├── tools.py            # Perplexity API client + JSON extractor
├── prompts.py          # SYSTEM_PROMPT, QUERY_GENERATION_PROMPT, HYDE_PROMPT
├── __init__.py
└── tests/
    ├── __init__.py
    └── test_agent.py   # 6 unit tests — all mocked, no API key needed
```

---

## ⚙️ Environment Variables

Add to `.env` at repo root (`/FYP/.env`):

```bash
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx
WEB_SEARCH_MODEL=sonar-pro
WEB_SEARCH_RECENCY_FILTER=week
```

> `tools.py` loads `.env` automatically via `python-dotenv`. No manual `export` needed.
> The Perplexity API URL is hardcoded — do NOT add `PERPLEXITY_API_URL` to `.env`.

### Available Models (Perplexity)

| Model | Speed | Best For |
|---|---|---|
| `sonar-pro` | Fast | **Default. Production use.** Real-time search + strong reasoning |
| `sonar` | Fastest | Quick news triage, lightweight queries |
| `sonar-reasoning-pro` | Slow | Deep multi-step analysis when latency is acceptable |

---

## ✅ Output Schema

The agent always returns this exact `WebSearchOutput` schema.
Supervisor reads `state["web_search_output"]` after parallel execution.

```json
{
  "agent": "web_search",
  "ticker": "NVDA",
  "query_date": "2026-02-24",
  "breaking_news": [
    {
      "title": "NVDA faces new DOJ antitrust probe into AI chip dominance",
      "url": "https://reuters.com/...",
      "published_date": "2026-02-23",
      "source_tier": 2,
      "relevance_score": 0.94,
      "verified": true
    }
  ],
  "sentiment_signal": "BEARISH",
  "sentiment_rationale": "Reuters (2026-02-23): DOJ opened Section 2 Sherman Act probe into NVDA's CUDA ecosystem lock-in.",
  "unknown_risk_flags": [
    {
      "risk": "DOJ antitrust probe not yet reflected in sell-side price targets or consensus estimates",
      "source_url": "https://reuters.com/...",
      "severity": "HIGH"
    }
  ],
  "competitor_signals": [
    {
      "company": "AMD",
      "signal": "AMD gaining datacenter share; CEO cited 'no regulatory exposure' — indirect positive vs NVDA",
      "source_url": "https://wsj.com/..."
    }
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "Quant agent flags NVDA as undervalued; this agent surfaces unpriced DOJ risk. Synthesizer must arbitrate.",
    "conflict_with_agent": "quant_fundamental"
  },
  "fallback_triggered": false,
  "confidence": 0.88,
  "raw_citations": ["https://reuters.com/...", "https://wsj.com/..."],
  "error": null
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `breaking_news` | `list[dict]` | Top news items ranked by relevance. May be empty if no material news. |
| `source_tier` | `int` 1/2/3 | 1=SEC/Regulator, 2=Reuters/Bloomberg/FT/WSJ, 3=Corporate IR |
| `sentiment_signal` | `str` | `BULLISH \| BEARISH \| NEUTRAL \| MIXED` |
| `unknown_risk_flags` | `list[dict]` | Risks NOT in static DB — primary Synthesizer conflict input |
| `competitor_signals` | `list[dict]` | Peer/supplier news that indirectly affects the target |
| `supervisor_escalation.action` | `str` | `CONFLICT_SIGNAL \| CONFIRMATORY_SIGNAL \| STANDALONE` |
| `fallback_triggered` | `bool` | `true` if Perplexity failed; output is degraded |
| `confidence` | `float` | 0.0–1.0. Supervisor must discard if `< 0.4` |
| `error` | `str \| null` | `null` = success. Non-null = check `fallback_triggered` |

---

## 🔌 Supervisor / LangGraph Integration

```python
# agents/supervisor/graph.py
from agents.web_search.agent import web_search_node

# Register as parallel execution node
graph.add_node("web_search", web_search_node)

# Required input keys the Supervisor must set before invoking:
# state["query"]             — str: original user question
# state["ticker"]            — str | None: resolved ticker symbol (e.g. "NVDA")
# state["recency_filter"]    — str | None: "day"|"week"|"month" (default: "week")
# state["web_search_model"]  — str | None: override model (default: "sonar-pro")

# Output keys written to shared state:
state["web_search_output"]["breaking_news"]           # list — feed to Synthesizer
state["web_search_output"]["unknown_risk_flags"]       # list — key conflict input
state["web_search_output"]["competitor_signals"]       # list — sector context
state["web_search_output"]["sentiment_signal"]         # str  — BULLISH/BEARISH/etc
state["web_search_output"]["supervisor_escalation"]    # dict — CONFLICT or CONFIRM
state["web_search_output"]["confidence"]               # float — discard if < 0.4
state["web_search_output"]["error"]                    # ALWAYS check before using
```

### Supervisor Decision Rules

```python
ws = state["web_search_output"]

# 1. Always check error first
if ws["error"] is not None or ws["confidence"] < 0.4:
    # Treat as INSUFFICIENT_DATA — do not use as conflict signal
    pass

# 2. Escalate to Synthesizer if conflict detected
if ws["supervisor_escalation"]["action"] == "CONFLICT_SIGNAL":
    # Pass to Synthesizer with conflicting agent name
    conflicting_agent = ws["supervisor_escalation"]["conflict_with_agent"]

# 3. Use unknown_risk_flags as Synthesizer input regardless
risk_flags = ws["unknown_risk_flags"]  # Always pass these to Synthesizer
```

---

## 🧪 Testing

```bash
# 1. Mocked unit tests (no API key needed)
pytest agents/web_search/tests/test_agent.py -v
# Expected: 6 passed in ~0.02s

# 2. Verify .env loads correctly
python - <<'EOF'
from agents.web_search.tools import PERPLEXITY_API_KEY, PERPLEXITY_API_URL, DEFAULT_MODEL
print("KEY loaded:", bool(PERPLEXITY_API_KEY))
print("URL:      ", PERPLEXITY_API_URL)
print("MODEL:    ", DEFAULT_MODEL)
EOF

# 3. Live smoke test
python - <<'EOF'
from agents.web_search.agent import run_web_search_agent
import json
result = run_web_search_agent({
    "query": "NVDA latest regulatory risk Q1 2026",
    "ticker": "NVDA",
    "recency_filter": "week",
    "model": "sonar-pro"
})
print(json.dumps(result, indent=2))
# Healthy: "error": null, "fallback_triggered": false, "confidence" > 0.5
EOF
```

---

## 🛡️ Error Handling

| Scenario | `fallback_triggered` | `confidence` | `error` | Supervisor Action |
|---|---|---|---|---|
| ✅ Success | `false` | 0.7–1.0 | `null` | Use output normally |
| Perplexity timeout / 5xx | `true` | 0.2 | HTTP error string | Discard, mark INSUFFICIENT_DATA |
| Model returns invalid JSON | `true` | 0.3 | `"JSON parse failure"` | Discard, mark INSUFFICIENT_DATA |
| Missing `PERPLEXITY_API_KEY` | — | — | `EnvironmentError` raised | Fix `.env` immediately |

> **Rule:** Supervisor must check `error` field before consuming any output field.
> If `confidence < 0.4`, do NOT use as a conflict signal — treat as no data.

---

## 📚 See Also

- [`examples.md`](./examples.md) — Real query examples with expected output patterns and source quality notes
- [`prompts.py`](./prompts.py) — Full system prompt, Step-Back and HyDE templates
- [`tools.py`](./tools.py) — Perplexity API client implementation
- Main README: [`/README.md`](../../README.md) — Full 9-agent system architecture

---

*Last updated: 2026-02-24 | Author: hck717 | Status: ✅ Live-tested*
