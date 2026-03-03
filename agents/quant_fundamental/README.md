# Quantitative Fundamental Agent

> **Status:** Complete (live-tested, all 5 tickers validated with local docker-compose databases)
> **Nickname:** The "Math Auditor"
> **Role in system:** Detects financial anomalies and computes fundamental factors with mathematical accuracy

---

## Role

The Quantitative Fundamental Agent is the **numerical integrity layer** of the multi-agent equity research system. It reads structured financial data directly from PostgreSQL, computes every factor deterministically in Python, and returns an audited factor report to the Supervisor.

It does **NOT** use RAG, DuckDB, or cloud databases. All data comes from local Docker PostgreSQL. The LLM is used only to write the `quantitative_summary` narrative string from the pre-computed numeric outputs — it never performs arithmetic.

**Handled by this agent:**
- Fundamental factor computation: Value, Quality, Momentum/Risk
- Financial anomaly detection via Z-score outlier flagging
- PostgreSQL data quality validation (field presence + range checks)
- Piotroski F-Score and Beneish M-Score calculation
- OLAP-style Chain-of-Table reasoning over structured financial data
- Rolling Sharpe Ratio and 12-month price return

**Handled by other agents (do not overlap):**
- DCF valuation and peer Comps → Financial Modelling Agent
- Macro rate environment and FX → (Phase 3 — not yet implemented)
- Qualitative strategy and filings → Business Analyst Agent
- Real-time news → Web Search Agent

---

## Architecture: 8-Node LangGraph Pipeline (Non-RAG)

This agent does **not** use RAG. All data comes from structured PostgreSQL tables. The pipeline is a deterministic OLAP computation chain — no retrieval confidence scoring, no fallback loops.

```
ticker (or natural-language prompt)
    │
    ▼
fetch_financials          ←  PostgreSQL: raw_fundamentals
                               (ratios_ttm, key_metrics_ttm, financial_scores,
                                earnings_history, analyst_estimates_eodhd)
    │
    ▼
chain_of_table_reasoning  ←  Step 1: SELECT — identify populated tables
                               Step 2: FILTER — extract key numeric fields
                               Step 3: CALCULATE — derived ratios (gross margin, ROA, FCF)
                               Step 4: RANK — classify quality tier (HIGH / MEDIUM / LOW)
                               Step 5: IDENTIFY — flag data issues
    │
    ▼
data_quality_check        ←  Validate field presence and range from PostgreSQL data
                               (e.g. 0 < gross_margin < 1, pe_trailing > 0)
    │
    ▼
calculate_value_factors   ←  P/E, EV/EBITDA, P/FCF, EV/Revenue
    │
    ▼
calculate_quality_factors ←  ROE, ROIC, Piotroski F-Score (9-point), Beneish M-Score
    │
    ▼
calculate_momentum_risk   ←  Sharpe Ratio (12-month), 12-Month Price Return, Beta (60-day)
    │
    ▼
flag_anomalies            ←  Z-score > 2 on gross margin, EBIT margin, ROE → anomaly_flags[]
    │
    ▼
format_json_output        ←  Assemble output; LLM writes quantitative_summary only
    │
   END → return to Supervisor / Planner
```

---

## Chain-of-Table Reasoning

The agent applies structured OLAP reasoning in five sequential steps before any metric is computed:

| Step | Operation | Example |
|---|---|---|
| 1 | **SELECT** available tables | `ratios_ttm`, `key_metrics_ttm`, `financial_scores`, `price_history` |
| 2 | **FILTER** key numeric fields | `revenue`, `gross_profit`, `operating_income`, `net_income`, `total_assets` |
| 3 | **CALCULATE** derived ratios | `gross_margin = gross_profit / revenue` |
| 4 | **RANK** quality tier | `HIGH` if ROE > 20% and EBIT margin > 15% |
| 5 | **IDENTIFY** data issues | Missing income statement, limited price history |

This chain ensures the LLM summary prompt always receives a clean, pre-processed table — never raw JSONB payloads it could misinterpret.

---

## Data Quality Check

Replaces the former DuckDB dual-path verifier. Validates the data already read from PostgreSQL:

- Critical TTM fields are present (`priceToEarningsRatioTTM`, `grossProfitMarginTTM`, etc.)
- Values are in expected numeric ranges (e.g. `0 < gross_margin < 1`, `pe_trailing > 0`)
- Returns `QualityStatus.PASSED`, `ISSUES_FOUND`, or `SKIPPED`

No re-computation is performed — this is a presence and range check only.

---

## Factor Analysis

| Category | Metrics |
|---|---|
| **Value** | P/E (trailing), EV/EBITDA, P/FCF, EV/Revenue |
| **Quality** | ROE, ROIC, Piotroski F-Score (0–9), Beneish M-Score |
| **Momentum / Risk** | Beta (60-day rolling), Sharpe Ratio (12-month), 12-Month Price Return |
| **Key Metrics** | Gross Margin, EBIT Margin, FCF Conversion, DSO Days, Current Ratio, D/E |

### Piotroski F-Score (9-point)

| Signal | Criterion | Points |
|---|---|---|
| ROA > 0 | Positive net income relative to assets | 1 |
| Operating Cash Flow > 0 | Positive OCF | 1 |
| ROA improving YoY | ΔROA > 0 | 1 |
| Accruals | OCF / Total Assets > ROA (cash quality) | 1 |
| Leverage decreasing | Long-term debt ratio falling YoY | 1 |
| Liquidity improving | Current ratio rising YoY | 1 |
| No dilution | Shares outstanding not increasing | 1 |
| Gross margin improving | ΔGross Margin > 0 | 1 |
| Asset turnover improving | ΔAsset Turnover > 0 | 1 |

Score ≥ 7 → Strong. Score ≤ 2 → Weak / potential short.

### Beneish M-Score (8-variable)

Flags potential earnings manipulation. M-Score > −2.22 → `manipulation_risk: HIGH`.

Variables: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA. Requires prior-year income statement for YoY comparison — will be `null` if only one period is available.

### Z-Score Anomaly Detection

```python
# Flag any metric more than 2 standard deviations from its rolling mean
for metric in ["gross_margin", "ebit_margin", "roe"]:
    z = (current_value - rolling_mean) / rolling_std
    if abs(z) > 2:
        anomaly_flags.append({
            "metric": metric,
            "z_score": round(z, 2),
            "current_value": current_value,
            "rolling_mean": rolling_mean,
            "direction": "above" if z > 0 else "below",
        })
```

---

## Infrastructure

| Service | Container | Used by this agent | Notes |
|---|---|---|---|
| PostgreSQL | `fyp-postgres` | Yes — primary data source | `raw_fundamentals`, `raw_timeseries`, `market_eod_us` |
| Ollama | local | Yes — narrative summary only | `llama3.2:latest` default; `deepseek-r1:8b` supported |
| Qdrant | `fyp-qdrant` | No | Not used by this agent |
| Neo4j | `fyp-neo4j` | No | Not used by this agent |
| DuckDB | — | **Removed** | Replaced by PostgreSQL-based data quality check |

---

## Data Sources

All data is ingested into **PostgreSQL** via Airflow DAGs. This agent reads from three tables:

| Table | `data_name` values used | Purpose |
|---|---|---|
| `raw_fundamentals` | `ratios_ttm`, `key_metrics_ttm`, `financial_scores`, `earnings_history`, `analyst_estimates_eodhd` | Factor computation |
| `raw_timeseries` | `historical_prices_eod` | Momentum/risk factors (Sharpe, return, beta) |
| `market_eod_us` | EOD prices | Benchmark for beta (populated when benchmark data available) |

### PostgreSQL Schema

```
raw_fundamentals:  id, agent_name, ticker_symbol, data_name, as_of_date, payload JSONB, source, ingested_at
raw_timeseries:    id, agent_name, ticker_symbol, data_name, ts_date,    payload JSONB, source, ingested_at
market_eod_us:     id, ts_date, payload JSONB, source, ingested_at
```

### Key Payload Field Names (FMP)

| `data_name` | Key fields used |
|---|---|
| `ratios_ttm` | `priceToEarningsRatioTTM`, `priceToFreeCashFlowRatioTTM`, `debtToEquityRatioTTM`, `grossProfitMarginTTM`, `operatingProfitMarginTTM`, `currentRatioTTM` |
| `key_metrics_ttm` | `evToEBITDATTM`, `evToFreeCashFlowTTM`, `evToSalesTTM`, `returnOnEquityTTM`, `returnOnInvestedCapitalTTM`, `daysOfSalesOutstandingTTM`, `enterpriseValueTTM` |
| `financial_scores` | `piotroskiScore`, `altmanZScore`, `ebit`, `revenue`, `marketCap`, `totalAssets` |
| `historical_prices_eod` | `close`, `adjusted_close`, `open`, `high`, `low`, `volume` |

---

## LLM Configuration

```python
# Primary LLM — used ONLY for quantitative_summary narrative
llm_model     = "llama3.2:latest"   # env: LLM_MODEL_QUANTITATIVE (default)
                                     # also supported: "deepseek-r1:8b"
temperature   = 0.1                  # env: QUANT_LLM_TEMPERATURE
max_tokens    = 512                  # env: QUANT_LLM_MAX_TOKENS
request_timeout = 120                # env: QUANT_REQUEST_TIMEOUT (seconds)
think         = False                # suppresses <think> chain-of-thought (deepseek-r1)
```

**LLM scope is strictly limited:** The LLM receives the fully computed numeric factor table and writes the `quantitative_summary` string only. It never performs arithmetic. All `<think>...</think>` tags are stripped from the response before it is included in the output.

---

## Output Schema (JSON)

All numeric fields are Python-computed from PostgreSQL — never LLM-generated.

```json
{
  "agent": "quant_fundamental",
  "ticker": "AAPL",
  "as_of_date": "2026-02-28",
  "time_range": "TTM",
  "value_factors": {
    "pe_trailing": 33.08,
    "ev_ebitda": 25.6,
    "p_fcf": 31.85,
    "ev_revenue": 9.02
  },
  "quality_factors": {
    "roe": 1.5994,
    "roic": 0.5101,
    "piotroski_f_score": 9,
    "beneish_m_score": null,
    "manipulation_risk": null
  },
  "momentum_risk": {
    "beta_60d": null,
    "sharpe_ratio_12m": 0.6906,
    "return_12m_pct": 2.38
  },
  "key_metrics": {
    "gross_margin": 0.4733,
    "ebit_margin": 0.3238,
    "fcf_conversion": null,
    "dso_days": 58.92,
    "current_ratio": 0.9737,
    "debt_to_equity": 1.0263
  },
  "anomaly_flags": [],
  "data_quality": {
    "status": "PASSED",
    "checks_passed": 9,
    "checks_total": 9,
    "issues": []
  },
  "quantitative_summary": "AAPL has a strong quality signal with a Piotroski F-Score of 9 ...",
  "data_sources": {
    "fundamentals": "postgresql:raw_fundamentals",
    "price_history": "postgresql:raw_timeseries",
    "benchmark": "postgresql:market_eod_us",
    "llm_scope": "quantitative_summary narrative only",
    "llm_model": "llama3.2:latest"
  }
}
```

---

## File Structure

```
agents/quant_fundamental/
├── README.md              # This file
├── __init__.py            # Package init — exports run() and run_full_analysis()
├── agent.py               # LangGraph pipeline (8 nodes), extract_ticker_from_prompt(),
│                          #   run(), run_full_analysis(), CLI entrypoint
├── config.py              # Centralised env-var configuration (slots=True dataclass)
├── tools.py               # PostgreSQL query helpers, DataQualityChecker, AnomalyDetector
├── prompts.py             # SYSTEM_PROMPT, PLANNER_PREAMBLE, build_system_prompt()
├── llm.py                 # QuantLLMClient — Ollama HTTP, think-tag stripping, timeout
├── schema.py              # Dataclasses: ValueFactors, QualityFactors, MomentumRiskFactors,
│                          #   KeyMetrics, AnomalyFlag, DataQualityCheck, QualityStatus
├── health.py              # Health-check endpoint (PostgreSQL ping)
├── factors/
│   ├── __init__.py
│   ├── value.py           # P/E, EV/EBITDA, P/FCF, EV/Revenue
│   ├── quality.py         # ROE, ROIC, Piotroski F-Score, Beneish M-Score
│   └── momentum_risk.py   # Beta (rolling), Sharpe Ratio, 12-month return
└── tests/
    └── test_agent.py      # 44 tests: 37 unit + 7 integration
```

---

## Public API

```python
from agents.quant_fundamental.agent import run, run_full_analysis

# Direct ticker
result = run(ticker="AAPL")
result = run_full_analysis(ticker="AAPL")

# Natural-language prompt (planner-agent compatible)
result = run(prompt="Analyze AAPL fundamentals")
result = run_full_analysis(prompt="Run quant fundamental analysis for MSFT")

# Access fields
result["value_factors"]["pe_trailing"]          # → 33.08
result["quality_factors"]["piotroski_f_score"]  # → 9
result["data_quality"]["status"]                # → "PASSED"
result["quantitative_summary"]                  # → "3-5 sentence narrative"
```

Both `run()` and `run_full_analysis()` execute the identical 8-node pipeline. `run_full_analysis()` exists so the Supervisor can call all agents via a uniform `run_full_analysis(ticker=...)` interface.

---

## CLI Usage

Two input modes. `--ticker` and `--prompt` are mutually exclusive.

```bash
# Mode 1: Direct ticker symbol
.venv/bin/python -m agents.quant_fundamental.agent --ticker AAPL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker MSFT --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker GOOGL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker TSLA --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker NVDA --log-level WARNING

# Mode 2: Natural-language prompt (planner-agent compatible)
# Ticker is extracted automatically — supports 4 extraction strategies:
#   1. Explicit "ticker XXXX" keyword
#   2. Parenthesised "(AAPL)" or "[TSLA]"
#   3. Known-ticker set match (150+ symbols)
#   4. Heuristic uppercase word with English stoplist
.venv/bin/python -m agents.quant_fundamental.agent --prompt "Analyze AAPL fundamentals" --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --prompt "Run quant fundamental analysis for MSFT" --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --prompt "What are the fundamentals for ticker GOOGL?" --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --prompt "TSLA analysis please" --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --prompt "Please run the quantitative fundamental report for NVDA stock" --log-level WARNING

# Use a different LLM model
LLM_MODEL_QUANTITATIVE=deepseek-r1:8b \
  .venv/bin/python -m agents.quant_fundamental.agent --ticker AAPL --log-level WARNING

# Run test suite
.venv/bin/python -m pytest agents/quant_fundamental/tests/test_agent.py -v
.venv/bin/python -m pytest agents/quant_fundamental/tests/test_agent.py -v -m "not integration"
```

---

## Environment Variables

```bash
# LLM (narrative summary only)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_QUANTITATIVE=llama3.2:latest   # or deepseek-r1:8b
QUANT_LLM_TEMPERATURE=0.1
QUANT_LLM_MAX_TOKENS=512
QUANT_REQUEST_TIMEOUT=120                # seconds; prevents hang on slow models

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# Analysis configuration
ANALYSIS_TIME_RANGE=TTM            # TTM / FY2024 / Last4Q
BETA_LOOKBACK_DAYS=60              # Rolling beta window
SHARPE_LOOKBACK_DAYS=365           # Sharpe ratio look-back
ANOMALY_ZSCORE_THRESHOLD=2.0       # Z-score threshold for anomaly flagging
ROLLING_MEAN_YEARS=3               # Years of history for Z-score baseline
QUANT_AGENT_SQL_TIMEOUT=30         # PostgreSQL query timeout (seconds)
BENEISH_THRESHOLD=-2.22            # M-Score threshold for manipulation_risk: HIGH
PIOTROSKI_STRONG_THRESHOLD=7       # F-Score ≥ this → strong
PIOTROSKI_WEAK_THRESHOLD=2         # F-Score ≤ this → weak
```

---

## Validated Live Results

All 5 tickers validated end-to-end against live PostgreSQL data (docker-compose).

| Ticker | `data_quality.status` | `piotroski_f_score` | `pe_trailing` | `sharpe_ratio_12m` |
|---|---|---|---|---|
| AAPL | PASSED (9/9) | 9 | 33.08 | 0.69 |
| MSFT | PASSED (9/9) | 7 | 24.47 | −5.19 |
| GOOGL | PASSED (9/9) | 7 | 28.48 | −3.66 |
| TSLA | PASSED (9/9) | 6 | 342.78 | −2.49 |
| NVDA | PASSED (9/9) | 6 | 35.87 | −1.76 |

---

## Design Decisions

- **Non-RAG by design:** All inputs are structured numerical data from PostgreSQL. RAG adds no value for metric computation and would introduce retrieval noise into deterministic calculations.
- **DuckDB removed:** The former dual-path verifier (Python vs. DuckDB SQL) was replaced by a single-path PostgreSQL data quality check. DuckDB added complexity without catching real errors — all data originates from the same PostgreSQL source, so a second SQL pass was redundant.
- **LLM scope strictly limited:** All JSON fields except `quantitative_summary` are written from Python computation results — never from LLM generation. The LLM prompt includes explicit rules forbidding arithmetic.
- **`request_timeout=120`:** Prevents indefinite hangs on thinking models (e.g. `deepseek-r1:8b`) with large `num_predict` budgets. `"think": false` is also sent in the Ollama payload to suppress chain-of-thought, with `<think>` tag stripping as belt-and-suspenders.
- **`num_predict=512`:** A 3-5 sentence narrative needs ≤150 tokens. The 512 cap prevents the model from generating excessively long responses while still allowing headroom.
- **Natural-language prompt input:** `extract_ticker_from_prompt()` uses a 4-tier strategy to extract the ticker from planner-agent instructions, making the agent callable without requiring the planner to isolate the ticker symbol.
- **Chain-of-Table keeps context clean:** The five-step SELECT → FILTER → CALCULATE → RANK → IDENTIFY chain ensures the LLM summary prompt receives a lean, pre-processed table — not raw JSONB payloads.
- **Z-score baseline is 3-year rolling:** Short baselines (1 year) flagged too many false-positive anomalies for high-growth stocks. Three-year rolling mean/std provides a stable, business-cycle-aware baseline.
- **Beneish M-Score threshold:** The standard −2.22 threshold is used. Scores above −2.22 are flagged `manipulation_risk: HIGH`. Requires prior-year data; returns `null` if only one period is available.
- **Beta uses 60-day rolling window:** Aligns with the Financial Modelling Agent's WACC beta input. `beta_60d` will be `null` if no benchmark data is available in `market_eod_us`.

---

*Last updated: 2026-02-28 | Author: hck717*
