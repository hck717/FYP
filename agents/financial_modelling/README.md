# Financial Modelling Agent

> **Status:** Complete (live-tested, all 5 tickers validated with local docker-compose databases)
> **Role in system:** Deterministic quantitative valuation engine — all numbers are computed, never LLM-generated

---

## Role

The Financial Modelling Agent is the **quantitative valuation layer** of the multi-agent equity research system. It reads historical prices, fundamentals, and earnings data from PostgreSQL, executes deterministic calculations (DCF, Comps, technicals), and returns a fully computed valuation package to the Supervisor.

It does **NOT** generate numbers via LLM. The LLM is only used to interpret the computed outputs and write the `quantitative_summary` narrative. Every numeric output has a traceable formula.

**Handled by this agent:**
- DCF (Discounted Cash Flow) intrinsic value with Bear / Base / Bull scenarios
- Comparable company analysis (EV/EBITDA, P/E, P/S, EV/Revenue peer multiples)
- Technical analysis (trend, momentum, volatility indicators)
- Earnings history analysis (EPS actual vs. estimate, surprise %, beat streak)
- Dividend and buyback history
- Split-adjusted price history
- WACC calculation (CAPM Cost of Equity + Cost of Debt)
- LBO model (stretch goal — optional)

**Handled by other agents (do not overlap):**
- News sentiment and qualitative strategy → Business Analyst Agent
- Macroeconomic rates and FX → Macro Economic Agent
- Real-time breaking news → Web Search Agent

---

## Architecture: Deterministic Python Computation Engine (Non-RAG)

This agent does **not** use RAG. All data comes from structured PostgreSQL tables. The LangGraph pipeline is a sequential computation chain — no retrieval confidence scoring or fallback loops.

```
Query + Ticker
    │
    ▼
fetch_price_history         ←  PostgreSQL: raw_timeseries (1-year EOD prices, split-adjusted)
    │
    ▼
fetch_fundamentals          ←  PostgreSQL: raw_fundamentals (PE, EPS, EBITDA, FCF, Debt, Equity)
    │
    ▼
fetch_earnings_history      ←  PostgreSQL: raw_fundamentals (earnings_history)
                                → EPS actual vs. estimate, surprise %, beat streak
    │
    ▼
calculate_technicals        ←  Python (pandas-ta / manual)
                                → SMA 20/50/200, EMA 12/26, RSI(14), MACD,
                                   Bollinger Bands, ATR, HV30, Stochastic
    │
    ▼
run_dcf_model               ←  Python: 5-year FCF projection + WACC + Terminal Value
                                → Bear / Base / Bull scenario table
                                → Sensitivity matrix (WACC × Terminal Growth Rate)
    │
    ▼
run_comparable_analysis     ←  PostgreSQL: peer group EV/EBITDA, P/E, EV/Revenue
                            ←  Neo4j: COMPETES_WITH / BELONGS_TO edges for peer selection
                                → Median/mean peer multiples applied to target metrics
    │
    ▼
assess_analyst_estimates    ←  PostgreSQL: raw_fundamentals (analyst_estimates)
                                → Consensus EPS/Revenue vs. actuals, surprise history
    │
    ▼
format_json_output          ←  Structured JSON for Supervisor
    │
   END → return to Supervisor
```

---

## Infrastructure State

| Service | Container | Status | Notes |
|---|---|---|---|
| PostgreSQL | `fyp-postgres` | healthy | `raw_timeseries`, `raw_fundamentals` tables populated by EODHD + FMP DAGs. Earnings, dividends, splits stored in `raw_fundamentals` payload JSONB. |
| Neo4j | `fyp-neo4j` | healthy | 5 `Company` nodes available for peer selection via `COMPETES_WITH` / `BELONGS_TO` edges. Full peer graph populated once FMP DAG runs for all S&P 100 tickers. |
| Qdrant | `fyp-qdrant` | operational | Vector DB (not directly used by this agent, but runs in the local network). |
| Ollama | local | running | `deepseek-r1:8b` for narrative summary only; `temperature=0.1` (numeric analysis). Version 0.14.2. |

---

## Data Sources

All data is ingested into **PostgreSQL** via two Airflow DAGs: `fmp_complete_ingestion` (FMP Ultimate) and `eodhd_complete_ingestion` (EODHD All-In-One). Treasury rates and market risk premium are FMP-exclusive — these feed directly into DCF WACC and Cost of Equity calculations.

| Data Type | FMP | EODHD | Primary | PostgreSQL Table | Purpose |
|---|:---:|:---:|:---:|---|---|
| Financial Statements (Q & A) | ✓ | ✓ | Both | `raw_fundamentals` | Model inputs, DCF calculations |
| As-Reported Financials | ✓ | ✗ | FMP | `raw_fundamentals` | GAAP/IFRS compliance checking |
| DCF Valuation Models | ✓ | ✗ | FMP | `raw_fundamentals` | Valuation model inputs and outputs |
| Revenue Segmentation | ✓ | ✗ | FMP | `raw_fundamentals` | Segment analysis, growth projections |
| Analyst Estimates | ✓ | ✓ | Both | `raw_fundamentals` | Consensus vs. actual tracking |
| Historical Stock Prices | ✓ | ✓ | Both | `raw_timeseries` | Returns calculation, valuation |
| Dividend History | ✓ | ✓ | Both | `raw_fundamentals` | Dividend discount model, yield analysis |
| Treasury Rates | ✓ | ✗ | FMP | `raw_timeseries` | Risk-free rate (\(R_f\)) for WACC/DCF |
| Market Risk Premium | ✓ | ✗ | FMP | `raw_fundamentals` | Cost of equity calculation (\(R_m - R_f\)) |
| Economic Indicators | ✓ | ✓ | Both | `raw_timeseries` | Macro factors for scenario analysis |
| Peer Company Data | ✓ | ✓ | Both | `raw_fundamentals` | Comparable company analysis |
| Industry Benchmarks | ✓ | ✗ | FMP | `raw_fundamentals` | Sector comparison metrics |

### Source API Details

| API | Endpoint Type | Ingestion DAG | Update Frequency |
|---|---|---|---|
| FMP Ultimate | REST JSON (financials, DCF models, segments, treasury rates, market risk premium) | `fmp_complete_ingestion` | Daily at 02:00 UTC |
| EODHD All-In-One | REST JSON (fundamentals, EOD prices, analyst estimates, dividends) | `eodhd_complete_ingestion` | Daily at 01:00 UTC |

### Key PostgreSQL Tables

| Table | Content |
|---|---|
| `raw_fundamentals` | PE (trailing + forward), EPS, EBITDA, FCF, Total Debt, Total Equity, Market Cap, Analyst Estimates, Earnings History, Dividends, Splits, As-Reported Financials, Revenue Segments, DCF Model Inputs, Market Risk Premium — stored as `payload JSONB` keyed by `data_name` and `period` |
| `raw_timeseries` | EOD prices (OHLCV), intraday, SMA/EMA pre-computed, split-adjusted, Treasury Rates (3M, 2Y, 10Y, 30Y), Economic Indicators |
| `market_eod_us` | US equity market EOD — S&P 100 universe, used for peer beta, sector return, and Comps universe |

---

## Models Implemented

### Discounted Cash Flow (DCF)

**WACC Calculation:**

\[
\text{WACC} = \frac{E}{V} \times R_e + \frac{D}{V} \times R_d \times (1 - T)
\]

- \( R_e \) (Cost of Equity) via CAPM: \( R_e = R_f + \beta \times (R_m - R_f) \)
- \( R_f \) = 10Y Treasury Yield (sourced from `raw_timeseries` — FMP treasury rates endpoint)
- \( R_m - R_f \) = Market Risk Premium (sourced from `raw_fundamentals` — FMP market risk premium endpoint)
- \( \beta \) = 60-day rolling beta vs. S&P 500 (computed from `market_eod_us`)
- \( R_d \) = Interest Expense / Total Debt (from `raw_fundamentals`)
- \( T \) = Effective Tax Rate (from `raw_fundamentals`)

**Terminal Value (Gordon Growth Model):**

\[
\text{TV} = \frac{\text{FCF}_n \times (1 + g)}{\text{WACC} - g}
\]

**Scenario Engine:**

```python
scenarios = {
    "Bear": {"revenue_growth": -0.05, "ebit_margin": 0.10, "wacc": 0.12},
    "Base": {"revenue_growth":  0.08, "ebit_margin": 0.18, "wacc": 0.10},
    "Bull": {"revenue_growth":  0.20, "ebit_margin": 0.25, "wacc": 0.09},
}
# Output: {"Bear": $85, "Base": $130, "Bull": $195}
```

**Sensitivity Matrix** (WACC × Terminal Growth Rate):

| | g=1.5% | g=2.0% | g=2.5% | g=3.0% |
|---|---|---|---|---|
| WACC=8% | $148 | $155 | $163 | $172 |
| WACC=9% | $131 | $137 | $143 | $150 |
| WACC=10% | $118 | $123 | $128 | $134 |
| WACC=11% | $107 | $111 | $115 | $120 |

### Comparable Company Analysis (Comps)

- Peer group sourced from Neo4j `COMPETES_WITH` / `BELONGS_TO` relationships (top 5 peers by sector/industry)
- Peer financial data from `raw_fundamentals` (FMP + EODHD, both sourced)
- Industry benchmark multiples from `raw_fundamentals` (FMP — sector median EV/EBITDA, P/E, EV/Revenue)
- Multiples applied: EV/EBITDA, P/E (trailing + forward), P/S, EV/Revenue
- Output: Implied valuation range (min / median / max peer multiple applied to target metrics)
- Comparison: `vs_sector_avg` expressed as `premium +X%` or `discount -X%`

### Technical Analysis

**Trend Indicators:**
- SMA 20, SMA 50, SMA 200
- EMA 12, EMA 26
- Golden Cross (SMA 50 crosses above SMA 200) / Death Cross (SMA 50 crosses below SMA 200)

**Momentum Indicators:**
- RSI (14): >70 overbought, <30 oversold
- MACD and Signal Line divergence → `buy` / `sell` / `neutral`
- Stochastic Oscillator %K and %D

**Volatility:**
- Bollinger Bands (20-period, 2σ): price position relative to bands
- ATR (Average True Range, 14-day)
- Annualised Historical Volatility (HV 30)

**Support / Resistance:**
- 52-week high and low
- Rolling pivot points (last 20-day high/low)

---

## LLM & Models

```python
# Primary LLM — used ONLY for quantitative_summary narrative interpretation
llm_model = "deepseek-r1:8b"       # via Ollama at localhost:11434 (LLM_MODEL_FINANCIAL_MODELING)
temperature = 0.1                   # Very low: number interpretation, not creativity
request_timeout = None              # No timeout

# think tag suppression (Ollama >= 0.14.2)
think = False
```

**LLM scope is strictly limited:** The LLM receives the fully computed numeric outputs as context and is instructed only to write the `quantitative_summary` string and flag any anomalies in plain language. It does not perform arithmetic.

---

## Output Format (JSON)

The agent returns **structured JSON only** — no freeform Markdown prose. All numeric fields are Python-computed — never LLM-generated.

```json
{
  "agent": "financial_modelling",
  "ticker": "AAPL",
  "as_of_date": "2026-02-27",
  "current_price": 218.50,
  "valuation": {
    "dcf": {
      "intrinsic_value_base": 195.20,
      "intrinsic_value_bear": 148.00,
      "intrinsic_value_bull": 253.00,
      "upside_pct_base": -10.7,
      "wacc_used": 0.098,
      "terminal_growth_rate": 0.025,
      "scenario_probability": {"bear": 0.25, "base": 0.55, "bull": 0.20}
    },
    "comps": {
      "ev_ebitda": 22.4,
      "pe_trailing": 31.2,
      "pe_forward": 27.8,
      "ps_ttm": 8.1,
      "ev_revenue": 7.9,
      "vs_sector_avg": "premium +18%",
      "peer_group": ["MSFT", "GOOGL", "META", "AMZN", "NVDA"]
    },
    "implied_price_range": {
      "low": 148.00,
      "mid": 195.20,
      "high": 253.00
    }
  },
  "technicals": {
    "trend": "bullish",
    "rsi_14": 58.3,
    "macd_signal": "buy",
    "macd_histogram": 1.24,
    "sma_50": 212.40,
    "sma_200": 198.80,
    "sma_50_above_200": true,
    "golden_cross": false,
    "death_cross": false,
    "bollinger_position": "mid",
    "atr_14": 3.82,
    "hv_30": 0.198,
    "support": 205.00,
    "resistance": 230.00,
    "52w_high": 237.49,
    "52w_low": 164.08
  },
  "earnings": {
    "last_eps_actual": 2.40,
    "last_eps_estimate": 2.35,
    "surprise_pct": 2.1,
    "beat_streak": 6,
    "miss_streak": 0,
    "next_earnings_date": "2026-04-24"
  },
  "dividends": {
    "dividend_yield": 0.0052,
    "annual_dividend": 1.00,
    "payout_ratio": 0.145,
    "dividend_growth_5y_cagr": 0.054
  },
  "factor_scores": {
    "piotroski_f_score": 7,
    "beneish_m_score": -2.41,
    "altman_z_score": 4.82
  },
  "quantitative_summary": "Apple trades at a premium to peers (EV/EBITDA +18% vs. sector). DCF base case suggests -11% downside at current price, though bull scenario (+16% revenue growth) supports 253 intrinsic value. Technical momentum remains constructive — RSI neutral at 58, SMA50 above SMA200. Piotroski F-Score of 7 indicates good financial health.",
  "data_sources": {
    "price_data": "postgresql:raw_timeseries",
    "fundamentals": "postgresql:raw_fundamentals",
    "peer_group": "neo4j:COMPETES_WITH",
    "treasury_rates": "postgresql:raw_timeseries (FMP treasury endpoint)",
    "market_risk_premium": "postgresql:raw_fundamentals (FMP market risk premium endpoint)"
  }
}
```

---

## File Structure

```
agents/financial_modelling/
├── README.md              # This file
├── __init__.py            # Package init — exports run() and run_full_analysis()
├── agent.py               # LangGraph pipeline (8 nodes) + run() + run_full_analysis()
├── config.py              # Centralised env-var configuration
├── tools.py               # PostgreSQL query helpers, Neo4j peer selector
├── prompts.py             # System prompt + quantitative summary prompt + JSON schema
├── schema.py              # Dataclasses: ValuationResult, TechnicalSnapshot, EarningsRecord, etc.
├── models/
│   ├── dcf.py             # DCF engine: WACC, FCF projections, Terminal Value, scenario table
│   ├── valuation.py       # Comps: EV/EBITDA, P/E, P/S peer multiples from PostgreSQL
│   └── technicals.py      # RSI, MACD, Bollinger Bands, ATR, HV30, SMA/EMA
└── tests/
    └── test_agent.py      # Unit + integration tests (all mocked)
```

---

## Public API

The package exports two functions from `agents.financial_modelling`:

### `run(task, ticker, config)` — targeted query

```python
from agents.financial_modelling import run

result = run(task="What is Apple's DCF intrinsic value?", ticker="AAPL")
```

Answers a **single quantitative question**. Output is scoped to the relevant model (e.g. only DCF is run if the task is DCF-specific). Use this when the Supervisor has a targeted follow-up.

### `run_full_analysis(ticker, config)` — comprehensive valuation dossier

```python
from agents.financial_modelling import run_full_analysis

dossier = run_full_analysis(ticker="AAPL")
# dossier["valuation"]["dcf"]["intrinsic_value_base"]  → 195.20
# dossier["technicals"]["rsi_14"]                       → 58.3
# dossier["earnings"]["beat_streak"]                    → 6
# dossier["quantitative_summary"]                        → "..."
```

Runs **all models** (DCF, Comps, Technicals, Earnings, Dividends, Factor Scores) in a single pipeline pass. This is the intended entry point for the Synthesizer. Returns the full JSON schema above.

---

## Environment Variables

```bash
# LLM (narrative summary only)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_FINANCIAL_MODELING=deepseek-r1:8b

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# Neo4j (peer group selection)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# Analysis configuration
PRICE_HISTORY_DAYS=365          # Look-back window for technical analysis
DCF_FORECAST_YEARS=5            # Number of years to project FCF
DCF_TERMINAL_GROWTH_RATE=0.025  # Terminal growth rate (default; overridden by Macro Agent)
DCF_WACC=0.09                   # WACC default (overridden by live CAPM calculation)
COMPS_SECTOR_PEERS=5            # Number of peer companies for Comps analysis
```

---

## Run Commands

```bash
cd /Users/brianho/FYP && \
LLM_MODEL_FINANCIAL_MODELING=deepseek-r1:8b \
python -m agents.financial_modelling.agent \
  --ticker AAPL \
  --log-level WARNING
```

**Suggested test commands:**

```bash
# 1. AAPL — full valuation dossier (DCF, comps, technicals, dividends, factor scores)
python -m agents.financial_modelling.agent --ticker AAPL --log-level WARNING

# 2. MSFT — verifies dividend CAGR, payout ratio, and bearish RSI signal
python -m agents.financial_modelling.agent --ticker MSFT --log-level WARNING

# 3. GOOGL — checks EV/EBITDA sector discount computation and Altman Z-Score derivation
python -m agents.financial_modelling.agent --ticker GOOGL --log-level WARNING

# 4. TSLA — high-volatility stock; exercises Bollinger band lower-band signal and ATR output
python -m agents.financial_modelling.agent --ticker TSLA --log-level WARNING

# 5. NVDA — high-growth stock; tests DCF sensitivity matrix with elevated P/S and EV/Revenue multiples
python -m agents.financial_modelling.agent --ticker NVDA --log-level WARNING

# Run unit test suite
python -m pytest agents/financial_modelling/tests/ -q
```

---

## Validated Live Results

All 5 supported tickers have been validated end-to-end with real PostgreSQL and Neo4j data running locally via `docker-compose.yml`.

| Ticker | Status | Computations Verified | Citations in output | Fallback |
|---|---|---|---|---|
| AAPL | CORRECT | Yes | N/A (Computed) | false |
| TSLA | CORRECT | Yes | N/A (Computed) | false |
| MSFT | CORRECT | Yes | N/A (Computed) | false |
| NVDA | CORRECT | Yes | N/A (Computed) | false |
| GOOGL | CORRECT | Yes | N/A (Computed) | false |

---

## Design Decisions

- **Numbers are never LLM-generated:** All numeric outputs (DCF intrinsic value, RSI, EV/EBITDA, WACC, etc.) are computed in Python. The LLM is used exclusively to write the `quantitative_summary` interpretation string. This guarantees mathematical reproducibility.
- **Treasury rates and Market Risk Premium are FMP-exclusive:** EODHD does not provide these endpoints. The risk-free rate (\(R_f\)) and equity risk premium (\(R_m - R_f\)) used in CAPM are therefore sourced from FMP only and stored in `raw_timeseries` and `raw_fundamentals` respectively. If the FMP DAG has not yet run for the current day, the agent falls back to the previous day’s cached value.
- **Dual-path verification (from Quant Fundamental Agent pattern):** Where applicable, critical calculations (e.g. WACC) are cross-verified with the Quant Fundamental Agent’s outputs. If the two agents produce materially different WACC figures, the Supervisor flags a `CalculationAlert`.
- **WACC is live, not hardcoded:** The `DCF_WACC` env var is a default fallback only. At runtime, WACC is computed from live CAPM inputs: 10Y yield (from `raw_timeseries`), Market Risk Premium (from `raw_fundamentals`), rolling 60-day beta (from `market_eod_us`), and cost of debt from `raw_fundamentals`.
- **As-Reported Financials for GAAP/IFRS compliance:** FMP provides as-reported financials (pre-adjustment) alongside standardised financials. The agent uses as-reported data to check for unusual restatements or GAAP/IFRS presentation differences that could distort comparability in Comps.
- **Revenue Segmentation for growth projections:** FMP revenue segment data (geography + product line) is used to build differentiated growth assumptions per segment in the Bear/Base/Bull DCF scenarios, rather than applying a single blended growth rate.
- **Split-adjusted prices:** All price history retrieved from `raw_timeseries` is split-adjusted before technical indicator computation to avoid distorted signals from historical splits.
- **Peer group from Neo4j:** Comparable companies are selected via Neo4j `COMPETES_WITH` and `BELONGS_TO` Cypher queries. If Neo4j returns fewer than 5 peers, the agent falls back to selecting the top-5 companies by market cap in the same GICS sector from `raw_fundamentals`.
- **Industry benchmarks from FMP:** Sector-level median multiples (EV/EBITDA, P/E, EV/Revenue) are sourced from FMP’s industry benchmarks endpoint and used as the denominator for `vs_sector_avg` comparisons in Comps output.
- **Scenario probability weighting:** The three DCF scenarios are probability-weighted: `E[IV] = 0.25 × Bear + 0.55 × Base + 0.20 × Bull`. Weights are configurable via the config object.
- **`data_sources` block in output:** Every numeric output block includes a `data_sources` field citing the exact PostgreSQL table and originating API. This provides an auditable provenance chain for the Synthesizer’s citation verification stage.

---

*Last updated: 2026-03-01 | Author: hck717*
