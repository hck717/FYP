# Financial Modelling Agent

> **Status:** Complete (live-tested, all 5 tickers validated with local docker-compose databases)
> **Role in system:** Deterministic quantitative valuation engine — all numbers are computed, never LLM-generated

---

## Role

The Financial Modelling Agent is the **quantitative valuation layer** of the multi-agent equity research system. It reads historical prices, fundamentals, and earnings data exclusively from PostgreSQL, executes deterministic calculations (DCF, WACC, Comps, 3-Statement Model, Factor Scores), and returns a fully computed valuation package to the Supervisor.

It does **NOT** generate numbers via LLM. The LLM is only used to interpret the computed outputs and write the `quantitative_summary` narrative and MoE persona narratives. Every numeric output has a traceable formula.

**Handled by this agent:**
- DCF (Discounted Cash Flow) intrinsic value with Bear / Base / Bull scenarios + sensitivity matrix
- Mixture-of-Experts (MoE) DCF consensus: Optimist / Pessimist / Realist persona adjustments
- Macro environment snapshot: VIX-regime-adjusted Market Risk Premium fed into WACC
- Comparable company analysis (EV/EBITDA, P/E, P/S, EV/Revenue peer multiples)
- Technical analysis (trend, momentum, volatility indicators)
- Earnings history analysis (EPS actual vs. estimate, surprise %, beat/miss streak, next date)
- Dividend and buyback history (yield, annual dividend, payout ratio, 5-year CAGR)
- 3-Statement Model (linked IS + BS + CF with accounting linkage checks)
- Factor scores: Piotroski F-Score, Beneish M-Score, Altman Z-Score

**Handled by other agents (do not overlap):**
- News sentiment and qualitative strategy → Business Analyst Agent
- Real-time breaking news → Web Search Agent

---

## Architecture: 11-Node LangGraph Pipeline

This agent uses **LangGraph** as a sequential computation chain. There is no RAG, no retrieval confidence scoring, and no conditional branches. All 11 nodes execute in order.

```
ticker
    │
    ▼
fetch_price_history         ←  PostgreSQL: raw_timeseries (EOD prices, split-adjusted)
    │                           market_eod_us (S&P 500 benchmark for beta/HV30)
    ▼
fetch_fundamentals          ←  PostgreSQL: raw_fundamentals (income, balance, cashflow,
    │                           scores, key_metrics_ttm, ratios_ttm, enterprise_values)
    │                           earnings_surprises (dedicated table, 521 rows)
    │                           dividends_history (dedicated table, 394 rows)
    │                           valuation_metrics, outstanding_shares
    ▼
fetch_earnings_history      ←  bundle.earnings_surprises (primary)
    │                           → EPS actual/estimate, surprise %, beat/miss streak,
    │                             next earnings date
    ▼
calculate_technicals        ←  Python (pandas-ta / manual)
    │                           → SMA 20/50/200, EMA 12/26, RSI(14), MACD + signal line,
    │                             Bollinger Bands, ATR(14), HV30, Stochastic %K/%D
    ▼
macro_environment           ←  bundle.treasury_rates, bundle.benchmark_history (already fetched)
    │                           → 10Y Treasury yield, S&P 500 HV30 proxy for VIX regime
    │                           → VIX-adjusted MRP written into bundle.market_risk_premium
    ▼
run_dcf_model               ←  Python: WACC (CAPM), 3-stage FCF projection, Terminal Value
    │                           → Bear / Base / Bull scenario table
    │                           → Sensitivity matrix (WACC × Terminal Growth Rate)
    │                           → Reverse DCF implied CAGR at current market price
    ▼
moe_consensus               ←  DeepSeek API (deepseek-reasoner): 3 parallel persona threads
    │                           → Optimist / Pessimist / Realist DCF adjustments
    │                           → Probability-weighted Bear/Base/Bull consensus
    ▼
run_comparable_analysis     ←  Python + Neo4j peer group + PG peer multiples
    │                           → EV/EBITDA, P/E (trailing + forward), P/S, EV/Revenue, P/FCF
    │                           → vs_sector_avg: "premium +X%" or "discount -X%"
    │                           → valuation_metrics table fills any null multiples
    ▼
assess_analyst_estimates    ←  bundle.dividends_dedicated (primary: dividends_history table)
    │                           → dividend yield, annual dividend, payout ratio, 5y CAGR
    │                           → Piotroski F-Score (2 annual periods)
    │                           → Beneish M-Score (accruals-based)
    │                           → Altman Z-Score (5-variable)
    ▼
build_three_statement_model ←  Python: IS + BS + CF reconciliation
    │                           → income_annual / balance_annual / cashflow_annual (primary)
    │                           → revenue, gross_margin, net_margin, OCF, FCF, net_debt
    │                           → Linkage checks: RE linkage, cash linkage, BS balance
    ▼
format_json_output          ←  Structured JSON assembly
    │                           LLM writes quantitative_summary only (no arithmetic)
    │
   END → return to Supervisor
```

---

## Data Sources

All data is ingested into **PostgreSQL** via Airflow DAGs. The agent **never calls external APIs directly** — all reads go through PostgreSQL only.

### PostgreSQL Tables Used

| Table | Content | Used For |
|---|---|---|
| `raw_fundamentals` | JSONB payload keyed by `data_name` and `period_type`. Rows per ticker: `income_statement`, `balance_sheet`, `cash_flow`, `financial_ratios`, `financial_scores`, `key_metrics_ttm`, `ratios_ttm`, `enterprise_values` | DCF inputs, factor scores, comps multiples |
| `raw_timeseries` | EOD prices (OHLCV), split-adjusted, weekly / daily | Price history for technicals and beta |
| `market_eod_us` | US equity market EOD — S&P 100 universe | S&P 500 benchmark for rolling beta and HV30 |
| `earnings_surprises` | `ticker, period_date, eps_actual, eps_estimate, eps_surprise_pct, revenue_actual, revenue_estimate, revenue_surprise_pct, before_after_market` | EPS surprise, beat/miss streak, next earnings date |
| `dividends_history` | `ticker, amount, ex_date, pay_date, record_date` (amount > 0 filter) | Annual dividend, 5-year CAGR, payout ratio |
| `valuation_metrics` | `trailing_pe, forward_pe, ev_ebitda, ev_revenue, price_sales_ttm, free_cash_flow, market_cap, enterprise_value, beta, wacc` | Fill-null fallback for comps multiples |
| `outstanding_shares` | `ticker, shares_outstanding, as_of_date` | Share count for per-share calculations |
| `splits_history` | Split date, ratio | Price history adjustment reference |
| `treasury_rates` | 10Y, 2Y, 3M, 30Y Treasury yields | Risk-free rate for WACC / CAPM |
| `global_macro_indicators` | GDP, CPI, unemployment, VIX | Macro regime context |
| `economic_events` | Economic calendar events | Macro context |
| `corporate_bond_yields` | IG/HY bond yields | Cost of debt inputs |
| `forex_rates` | FX rates (EOD) | Currency adjustment for international peers |

### `raw_fundamentals` Schema

```
id, ticker_symbol, data_name, as_of_date, payload (JSONB), source, ingested_at, period_type
```

- `period_type = 'yearly'` or `'quarterly'`
- `data_name` values for each ticker: `balance_sheet`, `cash_flow`, `enterprise_values`, `financial_ratios`, `financial_scores`, `income_statement`, `key_metrics_ttm`, `ratios_ttm`

### EODHD Field Name Conventions (important gotcha)

EODHD uses different field names from FMP. Both are handled in `tools.py` with bidirectional aliases.

**Balance Sheet:**
- `totalStockholderEquity` (EODHD) ≡ `totalStockholdersEquity` (FMP)
- `totalLiab` (EODHD) ≡ `totalLiabilities` (FMP)
- `cash` (EODHD) ≡ `cashAndCashEquivalents` (FMP)
- `shortLongTermDebt`, `longTermDebtTotal` (EODHD) ≡ `shortTermDebt`, `longTermDebt` (FMP)
- `goodWill` (EODHD) ≡ `goodwill` (FMP)

**Cash Flow:**
- `totalCashFromOperatingActivities` (EODHD) ≡ `operatingCashFlow` (FMP)
- `capitalExpenditures` (EODHD) ≡ `capitalExpenditure` (FMP)
- `endPeriodCashFlow`, `beginPeriodCashFlow` (EODHD)
- `dividendsPaid`, `salePurchaseOfStock`, `netBorrowings`, `stockBasedCompensation` (EODHD)

**Income Statement:**
- `totalRevenue` (EODHD) ≡ `revenue` (FMP)
- `taxProvision` (EODHD) ≡ `incomeTaxExpense` (FMP)
- `researchDevelopment` (EODHD) ≡ `researchAndDevelopmentExpenses` (FMP)

---

## `FMDataBundle` Key Fields

The `FMDataBundle` dataclass (defined in `schema.py`) carries all data between pipeline nodes:

| Field | Source | Content |
|---|---|---|
| `income` / `balance` / `cashflow` | `raw_fundamentals` most recent row (may be quarterly) | Used for Piotroski, Altman, Beneish, DCF |
| `income_annual` / `balance_annual` / `cashflow_annual` | `raw_fundamentals` most recent `period_type='yearly'` row | Primary input to 3-Statement Model |
| `income_prior` / `balance_prior` / `cashflow_prior` | `raw_fundamentals` prior-year `period_type='yearly'` row | Piotroski YoY signals (F3, F5, F6, F8, F9) |
| `earnings_surprises` | `earnings_surprises` table, sorted `period_date DESC` | EPS actuals, estimates, surprise %, beat/miss streak |
| `dividends_dedicated` | `dividends_history` table, sorted `pay_date DESC`, `amount > 0` | Annual dividend total, 5-year CAGR |
| `valuation_metrics` | `valuation_metrics` table | Fill-null fallback for comps multiples |
| `price_history` | `raw_timeseries` weekly EOD (fallback: daily) | Technical indicators |
| `benchmark_history` | `market_eod_us` | S&P 500 rolling beta, HV30 for VIX proxy |
| `treasury_rates` | `raw_timeseries` or `treasury_rates` dedicated table | 10Y yield → risk-free rate |

---

## Models Implemented

### Discounted Cash Flow (DCF) — `models/dcf.py`

**WACC (live, not hardcoded):**

```
WACC = (E/V) × Re + (D/V) × Rd × (1 − T)

Re  = Rf + β × (Rm − Rf)          [CAPM]
Rf  = 10Y Treasury yield           [treasury_rates table]
β   = 60-day rolling vs. S&P 500   [market_eod_us]
Rm−Rf = VIX-adjusted MRP           [base MRP from raw_fundamentals, adjusted by HV30 regime]
Rd  = Interest Expense / Total Debt [raw_fundamentals]
T   = Effective tax rate            [raw_fundamentals]
```

**VIX Regime MRP Adjustment (macro_environment node):**

| HV30 Regime | Label | MRP Delta |
|---|---|---|
| < 15% | low | −0.005 |
| 15–25% | normal | 0 |
| 25–35% | high | +0.010 |
| > 35% | extreme | +0.020 |

**3-Stage FCF Projection + Terminal Value:**

```
TV = FCFn × (1 + g) / (WACC − g)    [Gordon Growth Model]
```

**Scenario Engine:**

| Scenario | Revenue Growth | EBIT Margin | WACC Offset |
|---|---|---|---|
| Bear | −5% | 10% | +2% |
| Base | +8% | 18% | 0% |
| Bull | +20% | 25% | −1% |

**Sensitivity Matrix:** WACC (4 levels) × Terminal Growth Rate (4 levels) → implied intrinsic value grid.

**Reverse DCF:** Implied revenue CAGR at current market price.

---

### Mixture-of-Experts (MoE) DCF Consensus — `agent.py`

Three parallel LLM persona threads (DeepSeek `deepseek-reasoner`) each adjust the base DCF result:

| Persona | TGR Delta | WACC Delta | Revenue Multiplier |
|---|---|---|---|
| Optimist | +0.5% | −1.0% | 1.25× |
| Pessimist | −0.5% | +1.5% | 0.70× |
| Realist | 0% | 0% | 1.0× |

- Consensus = equal-weight average of the three persona Bear/Base/Bull values
- Synthesis narrative generated by a fourth LLM call (DeepSeek `deepseek-reasoner`, fallback to `deepseek-chat`)

---

### Comparable Company Analysis (Comps) — `models/valuation.py`

- Peer group from Neo4j `COMPETES_WITH` / `BELONGS_TO` relationships (top 5 peers)
- Multiples computed: EV/EBITDA, EV/EBIT, P/E trailing, P/E forward, P/S, EV/Revenue, P/FCF, PEG
- Null multiples filled from `valuation_metrics` table
- `vs_sector_avg`: target multiple vs. peer median → expressed as "premium +X%" or "discount −X%"

---

### Technical Analysis — `models/technicals.py`

**Trend:** SMA 20/50/200, EMA 12/26, Golden Cross (SMA50 > SMA200), Death Cross

**Momentum:** RSI(14) — overbought >70 / oversold <30; MACD histogram + signal line (9-period EMA); Stochastic %K and %D

**Volatility:** Bollinger Bands (20-period, 2σ); ATR(14); Annualised HV30

**Support/Resistance:** 52-week high/low; 20-day rolling pivot points

---

### 3-Statement Model — `models/three_statement.py`

Produces linked Income Statement + Balance Sheet + Cash Flow for the two most recent annual fiscal periods, with three accounting linkage checks:

| Check | Formula | Tolerance |
|---|---|---|
| RE linkage | `ΔRetained Earnings ≈ Net Income − Dividends + Buybacks` | ±$5B absolute or ±10% relative |
| Cash linkage | `EndCash ≈ BeginCash + OCF + Capex + Financing` | ±$5B absolute or ±10% relative |
| BS balance | `Total Assets ≈ Total Liabilities + Total Equity` | ±$5B absolute or ±10% relative |

Primary data source: `bundle.income_annual` / `balance_annual` / `cashflow_annual` (annual period). Fallback: `bundle.income` / `balance` / `cashflow`.

**Verified results for all 5 tickers:**

| Ticker | Fiscal Periods | Revenue | Linkage |
|---|---|---|---|
| AAPL | FY2024-09-30, FY2025-09-30 | $391B → $416B | All pass |
| MSFT | FY2024-06-30, FY2025-06-30 | $245B → $282B | All pass |
| GOOGL | FY2024-12-31, FY2025-12-31 | $350B → $403B | All pass |
| TSLA | FY2024-12-31, FY2025-12-31 | $97.7B → $94.8B | All pass |
| NVDA | FY2025-01-31, FY2026-01-31 | $130B → $216B | All pass |

---

### Piotroski F-Score — `agent.py`

Full 9-signal model across two annual periods (`bundle.income/balance/cashflow` vs. `bundle.income_prior/balance_prior/cashflow_prior`).

| Pillar | Signal | Formula |
|---|---|---|
| Profitability | F1 | ROA > 0 |
| Profitability | F2 | OCF > 0 |
| Profitability | F3 | ΔROA > 0 (YoY improvement) |
| Profitability | F4 | OCF/Assets > ROA (cash earnings > accruals) |
| Leverage/Liquidity | F5 | ΔLeverage ≤ 0 (LTD/Assets not increased) |
| Leverage/Liquidity | F6 | ΔCurrent Ratio ≥ 0 (liquidity not decreased) |
| Leverage/Liquidity | F7 | No new shares issued |
| Efficiency | F8 | ΔGross Margin > 0 |
| Efficiency | F9 | ΔAsset Turnover > 0 |

Returns `None` if fewer than 5 of 9 signals are computable.

---

### Beneish M-Score — `agent.py`

Simplified accruals-based model (single period; full 8-variable model requires YoY deltas):

```
M ≈ −6.065 + 4.679 × (Net Income − OCF) / Total Assets
```

Threshold: M > −2.22 suggests possible earnings manipulation.

OCF sourced from `bundle.cashflow.operatingCashFlow` (direct from CF statement); fallback: `incomeQualityTTM × Net Income`.

---

### Altman Z-Score — `agent.py`

```
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

X1 = Working Capital / Total Assets
X2 = Retained Earnings / Total Assets
X3 = EBIT / Total Assets
X4 = Market Cap / Total Liabilities
X5 = Revenue / Total Assets
```

Sourced from `raw_fundamentals.financial_scores` (pre-computed EODHD value) if available; otherwise computed from IS/BS fields.

---

## LLM Configuration

```python
# Primary LLM — DeepSeek API (cloud)
llm_provider = "deepseek"
llm_model    = "deepseek-reasoner"
base_url     = "https://api.deepseek.com"
temperature  = 0.1   # narrative interpretation, not creativity

# Fallback for MoE persona narratives (if deepseek-reasoner returns empty)
fallback_model = "deepseek-chat"

# Fallback if DeepSeek API is unreachable
fallback_provider = "ollama"   # localhost:11434
```

**LLM scope is strictly limited:**
- `quantitative_summary` — 10–15 sentence interpretation of the pre-computed factor table
- MoE persona narratives — 2-sentence justification per persona (Optimist / Pessimist / Realist)
- MoE synthesis — 3-sentence Bear/Base/Bull consensus narrative

The LLM receives fully computed numeric outputs as context. It does **not** perform arithmetic.

---

## Output Format (JSON)

```json
{
  "agent": "financial_modelling",
  "ticker": "AAPL",
  "as_of_date": "2026-03-10",
  "current_price": 218.50,
  "valuation": {
    "dcf": {
      "intrinsic_value_base": 195.20,
      "intrinsic_value_bear": 148.00,
      "intrinsic_value_bull": 253.00,
      "intrinsic_value_weighted": 195.85,
      "upside_pct_base": -10.7,
      "wacc_used": 0.098,
      "terminal_growth_rate": 0.025,
      "beta_used": 1.21,
      "scenario_table": [...],
      "sensitivity_matrix": {...},
      "reverse_dcf_implied_cagr": 0.062
    },
    "comps": {
      "ev_ebitda": 22.4,
      "pe_trailing": 31.2,
      "pe_forward": 27.8,
      "ps_ttm": 8.1,
      "ev_revenue": 7.9,
      "p_fcf": 28.5,
      "vs_sector_avg": "premium +18%",
      "peer_group": ["MSFT", "GOOGL", "META", "AMZN", "NVDA"]
    },
    "implied_price_range": {"low": 148.00, "mid": 195.20, "high": 253.00}
  },
  "moe_consensus": {
    "personas": [
      {"persona": "Optimist", "bear": 162.0, "base": 214.0, "bull": 278.0, "narrative": "..."},
      {"persona": "Pessimist", "bear": 111.0, "base": 146.0, "bull": 190.0, "narrative": "..."},
      {"persona": "Realist",  "bear": 148.0, "base": 195.0, "bull": 253.0, "narrative": "..."}
    ],
    "consensus_bear": 140.0,
    "consensus_base": 185.0,
    "consensus_bull": 240.0,
    "consensus_narrative": "..."
  },
  "macro_environment": {
    "risk_free_rate_10y": 0.043,
    "benchmark_hv30": 0.182,
    "vix_regime": "normal",
    "base_mrp": 0.055,
    "mrp_delta": 0.0,
    "vix_adjusted_mrp": 0.055
  },
  "technicals": {
    "trend": "bullish",
    "rsi_14": 58.3,
    "macd_signal": "buy",
    "macd_signal_line": 2.15,
    "macd_histogram": 1.24,
    "sma_50": 212.40,
    "sma_200": 198.80,
    "sma_50_above_200": true,
    "golden_cross": false,
    "death_cross": false,
    "bollinger_upper": 228.00,
    "bollinger_lower": 196.00,
    "bollinger_position": "mid",
    "atr_14": 3.82,
    "hv_30": 0.198,
    "stochastic_k": 62.1,
    "stochastic_d": 58.4,
    "support": 205.00,
    "resistance": 230.00,
    "52w_high": 237.49,
    "52w_low": 164.08
  },
  "earnings": {
    "last_eps_actual": 2.84,
    "last_eps_estimate": 2.35,
    "surprise_pct": 20.85,
    "beat_streak": 12,
    "miss_streak": 0,
    "next_earnings_date": "2026-04-29"
  },
  "dividends": {
    "dividend_yield": 0.0052,
    "annual_dividend": 1.04,
    "payout_ratio": 0.145,
    "dividend_growth_5y_cagr": 0.0487
  },
  "factor_scores": {
    "piotroski_f_score": 7,
    "beneish_m_score": -2.41,
    "altman_z_score": 4.82
  },
  "three_statement_model": {
    "ticker": "AAPL",
    "income_statements": [
      {"period": "2024-09-30", "revenue": 391035000000, "gross_profit": ..., "net_income": ...},
      {"period": "2025-09-30", "revenue": 416000000000, ...}
    ],
    "balance_sheets": [...],
    "cash_flows": [...],
    "linkage_checks": [
      {"period": "2025-09-30", "re_linkage_holds": true, "cash_linkage_holds": true, "bs_balance_holds": true},
      {"period": "2024-09-30", "re_linkage_holds": null, "cash_linkage_holds": true, "bs_balance_holds": true}
    ]
  },
  "quantitative_summary": "Apple trades at a premium to peers...",
  "data_sources": {
    "price_data": "postgresql:raw_timeseries",
    "fundamentals": "postgresql:raw_fundamentals",
    "peer_group": "neo4j:COMPETES_WITH",
    "treasury_rates": "postgresql:raw_timeseries (FMP treasury endpoint)",
    "market_risk_premium": "postgresql:raw_fundamentals (FMP market risk premium endpoint)",
    "llm_scope": "quantitative_summary narrative only",
    "llm_model": "deepseek-reasoner"
  },
  "price_history": [...],
  "benchmark_history": [...]
}
```

---

## File Structure

```
agents/financial_modelling/
├── README.md              # This file
├── __init__.py            # Package init — exports run() and run_full_analysis()
├── agent.py               # LangGraph 11-node pipeline + all node functions
│                          #   Includes: Piotroski, Beneish, Altman computations
│                          #   MoE persona engine (3 parallel LLM threads)
│                          #   Dividend CAGR / payout from dividends_history table
├── config.py              # Centralised env-var config (DeepSeek, Ollama, PG, Neo4j)
├── tools.py               # PostgreSQL query helpers, FMToolkit, FMDataFetcher
│                          #   EODHD ↔ FMP field name bidirectional aliases
│                          #   _map_annual_income/balance/cashflow helper functions
│                          #   ALLOWED_DATA_TYPES whitelist enforcement
├── prompts.py             # System prompt + quantitative summary prompt
├── schema.py              # Dataclasses: FMDataBundle, DCFResult, CompsResult,
│                          #   TechnicalSnapshot, EarningsRecord, DividendRecord,
│                          #   FactorScores, ValuationResult
├── models/
│   ├── __init__.py        # Exports: ThreeStatementEngine, ThreeStatementModel
│   ├── dcf.py             # DCFEngine: WACC, CAPM, 3-stage FCF, Terminal Value,
│   │                      #   scenario table, sensitivity matrix, reverse DCF
│   │                      #   vix_mrp_adjustment(), compute_benchmark_hv30()
│   ├── valuation.py       # CompsEngine: peer multiples (EV/EBITDA, P/E, P/S, etc.)
│   ├── technicals.py      # TechnicalEngine: RSI, MACD, Bollinger, ATR, HV30,
│   │                      #   SMA/EMA, Stochastic, rolling beta
│   └── three_statement.py # ThreeStatementEngine: linked IS+BS+CF with linkage checks
│                          #   RE linkage (includes buybacks), cash linkage, BS balance
└── tests/
    └── test_agent.py      # Unit + integration tests (mocked)
```

---

## Public API

```python
from agents.financial_modelling import run, run_full_analysis

# Single ticker
report = run_full_analysis(ticker="AAPL")
report["valuation"]["dcf"]["intrinsic_value_base"]   # → float
report["technicals"]["rsi_14"]                        # → float
report["earnings"]["beat_streak"]                     # → int
report["three_statement_model"]["linkage_checks"]     # → list

# Natural-language prompt (ticker extracted automatically)
report = run(prompt="Analyze NVDA valuation")

# Multiple tickers
reports = run(prompt="Compare MSFT vs AAPL")  # → List[Dict]
```

---

## Environment Variables

```bash
# LLM — DeepSeek API (primary)
FM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
FM_LLM_MODEL=deepseek-reasoner

# LLM — Ollama (fallback if DeepSeek unreachable)
OLLAMA_BASE_URL=http://localhost:11434

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# Neo4j (peer group selection for Comps)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# Analysis configuration
PRICE_HISTORY_DAYS=365          # Look-back window for technical analysis
DCF_FORECAST_YEARS=5            # Number of years to project FCF
DCF_TERMINAL_GROWTH_RATE=0.025  # Terminal growth rate default
DCF_WACC=0.09                   # WACC default (overridden by live CAPM calculation)
COMPS_SECTOR_PEERS=5            # Number of peer companies for Comps analysis
```

---

## Run Commands

```bash
# Run inside the Airflow scheduler container (standard)
docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker AAPL --log-level INFO

# All 5 validated tickers
docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker AAPL --log-level WARNING

docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker MSFT --log-level WARNING

docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker GOOGL --log-level WARNING

docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker TSLA --log-level WARNING

docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --ticker NVDA --log-level WARNING

# Natural-language prompt mode
docker exec fyp-airflow-scheduler \
    python -m agents.financial_modelling.agent --prompt "Analyze AAPL valuation" --log-level WARNING

# Unit test suite
python -m pytest agents/financial_modelling/tests/ -q
```

---

## Data Boundary Enforcement

The agent is restricted to the financial modelling use-case only. All data queries are validated against `ALLOWED_DATA_TYPES` in `tools.py`.

### Allowed Data Types (whitelist)
- Basic Fundamentals (EPS, key metrics, ratios)
- Financial Statements (IS / BS / CF)
- Valuation Metrics
- Outstanding Shares History
- Dividend History / Stock Splits
- Earnings Surprises
- Treasury Rates / Macro Indicators
- Economic Events
- Bonds Data (Yields)
- Forex Historical Rates (EOD)
- Historical Stock Prices (EOD — used for beta/WACC and technicals)

### Excluded Data Types (reserved for other agents)
- Intraday / Delayed Live Quotes
- Screener API (Bulk)
- Short Interest & Shares Stats
- Any Business Analyst data (news, sentiment, insiders, etc.)

### Implementation
- `ALLOWED_DATA_TYPES` whitelist in `tools.py`
- `validate_data_name()` enforces the whitelist on every `raw_fundamentals` query
- **No direct external API calls** — all data reads go through PostgreSQL

---

## Design Decisions

- **Numbers are never LLM-generated.** All numeric outputs (DCF intrinsic value, RSI, EV/EBITDA, WACC, F-Score, etc.) are computed in Python. The LLM writes only narrative strings.
- **EODHD field name aliases.** EODHD uses different field names from FMP throughout IS/BS/CF payloads. `tools.py` applies bidirectional aliases so downstream models can use consistent FMP-style keys (`revenue`, `operatingCashFlow`, etc.) regardless of the originating API.
- **Annual vs. quarterly data split.** `bundle.income/balance/cashflow` carries the most recent DB row (may be quarterly). `bundle.income_annual/balance_annual/cashflow_annual` always carries the most recent `period_type='yearly'` row. The 3-Statement Model uses annual data for comparability.
- **RE linkage formula includes buybacks.** `ΔRE ≈ NI − Dividends + salePurchaseOfStock`. Buybacks are negative (cash outflow), which reduces retained earnings for repurchase-heavy companies like AAPL.
- **Linkage tolerances are wide by design.** `±$5B absolute or ±10% relative` is needed because RE linkage legitimately differs from the simplified formula due to AOCI, SBC vesting, and other equity movements not captured in the three-statement stub. TSLA's cash definition differs by ~$1.1B between BS and CF statement (EODHD convention); $5B tolerance handles this.
- **VIX-adjusted MRP.** WACC uses a live Market Risk Premium that is uplifted in high-volatility regimes (HV30 > 25%) and reduced in calm regimes (HV30 < 15%). This prevents WACC from being artificially low during market stress.
- **MoE DCF consensus.** Three parallel LLM personas adjust TGR and WACC assumptions and produce a probability-weighted Bear/Base/Bull range. This gives the Supervisor a distribution of outcomes, not just a point estimate.
- **`deepseek-chat` fallback for MoE narratives.** `deepseek-reasoner` uses chain-of-thought `<think>` tags that consume tokens before emitting the actual response. If the cleaned response is fewer than 20 characters, the agent retries with `deepseek-chat` (no reasoning overhead) for the persona narrative only.
- **WACC is live, not hardcoded.** `DCF_WACC` env var is a fallback only. At runtime, WACC is computed from live CAPM inputs: 10Y Treasury yield (from `treasury_rates`), VIX-adjusted MRP, rolling 60-day beta (from `market_eod_us`), and cost of debt from `raw_fundamentals`.

---

*Last updated: 2026-03-10 | Author: hck717*
