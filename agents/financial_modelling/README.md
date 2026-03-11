# Financial Modelling Agent

> **Status:** Complete (live-tested, all 5 tickers validated with local docker-compose databases)
> **Role in system:** Deterministic quantitative valuation engine — all numbers are computed, never LLM-generated

---

## Role

The Financial Modelling Agent is the **quantitative valuation layer** of the multi-agent equity research system. It reads historical prices, fundamentals, and earnings data exclusively from PostgreSQL, executes deterministic calculations (DCF, WACC, Comps, 3-Statement Model, Factor Scores), and returns a fully computed valuation package to the Supervisor.

It does **NOT** generate numbers via LLM. The LLM is only used to interpret the computed outputs and write the `quantitative_summary` narrative, `executive_summary`, `investment_recommendation`, and MoE persona narratives. Every numeric output has a traceable formula.

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
| `treasury_rates_dedicated` | `treasury_rates` PG table (dedicated ingestion) | US10Y → risk-free rate (primary source) |
| `treasury_rates` | `raw_timeseries` lookup for treasury data | 10Y yield fallback (often empty in practice) |

---

## Models Implemented

### Discounted Cash Flow (DCF) — `models/dcf.py`

**WACC (live, not hardcoded):**

```
WACC = (E/V) × Re + (D/V) × Rd × (1 − T)

Re  = Rf + β × (Rm − Rf)          [CAPM]
Rf  = US10Y yield from treasury_rates table (treasury_rates_dedicated field)
      → fallback: raw_timeseries treasury rows → hardcoded 4.3%
β   = Regression vs. S&P 500 benchmark (all available daily rows, ~400)
      → Damodaran sector beta fallback if no price history
Rm−Rf = Base MRP 4.5% (Damodaran January 2026 implied US ERP)
        adjusted by HV30 regime (VIX-proxy from macro_environment node)
Rd  = Interest Expense / Total Debt  [annual IS preferred; EBIT−IBT gap fallback]
      → 3.5% fallback for companies where interest is not separately reported
T   = Effective tax rate from annual IS (annual preferred over quarterly IS)
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

**Terminal Growth Rate — Sector-Adjusted (`_sector_terminal_growth` in `dcf.py`):**

The terminal growth rate is no longer a flat global default. It is resolved from the company's sector classification, falling back to `config.terminal_growth_rate` (default 3.0%) for unknown sectors:

| Sector | Terminal Growth | Rationale |
|---|---|---|
| Software / SaaS | 3.5% | Pricing power + recurring revenue |
| Semiconductors | 3.3% | AI/data-center secular tailwinds |
| Information Technology | 3.2% | Broad tech; hardware commoditises faster |
| Communication Services | 3.0% | Digital-growth tailwind, mature base |
| Health Care / Biotech | 3.0% | Demographics-driven; regulatory risk offset |
| Consumer Discretionary | 2.8% | Cyclical; margins mean-revert |
| Consumer Staples / Food | 2.7% | Defensive, inflation pass-through |
| Financials | 2.7% | GDP-linked, rate-cycle sensitive |
| Industrials / Aerospace | 2.7% | GDP-linked, capex-intensive |
| Energy / Oil & Gas | 2.5% | Commodity-linked; energy transition headwind |
| Materials / Mining | 2.5% | Commodity-linked |
| Real Estate / REIT | 2.5% | Rent inflation-linked |
| Utilities | 2.3% | Regulated; rate-case dependent |
| Unknown / Default | 3.0% | Fallback |

> No ticker-specific values. Rate is derived purely from sector classification.

**Stage 2 ROIC Convergence — Data-Driven Tiers (`_three_stage_pv` in `dcf.py`):**

In Stage 2 (years 11–20), ROIC fades from the current computed value to a stable terminal ROIC. The terminal ROIC premium over WACC is determined by the company's *demonstrated* ROIC level — not by ticker name:

| Computed ROIC | stable_roic | Interpretation |
|---|---|---|
| > 30% | WACC + 9% | Exceptional moat (e.g. AAPL at 38.9%) |
| 20–30% | WACC + 7% | Strong moat |
| 10–20% | WACC + 4% | Moderate moat (e.g. MSFT at ~16.5%) |
| < 10% | WACC + 2% | Thin moat / commodity business |

**Base Growth Rate — Priority Chain (`_derive_base_growth` in `dcf.py`):**

The base revenue growth rate used across scenarios is resolved by the following priority chain (first non-null value wins):

| Priority | Source | Notes |
|---|---|---|
| 1 | Analyst consensus mean revenue growth | From `bundle.analyst_estimates` |
| 2 | `revenueGrowthTTM` (key_metrics_ttm) | TTM revenue growth; excluded if null or zero |
| 3 | Annual IS YoY: `income_annual.revenue / income_prior.revenue − 1` | Clean fiscal-year-over-fiscal-year comparison (e.g. AAPL FY2025 vs FY2024 = 6.4%) |
| 3b | TTM / prior-annual fallback: `RevenueTTM / income_annual.revenue − 1` | Only if annual IS pair unavailable |
| 4 | `revenueGrowthAnnual` / `revenue_growth_annual` (key_metrics) | From pre-computed key metrics |
| 5 | Sector default | Tech 12%, Health 8%, Energy 4%, otherwise 6% |

> **Intentionally excluded:** `QuarterlyRevenueGrowthYOY` — this is a single-quarter YoY metric (e.g. Dec-2025 vs Dec-2024), not an annual growth rate. Using it would overstate or misrepresent the company's trend-line growth.

**Scenario Engine (data-driven, not hardcoded):**

Scenarios apply deltas to the data-derived `base_growth`:

| Scenario | Revenue Growth | EBIT Margin Multiplier | WACC Offset |
|---|---|---|---|
| Bear | `base_growth − 7pp` (floor: −3%) | 0.70× actual TTM margin | +1.5% |
| Base | `base_growth` | 1.00× actual TTM margin | 0% |
| Bull | `base_growth + 8pp` (cap: 35%) | 1.15× actual TTM margin | −1.5% |

Example for AAPL (base_growth = 6.4%, TTM EBIT margin = 35.4%, WACC = 9.35%):

| Scenario | Revenue Growth | EBIT Margin | WACC |
|---|---|---|---|
| Bear | −0.6% | 24.8% | 10.85% |
| Base | 6.4% | 35.4% | 9.35% |
| Bull | 14.4% | 40.7% | 7.85% |

**Sensitivity Matrix:** WACC (4 levels) × Terminal Growth Rate (4 levels) → implied intrinsic value grid.

**Reverse DCF:** Independently solves "what revenue CAGR justifies the current market price?" using the same 3-stage model. A high reverse CAGR (e.g. ~35% for AAPL) is a valid analytical signal indicating the market prices in growth well above the DCF base case — not a bug.

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

**Dual LLM calls in `format_json_output` node:**

| Call | Model | Output field | Notes |
|---|---|---|---|
| `_generate_summary_deepseek()` | `deepseek-reasoner` | `quantitative_summary` | Chain-of-thought reasoning over the full numeric context; `reasoning_content` fallback if `content` is empty (see Design Decisions) |
| `_generate_executive_block()` | `deepseek-chat` | `executive_summary` + `investment_recommendation` | Lighter model for structured prose; avoids `deepseek-reasoner` token overhead for short outputs |

**LLM scope is strictly limited:**
- `quantitative_summary` — 10–15 sentence interpretation of the pre-computed factor table
- `executive_summary` — 3–5 sentence plain-English summary of the overall investment picture
- `investment_recommendation` — one of: Strong Buy / Buy / Hold / Sell / Strong Sell, with a 1–2 sentence rationale
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
      "intrinsic_value_base": 145.79,
      "intrinsic_value_bear": 62.48,
      "intrinsic_value_bull": 313.96,
      "intrinsic_value_weighted": 158.60,
      "upside_pct_base": -44.1,
      "wacc_used": 0.09354,
      "terminal_growth_rate": 0.03,
      "beta_used": 1.1782,
      "pv_stage1": 962900000000,
      "pv_stage2": 558700000000,
      "pv_terminal": 677000000000,
      "equity_value_base": 2167200000000,
      "validation_warnings": [],
      "scenario_table": [...],
      "sensitivity_matrix": {...},
      "reverse_dcf_implied_cagr": 0.002
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
    "risk_free_rate_10y": 0.04133,
    "benchmark_hv30": 0.182,
    "vix_regime": "normal",
    "base_mrp": 0.045,
    "mrp_delta": 0.0,
    "vix_adjusted_mrp": 0.045
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
  "executive_summary": "Apple demonstrates strong capital returns with a Piotroski F-Score of 7 and a ROIC-WACC spread of +28pp...",
  "investment_recommendation": "Hold — DCF intrinsic value of $110 sits ~58% below the current market price of $261, implying the market prices in a 35% revenue CAGR that is unlikely to materialise at scale.",
  "data_sources": {
    "price_data": "postgresql:raw_timeseries",
    "fundamentals": "postgresql:raw_fundamentals",
    "peer_group": "neo4j:COMPETES_WITH",
    "treasury_rates": "postgresql:raw_timeseries (FMP treasury endpoint)",
    "market_risk_premium": "postgresql:raw_fundamentals (FMP market risk premium endpoint)",
    "llm_scope": "quantitative_summary, executive_summary, investment_recommendation narratives only",
    "llm_model_quantitative_summary": "deepseek-reasoner",
    "llm_model_executive_block": "deepseek-chat"
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
- **`deepseek-reasoner` empty `content` fix — `reasoning_content` fallback.** The DeepSeek API sometimes returns a response object where `message.content` is an empty string but `message.reasoning_content` contains the actual answer. `_generate_summary_deepseek()` in `agent.py` detects this case and falls back to `reasoning_content` before treating the response as a failure. Without this fix, `quantitative_summary` would be empty for any call where the model front-loads all output into the reasoning trace.
- **WACC is live, not hardcoded.** `DCF_WACC` env var is a fallback only. At runtime, WACC is computed from live CAPM inputs: 10Y Treasury yield (from `treasury_rates`), VIX-adjusted MRP, rolling 60-day beta (from `market_eod_us`), and cost of debt from `raw_fundamentals`.
- **DCF Fix 1 — Quarterly IS detection for tax rate.** EODHD's most-recent `income_statement` row is often the latest quarterly filing (e.g. `period_type='quarterly'`, Dec-2025). Deriving the effective tax rate from a quarterly IS overstates it (single-quarter tax provision / single-quarter pre-tax income). `DCFEngine.compute()` detects `_period_type in ("quarterly", "q")` and uses `bundle.income_annual` (the most recent full-year IS) for tax rate computation instead. For AAPL this corrects the rate from ~17.4% (quarterly) to ~15.6% (annual).
- **DCF Fix 2 — Annual capex preference over quarterly capex.** EODHD's `capitalExpenditures` (plural) in the quarterly cashflow row maps to `capitalExpenditure` (singular) in the bundle dict. Because both keys can be populated, the previous `if not bundle.cashflow.get("capitalExpenditure")` overlay guard was never triggered, leaving quarterly capex ($2.4B for AAPL) in place instead of the annual figure ($12.7B). The fix introduces a `_cf_get()` helper inside `DCFEngine.compute()` that unconditionally prefers `bundle.cashflow_annual` fields when the primary cashflow row is quarterly, and falls back to the primary row plus an optional ×4 annualisation heuristic only when `cashflow_annual` is also unavailable.
- **DCF Fix 3 — Annual IS YoY as base growth rate.** The growth rate was previously sourced from `QuarterlyRevenueGrowthYOY` (a single-quarter YoY metric from `key_metrics_ttm`), which reflects one quarter's anomalous performance and is not representative of trend-line revenue growth. The fix replaces this with a clean fiscal-year-over-fiscal-year comparison using `bundle.income_annual.revenue / bundle.income_prior.revenue − 1` (Priority 3 in the growth chain). For AAPL this gives 6.4% (FY2025 $416.2B / FY2024 $391.0B − 1); for MSFT it gives 14.9% (FY2025 $281.7B / FY2024 $245.1B − 1), correctly preserving MSFT's higher growth profile.
- **WACC Fix 1 — treasury_rates_dedicated as primary rf source.** `bundle.treasury_rates` (from `raw_timeseries` lookup) is unreliable and often empty. `bundle.treasury_rates_dedicated` (from the dedicated `treasury_rates` PostgreSQL table) always has data. `_compute_wacc()` now reads `treasury_rates_dedicated` first. For AAPL this corrects rf from the 4.3% hardcoded fallback to the real US10Y = 4.133%.
- **WACC Fix 2 — MRP default updated to Damodaran January 2026 ERP (4.5%).** The previous default of 5.5% is a textbook approximation from the 1990s, not a current market estimate. Damodaran's implied ERP for the US market as of January 2026 is 4.5%, reflecting elevated market valuations and low volatility. For AAPL this reduces re from ~11.1% to ~9.4%.
- **WACC Fix 3 — Beta lookback uses all available daily rows.** The DB holds ~400 daily price rows (~18 months). The beta regression now passes `len(bundle.price_history)` as the lookback, ensuring every available observation is used rather than truncating at the legacy 104-week default (which had the same practical effect but is conceptually cleaner).
- **WACC Fix 4 — Annual IS for rd derivation; 3.5% fallback.** Interest expense is now sourced from `bundle.income_annual` (avoids single-quarter noise). For companies where EODHD nets interest income against expense (e.g. AAPL), the EBIT−IBT gap is used as a last resort before the 3.5% investment-grade fallback (down from 4.0%).
- **Sector-adjusted terminal growth rate.** The terminal growth rate `g` is no longer a flat global default. `_sector_terminal_growth()` maps the company's GICS sector to an analytically grounded TGR (range: 2.3% utilities → 3.5% software), grounded in Damodaran's long-run sector assumptions and a 2% real GDP + 2% inflation baseline. No ticker-specific values.
- **ROIC-tier stable_roic in Stage 2.** The stable ROIC used for Stage 2 convergence is now determined by the company's *demonstrated* ROIC level, not a flat offset. Companies with ROIC > 30% earn a WACC+9% stable premium; 20–30% earns WACC+7%; 10–20% earns WACC+4%; below 10% earns WACC+2%. This correctly distinguishes AAPL (38.9% ROIC, exceptional moat) from MSFT (16.5% ROIC, moderate moat for a capex-intensive cloud business). No ticker-specific hardcoding.
- **Reinvestment rate cap (capex-implied ceiling).** `_project_fcf_roic()` caps the ROIC-derived reinvestment rate at `capex_pct / nopat_margin`. For capital-light companies (AAPL: capex only 2.9% of revenue), the ROIC model could overstate reinvestment (g/ROIC = 6.4%/38.9% = 16.4%), producing too little FCF. The cap prevents this while correctly leaving capital-intensive companies (MSFT: capex 21.1% of revenue) unconstrained. R&D is intentionally *not* included in the cap because GAAP-expensed R&D is already subtracted from EBIT — adding it again would double-count it.
- **DCF breakdown fields on DCFResult.** `pv_stage1`, `pv_stage2`, `pv_terminal`, and `equity_value_base` are now stored on the result object and logged at DEBUG level. This makes the PV decomposition fully traceable without re-running the model. Terminal Value as % of total EV is also logged (e.g. AAPL: TV = 30.8% of EV, well-behaved).
- **Input validation warnings.** `_validate_dcf_inputs()` checks for anomalous inputs (very low revenue, extreme margins, high capex pct, extreme growth, high D/E) and stores human-readable warnings in `DCFResult.validation_warnings`. Warnings are non-fatal and also logged at WARNING level for production monitoring.

---

*Last updated: 2026-03-11 | Author: hck717*
