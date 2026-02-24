# 🧩 Financial Modelling Agent

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。

---

## 🎯 Agent 職能

Financial Modelling Agent 負責 **定量金融分析同估值模型**。呢個 agent 直接從 PostgreSQL 讀取歷史價格、基本面、盈利數據，執行定量計算，生成如 DCF、P/E、P/S 等值的估值分析。

```
用戶問題
    │
    ▼
[Financial Modelling Agent]
    │
    ├── PostgreSQL 抓取歷史價格 + 基本面
    ├── 計算技術指標 (SMA, EMA, RSI, MACD)
    ├── 建構估值模型 (DCF, comparable)
    ├── 分析師預測 vs. 實際表現
    └── 回傳 quantitative analysis 給 Supervisor
```

---

## 📦 負責範圍

### 適用場景

- DCF (Discounted Cash Flow) 内在價值估算
- Comparable company analysis (P/E、EV/EBITDA 對比)
- 技術分析（trend、momentum、support / resistance）
- 盈利預測客觀評估（analyst estimate vs. actual）
- 股息 / 賢回歷史分析
- 樹股 (stock split) 對歷史價格嘅影響調整

### 不適用場景（由其他 agent 處理）

- 新聞情緒 → Business Analyst Agent
- 宏觀利率 / 通豗 → Macro Metrics Agent
- 最新事件 → Web Search Agent

---

## 🛠️ 技術實現

### 主要 Data Sources

| 圖表 | 內容 | 存僸位置 |
|---|---|---|
| `raw_timeseries` | EOD、intraday、技術指標、SMA、EMA | PostgreSQL |
| `raw_fundamentals` | PE、EPS、EBITDA、市値 | PostgreSQL |
| `market_eod_us` | 美股全局 EOD | PostgreSQL |
| `historical_prices_weekly/monthly` | 週線 / 月線 | PostgreSQL |
| `earnings_history` | 盈利分析 | PostgreSQL |
| `dividends_history` | 股息歷史 | PostgreSQL |
| `splits_history` | 樹股歷史 | PostgreSQL |

### LLM 設定

```python
# 金融模型需要數字推理能力
llm = ChatOllama(
    model="deepseek-r1:8b",      # DeepSeek 隱層思考適合數字推理
    base_url="http://localhost:11434",
    temperature=0.1,             # 數字分析用低 temperature
)
```

### Agent 工作流（LangGraph）

```
START
  │
  ▼
fetch_price_history         ←  PostgreSQL 抓取 1 年歷史 EOD 價格
  │
  ▼
fetch_fundamentals          ←  PostgreSQL 抓取最新 PE、EPS、EBITDA
  │
  ▼
fetch_earnings_history      ←  PostgreSQL 抓取盈利展望同實際比較
  │
  ▼
calculate_technicals        ←  計算 RSI、MACD、Bollinger Bands、SMA、EMA
  │
  ▼
run_dcf_model               ←  基於 free cash flow + WACC 估算內在價值
  │
  ▼
run_comparable_analysis     ←  P/E、P/S、EV/EBITDA 對比行業平均
  │
  ▼
assess_analyst_estimates    ←  分析師預測跟蹤性同 surprise 歷史
  │
  ▼
format_output               ←  全面定量報告，附上所有計算來源
  │
 END → 回傳 Supervisor
```

### 技術指標清單

**Trend Indicators：**
- SMA 20 / SMA 50 / SMA 200
- EMA 12 / EMA 26
- Golden cross / Death cross 識別

**Momentum Indicators：**
- RSI (14)：> 70 超買，< 30 超賣
- MACD 和 Signal line 差異
- Stochastic Oscillator

**Volatility：**
- Bollinger Bands (20, 2σ)
- ATR (Average True Range)
- 年化歷史波幅率 (HV 30)

**估值指標：**
- P/E (trailing + forward)
- P/S (Price / Sales TTM)
- EV/EBITDA
- PEG Ratio (P/E ÷ EPS growth)
- DCF 內在價值 vs. 市場價格

### Output 格式

```json
{
  "agent": "financial_modelling",
  "ticker": "AAPL",
  "as_of_date": "2026-02-24",
  "current_price": 218.50,
  "valuation": {
    "dcf_intrinsic_value": 195.20,
    "pe_trailing": 31.2,
    "pe_forward": 27.8,
    "ev_ebitda": 22.4,
    "vs_sector_avg": "premium +18%"
  },
  "technicals": {
    "trend": "bullish",
    "rsi_14": 58.3,
    "macd_signal": "buy",
    "support": 205.0,
    "resistance": 230.0,
    "sma_50_above_200": true
  },
  "earnings": {
    "last_eps_actual": 2.40,
    "last_eps_estimate": 2.35,
    "surprise_pct": 2.1,
    "beat_streak": 6
  },
  "quantitative_summary": "Apple 目前技術劢頭尚存， RSI 中性..."
}
```

---

## 📁 目錄結構

```
agents/financial_modelling/
├── README.md              # 本文件
├── agent.py               # Agent 主體逻輯（LangGraph）
├── tools.py               # PostgreSQL 查詢封裝
├── models/
│   ├── dcf.py             # DCF 內在價值計算
│   ├── valuation.py       # P/E、EV/EBITDA comparable
│   └── technicals.py      # RSI、MACD、Bollinger 計算
├── prompts.py             # Quantitative analysis system prompts
└── tests/
    └── test_agent.py
```

---

## ⚙️ 環境變數

```bash
# .env 檔案加入
OLLAMA_BASE_URL=http://localhost:11434
FINANCIAL_MODEL=deepseek-r1:8b

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# 分析設定
PRICE_HISTORY_DAYS=365          # 技術分析回望天數
DCF_FORECAST_YEARS=5            # DCF 預測年度
DCF_TERMINAL_GROWTH_RATE=0.025  # DCF 終値增長率
DCF_WACC=0.09                   # WACC 假設値（延續至獲取實際市場資料）
COMPS_SECTOR_PEERS=5            # Comparable 比較公司數目
```

---

## 🧪 快速測試

```bash
source .venv/bin/activate

# 分析指定股票
python agents/financial_modelling/agent.py --ticker AAPL

# 僅執行技術分析
python agents/financial_modelling/models/technicals.py --ticker TSLA

# 僅執行 DCF 模型
python agents/financial_modelling/models/dcf.py --ticker MSFT

# 執行測試
python -m pytest agents/financial_modelling/tests/ -v
```

---

## 📝 設計容訊

- **數字主導**：所有分析基於真實數字，LLM 負責解讀和結論生成，不改变數字
- **WACC 參數化**：DCF 的 WACC 可配置，後期隨市場利率自動更新
- **Split-adjusted**：樹股歷史將自動調整歷史價格，積成計算正確
- **Peer comparison**：COMPS 基於 Neo4j 查詢相同 sector / industry 的對標公司

---

*最後更新：2026-02-24 | 作者：hck717*
