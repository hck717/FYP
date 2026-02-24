# 🌏 Macro Metrics Agent

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。

---

## 🎯 Agent 職能

Macro Metrics Agent 負責分析 **完觀經濟環境**，評估宏觀因素對個別股票同整個市場嘅影響。呢個 agent 將完觀分析結果提供給 Business Analyst Agent 同 Financial Modelling Agent 作為背景 context。

```
用戶問題
    │
    ▼
[Macro Metrics Agent]
    │
    ├── 查詢 PostgreSQL (global_economic_calendar)
    ├── 調用 EODHD economic-events API
    ├── 分析利率、通豗、GDP 趨勢
    ├── 評估 sector 層面嘅 macro exposure
    └── 回傳 macro context 給 Supervisor
```

---

## 📦 負責範圍

### 適用場景

- 美聯儲 FOMC 決議對技術股嘅影響分析
- CPI / PCE 通豗數據解讀（對消費股嘅影響）
- GDP 增長 vs. 公司收入增長對比分析
- 就業數據 / PMI / consumer sentiment 趨勢
- 完觀日曆展望（未來 14 日嘅重要公告）

### 不適用場景（由其他 agent 處理）

- 公司層面新聞 → Web Search Agent
- 個別股票價格分析 → Financial Modelling Agent
- 公司財務報告 → Business Analyst Agent

---

## 🛠️ 技術實現

### 主要 Data Sources

| 圖表 / API | 內容 | 存僸位置 |
|---|---|---|
| `global_economic_calendar` | 經濟事件（FOMC、CPI、GDP、就業數据） | PostgreSQL |
| `global_ipo_calendar` | IPO 日曆 | PostgreSQL |
| `raw_timeseries` | 市場指數歷史價格 | PostgreSQL |
| EODHD `economic-events` API | 實時完觀事件 | 直接 API 呼叫 |
| EODHD `macro-indicators` API | 完觀指標歷史資料 | 直接 API 呼叫 |

### LLM 設定

```python
# 完觀分析需要較強嘅推理能力
llm = ChatOllama(
    model="deepseek-r1:8b",      # 適合長文本分析與推理
    base_url="http://localhost:11434",
    temperature=0.2,
)
```

### Agent 工作流（LangGraph）

```
START
  │
  ▼
fetch_economic_calendar     ←  查詢 PostgreSQL 最近 30 天嘅完觀事件
  │
  ▼
fetch_macro_indicators      ←  調用 EODHD API 抓取 CPI、GDP、利率數據
  │
  ▼
analyse_rate_environment    ←  利率半山利率曲線分析
  │
  ▼
assess_sector_impact        ←  分析完觀因素對查詢 ticker 嘅 sector 影響
  │
  ▼
forecast_upcoming_events    ←  列出未來 14 日内的重要事件
  │
  ▼
format_output               ←  整理成 structured macro context
  │
 END → 回傳 Supervisor
```

### 分析指標清單

**貨幣政策類：**
- Federal Funds Rate 、ECB Rate 、HKMA Base Rate
- 利率曲線 (yield curve)、倍期差異 (term spread)

**通豗類：**
- CPI YoY 、PPI YoY 、PCE
- 實賽定點 vs. 市場預期的差異（surprise factor）

**增長類：**
- GDP QoQ 、Real GDP Growth
- ISM Manufacturing PMI 、Services PMI
- Nonfarm Payrolls 、Unemployment Rate

### Output 格式

```json
{
  "agent": "macro_metrics",
  "as_of_date": "2026-02-24",
  "rate_environment": {
    "fed_funds_rate": 4.25,
    "trend": "easing",
    "next_fomc": "2026-03-18",
    "market_implied_cuts_2026": 2
  },
  "inflation": {
    "cpi_yoy": 2.8,
    "trend": "declining",
    "vs_expectation": "in-line"
  },
  "sector_impact": {
    "Technology": "neutral-positive",
    "Consumer Cyclical": "cautious",
    "Financials": "positive"
  },
  "upcoming_events": [
    {"date": "2026-03-05", "event": "US Nonfarm Payrolls", "importance": "high"}
  ],
  "macro_summary": "當前貨幣政策屬測寬横平制居間..."
}
```

---

## 📁 目錄結構

```
agents/macro_metrics/
├── README.md          # 本文件
├── agent.py           # Agent 主體逻輯（LangGraph）
├── tools.py           # PostgreSQL 查詢、EODHD API 封裝
├── indicators.py      # 完觀指標計算與解讀逻輯
├── prompts.py         # Macro 分析 system prompt
└── tests/
    └── test_agent.py
```

---

## ⚙️ 環境變數

```bash
# .env 檔案加入
EODHD_API_KEY=your_eodhd_api_key
OLLAMA_BASE_URL=http://localhost:11434
MACRO_MODEL=deepseek-r1:8b

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# 展望窗口
MACRO_LOOKBACK_DAYS=90     # 回望分析 90 天
MACRO_FORWARD_DAYS=14      # 展望未來 14 天事件
```

---

## 🧪 快速測試

```bash
source .venv/bin/activate

# 查詢完觀日曆
python agents/macro_metrics/agent.py --mode calendar

# 分析完觀環境對特定 sector 嘅影響
python agents/macro_metrics/agent.py --sector Technology

# 執行測試
python -m pytest agents/macro_metrics/tests/ -v
```

---

## 📝 設計容訊

- **決策語氣**：將完觀數據翻譯成對個別公司嘅定性影響評估
- **多地區套繖**：考慮美國、歐洲、香港嘅完觀進度分歧
- **Surprise factor 計算**：實際公布數據 vs. 市場預期的差異對陰用嘅賭凰識別
- **缓存機制**：完觀數據一天只發一次，PostgreSQL cache 分析結果

---

*最後更新：2026-02-24 | 作者：hck717*
