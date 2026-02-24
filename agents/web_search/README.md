# 🔍 Web Search Agent

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。

---

## 🎯 Agent 職能

Web Search Agent 負責搜尋 **實時、最新嘅網絡資訊**，展作其他 agent 唔能從静態 database 獲得嘅資訊。主要用途包括最新盈利公告、監管消息、繼任 CEO 老闆雙、突發事件等。

```
用戶問題
    │
    ▼
[Web Search Agent]
    │
    ├── 生成 search queries
    ├── 呼叫 Tavily / DuckDuckGo Search tool
    ├── 筋選高相關結果
    ├── 提取主要內容與來源
    └── 回傳 structured 結果給 Supervisor
```

---

## 📦 負責範圍

### 適用場景

- 最新盈利報告、沪期上市 / 退市公告
- 監管機構對公司嘅調查、罰款、訴訟
- 產品發布會、策略轉變、重大并購
- 主要管理層變動（CEO、CFO 老闆雙）
- EODHD 數據庫尚未更新嘅突發事件

### 不適用場景（由其他 agent 處理）

- 歷史股價數據 → Financial Modelling Agent
- 公司財務報告基本面 → Business Analyst Agent
- GDP、CPI 等完年資料 → Macro Metrics Agent

---

## 🛠️ 技術實現

### Tools

| Tool | 用途 | 實現 |
|---|---|---|
| `TavilySearchResults` | 主要搜尋工具，返回結構化結果 | LangChain `langchain-community` |
| `DuckDuckGoSearchRun` | Fallback 搜尋（唔需要 API key） | LangChain `langchain-community` |
| `WebBaseLoader` | 抓取指定 URL 嘅全文內容 | LangChain document loaders |

### LLM 設定

```python
# 使用 Ollama 本地 LLM
llm = ChatOllama(
    model="qwen2.5:7b",          # 主要推理 model
    base_url="http://localhost:11434",
    temperature=0.1,             # 保持較低，減少虜構
)
```

### Agent 工作流（LangGraph）

```
START
  │
  ▼
plan_queries        ←  LLM 將用戶問題分解為 2–3 個 search queries
  │
  ▼
execute_search      ←  並行呼叫 Tavily，每個 query 最多取 5 個結果
  │
  ▼
filter_results      ←  去除重複、邎期文章（> 7 日）、低相關度結果
  │
  ▼
format_output       ←  整理成 structured JSON 和 markdown 摘要
  │
  ▼
 END → 回傳 Supervisor
```

### Output 格式

```json
{
  "agent": "web_search",
  "ticker": "AAPL",
  "query_used": "Apple Inc latest news 2026",
  "results": [
    {
      "title": "Apple Reports Record Q1 2026 Revenue",
      "url": "https://...",
      "snippet": "Apple Inc. today announced...",
      "published_date": "2026-02-20",
      "relevance_score": 0.92
    }
  ],
  "summary": "Apple 報告 2026 Q1 创紀錄盈利..."
}
```

---

## 📁 目錄結構

```
agents/web_search/
├── README.md          # 本文件
├── agent.py           # Agent 主體逻輯（LangGraph）
├── tools.py           # Tavily / DuckDuckGo 封裝
├── prompts.py         # System prompt 和 query 生成 template
└── tests/
    └── test_agent.py     # 單元測試
```

---

## ⚙️ 環境變數

```bash
# .env 檔案加入
TAVILY_API_KEY=your_tavily_api_key     # 從 https://tavily.com 獲得
OLLAMA_BASE_URL=http://localhost:11434
WEB_SEARCH_MODEL=qwen2.5:7b
WEB_SEARCH_MAX_RESULTS=5               # 每個 query 返回最多結果數
WEB_SEARCH_MAX_AGE_DAYS=7              # 邎期文章過濾間隔
```

---

## 🧪 快速測試

```bash
# 啟動 virtual environment
source .venv/bin/activate

# 直接執行 agent
python agents/web_search/agent.py --ticker AAPL --query "latest news"

# 執行測試
python -m pytest agents/web_search/tests/ -v
```

---

## 📝 設計容訊

- **Stateless** 設計：每次呼叫都是独立的，唔依賴 session state
- **Fallback chain**：Tavily 失敗→ DuckDuckGo 备用
- **Hallucination guard**：所有搜尋結果必須附上原始 URL，由 Critic Agent 核實
- **Rate limiting**：預設每次呼叫間隔 1 秒，防止被封 IP

---

*最後更新：2026-02-24 | 作者：hck717*
