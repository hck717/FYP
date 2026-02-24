# 🔍 Web Search Agent (Poe / DeepSeek)

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。  
> **Status:** Production-ready skeleton (Supervisor-callable, JSON output, tests).  
> **Model:** `deepseek-v3.2-exp` via Poe API (primary).  

---

## 🎯 Agent 職能

Web Search Agent 負責搜尋 **實時、最新嘅網絡資訊**，用嚟補足其他 agents 從本地靜態 knowledge base（PostgreSQL / Qdrant / Neo4j）攞唔到嘅 breaking updates，例如盈利速報、監管消息、訴訟、CEO/CFO 變動、重大 M&A、突發事故等。

本 Agent 係 **stateless**：每次呼叫獨立，輸出一個可被 Supervisor 直接 ingest 嘅 structured JSON。

---

## 🧠 設計理念（同你 FYP 架構一致）

- **Step-Back Prompting**：先從宏觀/同業/供應鏈角度諗「可能發生咩」再落 query。
- **HyDE**：用 hypothetical ideal article 去校準語意方向，提升 search precision。
- **Freshness reranking**：偏好近 7 日內容；超過 30 日標記 `[HISTORICAL]`。
- **Hallucination guard**：每個 factual claim 必須有 URL + date；single-source 要標記 `⚠️ UNCONFIRMED`。
- **Supervisor-ready output**：輸出固定 schema JSON，Supervisor/Synthesizer 可直接做 conflict detection。

---

## 🧩 目錄結構

agents/web_search/
├── README.md
├── agent.py # LangGraph node wrapper + agent runtime
├── tools.py # Poe API client + (optional) DuckDuckGo fallback
├── prompts.py # System prompt + templates (Step-back / HyDE)
└── tests/
└── test_agent.py # Unit tests (API mocked)

text

---

## ⚙️ 環境變數

加入到 `.env`：

```bash
POE_API_KEY=your_key_here
WEB_SEARCH_MODEL=deepseek-v3.2-exp
WEB_SEARCH_RECENCY_FILTER=week
你如果用 docker / airflow，記得將 .env 注入 scheduler / worker containers。

✅ Output Schema（Supervisor Consumption）
Agent 必須 return JSON：

json
{
  "agent": "web_search",
  "ticker": "AAPL",
  "query_date": "2026-02-24",
  "breaking_news": [
    {
      "title": "...",
      "url": "...",
      "published_date": "2026-02-24",
      "source_tier": 1,
      "relevance_score": 0.92,
      "verified": true
    }
  ],
  "sentiment_signal": "MIXED",
  "sentiment_rationale": "1 sentence with a cited URL",
  "unknown_risk_flags": [
    {"risk": "...", "source_url": "...", "severity": "HIGH"}
  ],
  "competitor_signals": [
    {"company": "...", "signal": "...", "source_url": "..."}
  ],
  "supervisor_escalation": {
    "action": "CONFLICT_SIGNAL",
    "rationale": "...",
    "conflict_with_agent": "consensus_strategy"
  },
  "fallback_triggered": false,
  "confidence": 0.75,
  "raw_citations": ["https://..."],
  "error": null
}
🧪 測試
bash
pip install -r requirements.txt
pytest agents/web_search/tests/ -v
🔌 Supervisor / LangGraph Integration
Supervisor graph 中加入：

python
from agents.web_search.agent import web_search_node
graph.add_node("web_search", web_search_node)

# outputs:
state["web_search_output"]["breaking_news"]
state["web_search_output"]["unknown_risk_flags"]
state["web_search_output"]["supervisor_escalation"]
最後更新：2026-02-24 | 作者：hck717


