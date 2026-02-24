# 🤖 FYP Agent 層 — 概覽

> **語言說明：** 本目錄以廣東話書寫，所有技術術語保留英文原名。

---

## 系統架構概覽

呢個 `agents` 模組包含四個專門化嘅 AI agent，每個負責唔同類型嘅金融分析工作。佔們由一個 **Supervisor** 統一調度，形成一個 multi-agent 系統。

```
用戶輸入 (query)
       │
       ▼
  [Supervisor Agent]  ←  路由、分配、合併結果
       │
  ┌───┬───┬───┬───┐
  ▼    ▼    ▼    ▼
[WS] [MM] [BA] [FM]   ←  四個專門 agent 並行執行
  │    │    │    │
  ▼    ▼    ▼    ▼
各自呈回結果同來源引用
       │
       ▼
  [Critic Agent]  ←  核實、去重、測試一致性
       │
       ▼
  [Summarizer]    ←  生成最終 Markdown 報告
```

---

## 四個 Agent 簡介

| Agent | 目錄 | 主要負責 | 主要工具 |
|---|---|---|---|
| 🔍 **Web Search Agent** | `web_search/` | 實時網絡搜尋、最新新聞 | Tavily / DuckDuckGo Search |
| 🌏 **Macro Metrics Agent** | `macro_metrics/` | 宏觀經濟指標分析 | EODHD economic-events API, PostgreSQL |
| 📈 **Business Analyst Agent** | `business_analyst/` | 公司基本面、新聞情緒分析 | Qdrant RAG, Neo4j, Ollama |
| 🧩 **Financial Modelling Agent** | `financial_modelling/` | 定量模型、估值、技術分析 | PostgreSQL, DCF, 技術指標 |

---

## 共用技術堆疊

| 工具 | 用途 |
|---|---|
| **LangGraph** | Agent 狀態機 / 工作流管理 |
| **LangChain** | LLM 呼叫、tool calling、prompt management |
| **Ollama** | 本地 LLM 推理（qwen2.5:7b、deepseek-r1:8b、llama3.2）|
| **Qdrant** | 新聞 embedding 語義搜尋（RAG）|
| **Neo4j** | 公司知識圖譜查詢 |
| **PostgreSQL** | 历史時序數據、基本面指標 |

---

## 專項 README

- 🔍 [Web Search Agent](web_search/README.md)
- 🌏 [Macro Metrics Agent](macro_metrics/README.md)
- 📈 [Business Analyst Agent](business_analyst/README.md)
- 🧩 [Financial Modelling Agent](financial_modelling/README.md)

---

*最後更新：2026-02-24 | 作者：hck717*
