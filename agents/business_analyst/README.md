# 📈 Business Analyst Agent

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。

---

## 🎯 Agent 職能

Business Analyst Agent 負責深入分析 **公司基本面、新聞情緒同市場定位**。呢個 agent 特別依賴 **RAG (Retrieval-Augmented Generation)** 資料库，從 Qdrant 入籏相關新聞將 LLM 左頁如文，到達 fact-grounded 的分析結果。

```
用戶問題
    │
    ▼
[Business Analyst Agent]
    │
    ├── RAG: Qdrant 向量搜尋相關新聞
    ├── Neo4j 查詢公司 knowledge graph
    ├── 分析新聞情緒趨勢
    ├── 評估競爭定位同行業背景
    └── 回傳 qualitative analysis 給 Supervisor
```

---

## 📦 負責範圍

### 適用場景

- 公司競爭優勢 (competitive moat) 分析
- 管理層質素評估（從公司描述同新聞）
- 新聞情緒分析（bearish / bullish %）
- 公司 ESG 、社會責任、風險因素
- 行業地位、市場份額、基本面摘要

### 不適用場景（由其他 agent 處理）

- 廾稿分析 / 先進指標 → Financial Modelling Agent
- 實時新聞 → Web Search Agent
- 宏觀環境 → Macro Metrics Agent

---

## 🛠️ 技術實現

### 主要 Data Sources

| 圖表 / 資料庫 | 內容 | 存僸位置 |
|---|---|---|
| `financial_documents` | 100 條最新新聞 embedding | Qdrant |
| `sentiment_trends` | bearish / bullish / neutral % | PostgreSQL |
| `:Company` nodes | 46 個 company properties | Neo4j |
| `company_profile.json` | EODHD 完整公司描述 | 本地檔案 |

### RAG 流程詳解

```
用戶問題
    │
    ▼
1. Embed query
   → Ollama nomic-embed-text 將問題轉化成 768-dim vector
    │
    ▼
2. Qdrant similarity search
   → cosine similarity 搜尋 top-k 相關新聞片段
   → 預設 k=8，分數 threshold > 0.6
    │
    ▼
3. Context assembly
   → 將搜尋結果 + Neo4j 公司資料 + sentiment 數據合併
    │
    ▼
4. LLM analysis
   → 基於 context 生成分析報告
   → 每個論點必須付上引用來源
```

### LLM 設定

```python
# Business analysis 需要平衡速度和質量
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.3,             # 稍高一點允許推理轉折
)

# Embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
```

### Agent 工作流（LangGraph）

```
START
  │
  ▼
fetch_company_profile       ←  Neo4j 查詢 Company node，獲得 46 個 properties
  │
  ▼
fetch_sentiment_data        ←  PostgreSQL 查詢 sentiment_trends
  │
  ▼
rag_news_search             ←  Qdrant cosine search top-8 相關新聞
  │
  ▼
assemble_context            ←  合併公司資料 + 情緒 + 新聞 context
  │
  ▼
analyse_competitive_moat    ←  LLM 分析公司競爭優勢
  │
  ▼
analyse_management_quality  ←  LLM 從公司描述和新聞評估管理層
  │
  ▼
assess_risks                ←  LLM 識別主要風險因素
  │
  ▼
format_output               ←  整理分析結果，每點附上引用
  │
 END → 回傳 Supervisor
```

### Output 格式

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "company_overview": {
    "name": "Apple Inc",
    "sector": "Technology",
    "market_cap": 3250000000000,
    "pe_ratio": 31.2,
    "profit_margin": 0.263
  },
  "sentiment": {
    "bullish_pct": 65,
    "bearish_pct": 20,
    "neutral_pct": 15,
    "trend": "improving"
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": ["ecosystem lock-in", "brand premium", "services revenue"],
    "sources": ["news_id_001", "news_id_042"]
  },
  "key_risks": [
    {"risk": "China market slowdown", "severity": "medium", "source": "news_id_018"}
  ],
  "qualitative_summary": "Apple 在高端用戶市場保持差異化居第一..."
}
```

---

## 📁 目錄結構

```
agents/business_analyst/
├── README.md          # 本文件
├── agent.py           # Agent 主體逻輯（LangGraph）
├── tools.py           # Qdrant RAG、Neo4j、PostgreSQL 封裝
├── rag.py             # RAG pipeline 詳細實現
├── prompts.py         # Qualitative analysis system prompts
└── tests/
    └── test_agent.py
```

---

## ⚙️ 環境變數

```bash
# .env 檔案加入
OLLAMA_BASE_URL=http://localhost:11434
BUSINESS_ANALYST_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=financial_documents
RAG_TOP_K=8                             # 搜尋返回最多 k 個結果
RAG_SCORE_THRESHOLD=0.6                 # 最低相似度門檻

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
```

---

## 🧪 快速測試

```bash
source .venv/bin/activate

# 分析指定公司
python agents/business_analyst/agent.py --ticker AAPL

# 測試 RAG 搜尋功能
python agents/business_analyst/rag.py --query "Apple iPhone sales" --ticker AAPL

# 執行測試
python -m pytest agents/business_analyst/tests/ -v
```

---

## 📝 設計容訊

- **Grounded analysis**：分析論點必須引用 Qdrant 搜尋返回嘅具體新聞，由 Critic Agent 核實
- **Sentiment weighting**：bulk sentiment 數據作為 LLM 分析嘅背景 context，不是結論
- **Fallback strategy**：Qdrant 無結果時回路 Neo4j company description 發動分析
- **包含數據**：記懟公司 46 個 properties 計入 prompt context，包含 PE、EBITDA、Market Cap

---

*最後更新：2026-02-24 | 作者：hck717*
