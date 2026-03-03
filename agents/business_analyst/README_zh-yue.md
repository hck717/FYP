# 商業分析師代理人 (Business Analyst Agent)

> **狀態：** 完成（已用全部五隻股票進行真實測試）
> **基於：** [`skills/business_analyst_crag`](https://github.com/hck717/Agent-skills-POC/tree/main/skills/business_analyst_crag)（POC v4.2）

---

## 職責

商業分析師代理人係多代理人股票研究系統入面嘅**定性智能層**。佢負責回答有關公司結構性問題——競爭護城河、商業模式、風險因素同策略定位——透過喺本地知識庫撈取有核實來源嘅事實。

佢**唔會**作出買入或賣出嘅判斷。佢負責將有根據嘅事實同資料空白呈現俾監督代理人（Supervisor）同整合代理人（Synthesizer）去詮釋。

**呢個代理人負責處理：**
- 競爭護城河同策略定位分析
- 商業模式同收入構成分析
- 風險因素綜合分析（來自申報文件同新聞）
- 歷史情緒趨勢（看漲／看跌／中性百分比）
- 有引用來源嘅定性敘述綜合

**由其他代理人負責（唔好重疊）：**
- 即時新聞同突發事件 → 網絡搜尋代理人
- 財務比率、DCF、估值 → 財務建模代理人
- 宏觀環境 → 宏觀指標代理人

---

## 架構：CRAG（糾正式 RAG）

呢個代理人實現咗**圖增強糾正式 RAG**——喺生成答案之前評估檢索信心，然後按情況調整策略。

```
查詢 + 股票代號
    │
    ▼
fetch_sentiment_data      ←  PostgreSQL：看漲／看跌／中性百分比
    │
    ▼
hybrid_retrieval          ←  Qdrant 向量搜尋（768維，nomic-embed-text）【主要】
                          ←  Neo4j 向量索引（備用——目前返回 0 個結果）
                          ←  BM25 稀疏關鍵詞評分
    │
    ▼
hybrid_rerank             ←  30% BM25 + 70% Cross-Encoder（ms-marco-MiniLM-L-6-v2）
                          ←  當最高 CE 分數 < 0.4 時：改用 70% 密集向量 + 30% BM25 混合
    │
    ▼
crag_evaluate             ←  CORRECT（>0.55）/ AMBIGUOUS（0.35–0.55）/ INCORRECT（<0.35）
    │
    ├─ CORRECT    → generate_analysis（LLM 根據已檢索上下文生成）
    ├─ AMBIGUOUS  → rewrite_query → 重試 hybrid_retrieval（最多重寫 1 次）
    └─ INCORRECT  → web_search_fallback（調用網絡搜尋代理人存根）
    │
    ▼
format_json_output        ←  結構化 JSON，交俾監督代理人
    │
   END → 返回監督代理人
```

---

## 基礎設施狀態

| 服務 | 容器 | 狀態 | 備註 |
|---|---|---|---|
| PostgreSQL | `fyp-postgres` | 正常運行 | `sentiment_trends` 表：5 行（AAPL/TSLA/MSFT/NVDA/GOOGL） |
| Neo4j | `fyp-neo4j` | 正常運行 | 5 個 `Company` 節點，每個節點有 85+ 個真實財務屬性（市值、市盈率、利潤率、行業等）。無 `Chunk` 節點，無關係邊。向量搜尋返回 0 個結果；代理人自動退回 Qdrant。`company_overview` 欄位使用 Company 節點屬性。 |
| Qdrant | `fyp-qdrant` | 正常運行 | 約 2,390 個向量（AAPL：407，TSLA：419，NVDA：423，MSFT：422，GOOGL：329），768維，`financial_documents` 集合。 |
| Ollama | 本地 | 運行中 | 模型：`nomic-embed-text:latest`、`deepseek-r1:8b`、`qwen2.5:7b`、`llama3.2:latest`。版本 0.14.2。 |

---

## 數據來源

| 來源 | 內容 | 儲存方式 |
|---|---|---|
| `financial_documents` | 新聞文章、業績摘要、分析師報告 | Qdrant（約 2,390 個向量，768維） |
| `sentiment_trends` | 每隻股票嘅看漲／看跌／中性百分比及趨勢方向 | PostgreSQL |
| `:Company` 節點 | 每隻股票有 85+ 個財務屬性嘅公司節點 | Neo4j（屬性用於 `company_overview`；無文件數據） |

**情緒數據（實際數值）：**

| 股票代號 | 看漲 % | 看跌 % | 中性 % | 趨勢 |
|---|---|---|---|---|
| AAPL | 72.4 | 14.2 | 13.4 | 改善中 |
| MSFT | 68.9 | 17.3 | 13.8 | 改善中 |
| NVDA | 79.3 | 11.5 | 9.2 | 改善中 |
| GOOGL | 61.8 | 22.4 | 15.8 | 穩定 |
| TSLA | 45.1 | 38.7 | 16.2 | 惡化中 |

**關於 Neo4j 嘅說明：** 圖數據庫目前冇 `Chunk` 節點或知識圖譜關係（`FACES_RISK`、`HAS_STRATEGY`、`COMPETES_WITH`、`HAS_FACT`）。所有文件檢索均透過 Qdrant 進行。Neo4j 有關缺失屬性嘅警告係預期內，唔影響功能。

---

## LLM 及模型

```python
# 主要 LLM——推理級模型，用於定性分析
llm = "deepseek-r1:8b"           # 透過 Ollama，地址：localhost:11434
temperature = 0.2                 # 低溫：偏向事實依據
num_predict = 8192                # 最大生成 token 數（為詳細輸出而增加）
request_timeout = None            # 無超時——deepseek-r1 有時需要較長時間

# 嵌入模型——尺寸必須符合 Qdrant 向量索引
embedder = "nomic-embed-text"     # 透過 Ollama /api/embed，768維

# 重排序器——Cross-Encoder，用於 CRAG 評估同最終重排序
reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # sentence-transformers，本地 CPU 運行

# 檢索
top_k = 15                        # 從 Qdrant 檢索的塊數（由 8 增加）
chunks_fed_to_llm = 10            # 傳入上下文的最高 N 個塊（由 6 增加）
chars_per_chunk = 800             # 每個塊傳給 LLM 的字符數（由 400 增加）
```

**deepseek-r1:8b 行為說明：**
- Ollama API 請求中設置咗 `"think": False`，喺 API 層面抑制 `<think>...</think>` 推理塊（需要 Ollama ≥ 0.14.2）。`llm.py` 中嘅 `_strip_think_tags()` 亦會作為防禦性備用方案。
- 有時會用 ` ```json ``` ` Markdown 圍欄包裹輸出——JSON 解析前由 `_strip_markdown_fences()` 去除。

---

## CRAG 評估邏輯

| 分數 | 狀態 | 行動 |
|---|---|---|
| > 0.55 | `CORRECT` | 直接用已檢索上下文生成 |
| 0.35 – 0.55 | `AMBIGUOUS` | LLM 重寫查詢 → 重試檢索一次（最多 1 次循環） |
| < 0.35 | `INCORRECT` | 觸發網絡搜尋代理人備用方案 |

當所有候選文件嘅 Cross-Encoder 最高分數 < 0.4 時，重排序器退回使用 `0.7×密集向量 + 0.3×BM25` 混合方式，而唔係 CE 加權分數。

---

## 輸出格式（JSON）

代理人**只返回結構化 JSON**——無自由格式 Markdown 文字。監督代理人直接讀取呢個輸出。

每個事實聲明都會引用一個來自 Qdrant 的 `chunk_id`（格式：`qdrant::{ticker}::{title_slug}`）。

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "query_date": "2026-02-26",
  "company_overview": {
    "name": "Apple Inc",
    "sector": "Technology",
    "market_cap": 3200000000000,
    "pe_ratio": 28.5,
    "profit_margin": 0.253
  },
  "sentiment": {
    "bullish_pct": 72.4,
    "bearish_pct": 14.2,
    "neutral_pct": 13.4,
    "trend": "improving",
    "source": "postgresql:sentiment_trends",
    "sentiment_interpretation": "說明情緒數據如何印證或反駁文件發現嘅敘述 [chunk_id: ...]"
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": [
      "生態系統鎖定效應 [chunk_id: qdrant::AAPL::...]"
    ],
    "vulnerabilities": [
      "中國市場依賴風險 [chunk_id: qdrant::AAPL::...]"
    ],
    "sources": ["qdrant::AAPL::..."]
  },
  "qualitative_analysis": {
    "narrative": "至少 3 句直接回答分析師問題嘅敘述，附有 [chunk_id] 引用",
    "sentiment_signal": "情緒數據如何印證或反駁文件發現 [chunk_id: ...]",
    "strategic_implication": "最重要嘅 2-3 年商業模式影響 [chunk_id: ...]",
    "data_quality_note": "對已檢索上下文質量及空白嘅誠實評估"
  },
  "key_risks": [
    {
      "risk": "帶有 [chunk_id] 引用嘅風險描述",
      "severity": "HIGH",
      "mitigation_observed": "觀察到嘅緩解措施或 null",
      "source": "chunk_id 字符串"
    }
  ],
  "missing_context": [
    {
      "gap": "缺失內容描述",
      "severity": "HIGH"
    }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.85,
  "fallback_triggered": false,
  "qualitative_summary": "1-2 句帶有 chunk_id 引用嘅事實摘要——無情緒判斷"
}
```

**Qdrant 來源嘅 chunk_id 格式：** `qdrant::{TICKER}::{title_slug}`（由 Qdrant payload 中嘅 `ticker_symbol` 同 `title` 欄位生成）。Slug 截斷至約 50 個字符，保留文章標題中嘅 Unicode 字符（例如彎曲引號 U+2019）。

---

## 公開 API

呢個包從 `agents.business_analyst` 導出兩個函數：

### `run(task, ticker, config)` — 針對性查詢

```python
from agents.business_analyst import run

result = run(task="What is Apple's competitive moat?", ticker="AAPL")
```

回答**單一分析師問題**。輸出範圍受限於 `task` 字串。當監督代理人或整合代理人有具體後續問題時使用。

### `run_full_analysis(ticker, config)` — 全面公司檔案

```python
from agents.business_analyst import run_full_analysis

dossier = run_full_analysis(ticker="AAPL")
# dossier["competitive_moat"]["rating"]  → "wide"
# dossier["key_risks"]                   → [{risk, severity, source}, ...]
# dossier["qualitative_summary"]          → "1-2 句執行摘要"
```

以**單次流水線運行**發出涵蓋五大支柱嘅全面任務：
1. 競爭護城河——評級（wide/narrow/none）、優勢、弱點
2. 商業模式及主要收入來源
3. 策略定位同最重要嘅 2-3 年業務影響
4. 關鍵風險因素，附嚴重程度（HIGH/MEDIUM/LOW）及觀察到嘅緩解措施
5. 當前情緒如何印證或反駁文件證據

**呢個係整合代理人（Synthesizer）嘅預期入口點。** 佢返回同 `run()` 相同嘅 JSON 結構，所有分析部分均已填充，令整合代理人無需多次調用代理人即可獲得完整嘅定性智能包。

---

## 文件結構

```
agents/business_analyst/
├── README.md              # 英文版說明文件
├── README_zh-yue.md       # 廣東話版說明文件（即此文件）
├── __init__.py            # 包初始化——導出 run() 同 run_full_analysis()
├── agent.py               # LangGraph CRAG 流水線（8 個節點）+ run_full_analysis()
├── config.py              # 集中式環境變數配置
├── health.py              # 服務健康檢查腳本
├── llm.py                 # Ollama LLM 客戶端（生成、查詢重寫、JSON 提取）
├── prompts.py             # 系統提示 + JSON 結構提示 + 查詢重寫提示
├── schema.py              # 數據類：Chunk、RetrievalResult、SentimentSnapshot、CRAGStatus
├── tools.py               # Neo4j、Qdrant、PostgreSQL 連接器 + CRAG 評估器 + 重排序器
├── web_search_interface.py # 網絡搜尋代理人備用存根
└── tests/
    └── test_agent.py      # 39 個單元及整合測試（全部模擬，全部通過）
```

---

## 環境變數

```bash
# LLM
OLLAMA_BASE_URL=http://localhost:11434
BUSINESS_ANALYST_MODEL=deepseek-r1:8b

# 嵌入模型（透過 Ollama——必須符合 Qdrant 向量維度）
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# 重排序器（透過 sentence-transformers 本地加載）
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=financial_documents
RAG_TOP_K=8
RAG_SCORE_THRESHOLD=0.6

# Neo4j（Company 節點屬性用於 company_overview；向量搜尋返回 0——自動退回 Qdrant）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# PostgreSQL（情緒數據）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
```

---

## 運行命令

**基本命令（按需替換 `--ticker` 同 `--task`）：**

```bash
cd /Users/brianho/FYP && \
BUSINESS_ANALYST_MODEL=deepseek-r1:8b \
EMBEDDING_MODEL=nomic-embed-text \
EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker AAPL \
  --task "What is Apple's main business model and revenue sources?" \
  --log-level WARNING
```

**5 個建議測試命令：**

```bash
# 1. AAPL——競爭護城河（預設任務，Qdrant 中有良好根基）
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=nomic-embed-text EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker AAPL --log-level WARNING

# 2. TSLA——風險因素（測試 key_risks 輸出塊）
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=nomic-embed-text EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker TSLA \
  --task "What are Tesla's key business risks and competitive vulnerabilities?" \
  --log-level WARNING

# 3. NVDA——策略定位（測試 competitive_moat 塊）
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=nomic-embed-text EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker NVDA \
  --task "How defensible is NVIDIA's AI chip moat against in-house alternatives?" \
  --log-level WARNING

# 4. MSFT——服務及雲端策略（測試 qualitative_analysis 敘述）
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=nomic-embed-text EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker MSFT \
  --task "Assess Microsoft's cloud and AI services strategy and revenue mix." \
  --log-level WARNING

# 5. GOOGL——引用根基檢查（INFO 日誌級別可見詳情）
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=nomic-embed-text EMBEDDING_DIMENSION=768 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker GOOGL \
  --task "What is Alphabet's advertising dependency risk and diversification strategy?" \
  --log-level INFO
```

---

## 已驗證嘅真實測試結果

全部 5 隻支持嘅股票已用真實嘅 Qdrant、Neo4j 同 PostgreSQL 數據進行端對端驗證。

| 股票代號 | CRAG 狀態 | 信心分數 | 輸出中嘅引用數量 | 引用警告 | 備用方案觸發 |
|---|---|---|---|---|---|
| AAPL | CORRECT | 0.72 | 9 | 0 | false |
| TSLA | CORRECT | — | 9 | 0 | false |
| MSFT | CORRECT | — | 5 | 1（LLM 錯字，已由後處理器清除） | false |
| NVDA | CORRECT | — | 9 | 0 | false |
| GOOGL | CORRECT | — | 10 | 0 | false |

**備註：**
- MSFT：LLM 喺一個塊 slug 中生成咗 `Quiety` 而非 `Quietly`。`_strip_ungrounded_inline_citations()` 偵測並移除咗錯誤拼寫嘅引用；正確拼寫嘅引用已保留。
- GOOGL：塊 ID 包含 Unicode 彎曲引號（U+2019）。`json.dumps(ensure_ascii=False)` 加上 Unicode NFKD 標準化確保呢啲字符能正確匹配，唔會出現誤報嘅引用警告。
- Neo4j 對所有股票返回 0 個塊（無 `Chunk` 節點）。Qdrant 備用方案自動激活。所有引用均以 `qdrant::` 為前綴。

---

## 設計決策

- **CRAG 優於基本 RAG：** 喺生成之前評估檢索信心——唔會靜默地接受低質量答案。
- **網絡搜尋備用係設計意圖：** 當上下文分數 < 0.35 時，代理人調用網絡搜尋代理人——呢係代理人之間嘅協作，唔係失敗。
- **只輸出 JSON：** 所有輸出係結構化 JSON，由監督代理人直接讀取——無自由格式 Markdown 文字。
- **情緒作為上下文，而非結論：** PostgreSQL 嘅情緒百分比作為背景上下文注入；LLM 將其對照文件證據去詮釋，而唔係直接繼承標籤。
- **引用強制執行（規則 9）：** `key_risks`、`competitive_moat` 同 `qualitative_analysis` 中每個聲明都必須引用真實嘅 `chunk_id`。系統提示明確禁止生成虛構 ID。
- **行內引用後處理器：** `_strip_ungrounded_inline_citations()` 喺 LLM 生成後遞歸遍歷輸出字典，將任何唔符合已檢索塊 ID 嘅 `[qdrant::TICKER::slug]` 標記替換為 `[source unavailable]`。呢係針對 LLM 幻覺式捏造塊 ID 嘅安全網。
- **Unicode 標準化處理塊 ID：** Qdrant 塊 ID 可能包含來自文章標題嘅 Unicode 字符（例如 U+2019 彎曲引號）。引用 ID 同實際 ID 在比較前均進行 NFKD 標準化至 ASCII，以防止誤報嘅根基失敗。
- **`json.dumps(ensure_ascii=False)`：** 包含 Unicode 字符嘅 LLM 輸出序列化時不進行 ASCII 轉義，以保留原始字符供正則表達式引用匹配使用。
- **無超時：** `request_timeout = None`——deepseek-r1:8b 處理複雜提示時可能需要幾分鐘；硬超時會導致錯誤嘅 `GENERATION_ERROR` 失敗。
- **`"think": False` API 參數：** 喺 Ollama 請求 payload 中傳遞，喺 API 層面抑制 deepseek-r1 嘅 `<think>` 塊（需要 Ollama ≥ 0.14.2）。比 `/no_think` 指令更可靠。防禦性 `_strip_think_tags()` 亦作為備用方案運行。
- **`_strip_markdown_fences`：** deepseek-r1 有時會用 ` ```json ``` ` 圍欄包裹輸出；呢啲圍欄喺 `llm.py` 中 JSON 解析之前被移除。
- **Qdrant 係主要檢索來源：** Neo4j 向量索引已接入，但返回 0 個結果（無已攝取嘅 `Chunk` 節點）。代理人自動退回 Qdrant，所有生產引用均來自 Qdrant。

---

## 測試

```bash
# 運行全部 39 個測試
cd /Users/brianho/FYP && .venv/bin/python -m pytest agents/business_analyst/tests/ -q

# 預期結果：39 passed
```

所有測試均為單元／整合測試，外部服務（Qdrant、Neo4j、PostgreSQL、Ollama）均已模擬。運行測試套件唔需要真實基礎設施。

---

*最後更新：2026-02-26*
