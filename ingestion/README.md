# 📥 FYP 數據 Ingestion 系統 — 完整說明文件

> **語言說明：** 本文件以廣東話書寫，所有技術術語保留英文原名。

---

## 📋 目錄

1. [系統架構概覽](#1-系統架構概覽)
2. [目錄結構](#2-目錄結構)
3. [數據流程圖](#3-數據流程圖)
4. [前置條件](#4-前置條件)
5. [第一次設置](#5-第一次設置)
6. [啟動所有服務](#6-啟動所有服務)
7. [Airflow DAG 操作](#7-airflow-dag-操作)
8. [各 Database 管理](#8-各-database-管理)
9. [Health Check](#9-health-check)
10. [日常操作指南](#10-日常操作指南)
11. [故障排除](#11-故障排除)

---

## 1. 系統架構概覽

呢個 `ingestion` 模組負責每日自動抓取金融數據，然後分別存入三個唔同嘅 database，供後續嘅 AI agent 使用。

```
[EODHD API]
     │
     ▼
[Airflow DAG]  ←  每日 01:00 HKT 自動執行
     │
     ├──► scrape  →  JSON / CSV 存到本地磁碟
     │
     ├──► load_postgres.py   →  [PostgreSQL]   時序數據、基本面數字
     ├──► load_neo4j.py      →  [Neo4j]        公司 knowledge graph
     └──► load_qdrant.py     →  [Qdrant]       新聞 embedding（RAG 用）
                                    ↑
                              [Ollama local]
                          nomic-embed-text model
```

### 三個 Database 嘅分工

| Database | 用途 | 數據類型 |
|---|---|---|
| **PostgreSQL** | 結構化數字數據 | 股價、財務指標、技術指標、宏觀數據 |
| **Neo4j** | 知識圖譜 | 公司資料、行業關係、財務摘要 |
| **Qdrant** | 向量搜尋（RAG） | 新聞文章 embedding，供 LLM 語義搜尋 |

---

## 2. 目錄結構

```
ingestion/
├── README.md                          # 本文件
├── inspect_data.py                    # 一鍵 health check 腳本
│
├── dags/
│   └── dag_eodhd_ingestion_unified.py # Airflow DAG 定義（主入口）
│
└── etl/
    ├── load_postgres.py               # PostgreSQL loader
    ├── load_neo4j.py                  # Neo4j loader
    ├── load_qdrant.py                 # Qdrant + Ollama embedding loader
    ├── README_etl_loaders.md          # ETL loaders 技術文件
    └── agent_data/                    # 本地 JSON/CSV 快取（由 DAG 生成）
        ├── business_analyst/
        │   └── {TICKER}/
        │       ├── metadata.json
        │       ├── financial_news.json / .csv
        │       ├── sentiment_trends.json / .csv
        │       └── company_profile.json / .csv
        ├── quantitative_fundamental/
        │   └── {TICKER}/              # 10 個 endpoints
        └── financial_modeling/
            └── {TICKER}/              # 10 個 endpoints
```

---

## 3. 數據流程圖

```
每日 Airflow 執行順序（每個 ticker 並行）：

eodhd_scrape_{agent}_{ticker}
        │
        ▼
 ┌──────┬──────┬──────┐
 │      │      │      │
 ▼      ▼      ▼      ▼
PG    Neo4j  Qdrant   (三個 loader 並行執行)
        │
        ▼
eodhd_generate_summary
```

### 三個 Agent 同佢哋嘅 endpoints

**`business_analyst`** (3 endpoints)
- `financial_news` → Qdrant（最新 100 條新聞，做 embedding）
- `sentiment_trends` → PostgreSQL（看好/看淡百分比）
- `company_profile` → Neo4j（公司 node，46 個 properties）

**`quantitative_fundamental`** (10 endpoints)
- 實時報價、EOD 歷史價格、intraday 1m/5m/15m/1h、基本面、期權、SMA、EMA → PostgreSQL

**`financial_modeling`** (10 endpoints)
- 週線/月線價格、股息、拆股、盈利、IPO、完整基本面、分析師預測、宏觀日曆、交易所資料、美股 bulk EOD → PostgreSQL

---

## 4. 前置條件

喺開始之前，確保以下軟件已安裝：

```bash
# 檢查版本
docker --version          # 需要 Docker Desktop >= 4.x
docker compose version    # 需要 >= 2.x
python3 --version         # 需要 >= 3.11
ollama --version          # 需要 >= 0.5.x
```

### 環境變數設置

喺 repo 根目錄建立 `.env` 檔案（如果未有）：

```bash
# .env
EODHD_API_KEY=your_eodhd_api_key_here
TRACKED_TICKERS=AAPL,GOOGL,MSFT,NVDA,TSLA

# PostgreSQL
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Neo4j
NEO4J_AUTH=neo4j/changeme_neo4j_password

# Qdrant
QDRANT_COLLECTION_NAME=financial_documents

# Ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
EMBEDDING_MODEL=nomic-embed-text
```

---

## 5. 第一次設置

### Step 1 — Clone repo 同安裝 Python dependencies

```bash
git clone https://github.com/hck717/FYP.git
cd FYP

# 建立 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### Step 2 — 安裝同拉取 Ollama model

```bash
# 安裝 Ollama（如果未有）
brew install ollama        # macOS

# 啟動 Ollama server（背景執行）
ollama serve &

# 拉取 embedding model（只需做一次，約 274 MB）
ollama pull nomic-embed-text

# 確認 model 已載入
curl http://localhost:11434/api/tags
# 應該見到 nomic-embed-text:latest
```

### Step 3 — 建立 Docker image

```bash
cd FYP

# 第一次建立（需要幾分鐘）
docker compose build --no-cache

# 確認 images 建立成功
docker images | grep fyp
```

---

## 6. 啟動所有服務

### 正常啟動

```bash
# 啟動全部 containers（背景執行）
docker compose up -d

# 等待約 30 秒讓所有服務初始化
sleep 30

# 確認全部 containers 正常運行
docker ps
```

預期見到以下 containers 全部 `Up` 或 `healthy`：

| Container | 服務 | Port |
|---|---|---|
| `fyp-airflow-webserver` | Airflow UI | `localhost:8080` |
| `fyp-airflow-scheduler` | DAG 執行引擎 | — |
| `fyp-postgres` | PostgreSQL | `localhost:5432` |
| `fyp-neo4j` | Neo4j | `localhost:7474` (HTTP) / `7687` (Bolt) |
| `fyp-qdrant` | Qdrant | `localhost:6333` |

### 停止服務

```bash
# 停止但保留數據
docker compose down

# 停止並刪除所有數據（完全重置）
docker compose down -v
```

### 重建（代碼有更新後）

```bash
git pull
docker compose build --no-cache
docker compose up -d
```

---

## 7. Airflow DAG 操作

### 訪問 Airflow UI

```
URL:      http://localhost:8080
Username: airflow
Password: airflow
```

DAG 名稱：**`eodhd_complete_ingestion`**

### 手動觸發 DAG（命令行）

```bash
# 觸發完整 ingestion（所有 agents、所有 tickers）
docker exec fyp-airflow-scheduler airflow dags trigger eodhd_complete_ingestion

# 查看 DAG run 狀態
docker exec fyp-airflow-scheduler airflow dags list-runs -d eodhd_complete_ingestion
```

### 手動測試單個 task（Debug 用）

```bash
# 測試單個 ticker 嘅 Qdrant load task
docker exec fyp-airflow-scheduler airflow tasks test \
  eodhd_complete_ingestion \
  eodhd_load_qdrant_business_analyst_AAPL \
  2026-02-24

# 測試 PostgreSQL load
docker exec fyp-airflow-scheduler airflow tasks test \
  eodhd_complete_ingestion \
  eodhd_load_postgres_business_analyst_AAPL \
  2026-02-24
```

> ⚠️ `tasks test` 命令會出現 `ValueError: DAG run not found` 嘅錯誤訊息，呢個係 Airflow 嘅已知問題（test mode 唔寫 XCom），**實際 task 已經成功執行**，可以忽略。

### 重新 embed 所有 tickers（Qdrant 需要重建時）

```bash
for ticker in AAPL GOOGL MSFT NVDA TSLA; do
  docker exec fyp-airflow-scheduler airflow tasks test \
    eodhd_complete_ingestion \
    eodhd_load_qdrant_business_analyst_${ticker} \
    2026-02-24
done
```

### DAG Schedule

| 設定 | 值 |
|---|---|
| **執行時間** | 每日 01:00 UTC（即 HKT 09:00）|
| **Retry 次數** | 預設 2 次，Qdrant tasks 3 次 |
| **Retry 間隔** | 5 分鐘（Qdrant tasks 2 分鐘）|
| **Catchup** | 關閉（唔會補跑舊日期）|

---

## 8. 各 Database 管理

### 8.1 PostgreSQL

```bash
# 連接到 PostgreSQL
docker exec -it fyp-postgres psql -U airflow -d airflow

# 常用查詢
-- 查看所有 tables
\dt

-- 查看各 ticker 嘅數據量
SELECT ticker, COUNT(*) as rows,
       MIN(date) as from_date, MAX(date) as to_date
FROM raw_timeseries
GROUP BY ticker ORDER BY ticker;

-- 查看最新入庫時間
SELECT MAX(ingested_at) FROM raw_timeseries;
SELECT MAX(ingested_at) FROM raw_fundamentals;

-- 清空所有數據（完全重置）
TRUNCATE raw_timeseries, raw_fundamentals,
         market_eod_us, global_economic_calendar,
         global_ipo_calendar;

-- 退出
\q
```

**主要 Tables：**

| Table | 內容 |
|---|---|
| `raw_timeseries` | 所有時序數據（價格、intraday、技術指標）|
| `raw_fundamentals` | 基本面數據（PE、EPS、市值等）|
| `market_eod_us` | 美股 bulk EOD（全市場）|
| `global_economic_calendar` | 宏觀經濟事件日曆 |
| `global_ipo_calendar` | IPO 日曆 |

### 8.2 Neo4j

```bash
# 方法一：Browser UI
# URL: http://localhost:7474
# Username: neo4j
# Password: changeme_neo4j_password

# 方法二：命令行 cypher-shell
docker exec -it fyp-neo4j cypher-shell \
  -u neo4j -p changeme_neo4j_password
```

**常用 Cypher 查詢：**

```cypher
-- 查看所有 Company nodes
MATCH (c:Company) RETURN c.ticker, c.Name, c.Sector LIMIT 10;

-- 查看特定公司嘅全部 properties
MATCH (c:Company {ticker: 'AAPL'}) RETURN c;

-- 查看公司市值
MATCH (c:Company)
RETURN c.ticker, c.Name,
       c.Highlights_MarketCapitalization,
       c.Valuation_TrailingPE
ORDER BY c.Highlights_MarketCapitalization DESC;

-- 清空所有數據（重置）
MATCH (n) DETACH DELETE n;
```

**已儲存嘅 Node properties（46 個）：**
基本資料 (Name, Sector, Industry, Exchange)、財務指標 (Highlights_MarketCapitalization, Highlights_PERatio, Highlights_ProfitMargin)、估值指標 (Valuation_TrailingPE, Valuation_ForwardPE, Valuation_EnterpriseValue) 等。

### 8.3 Qdrant

```bash
# 查看所有 collections
curl -s http://localhost:6333/collections | python3 -m json.tool

# 查看 financial_documents collection 狀態
curl -s http://localhost:6333/collections/financial_documents | python3 -c "
import sys, json
d = json.load(sys.stdin)['result']
print(f'Points          : {d[\"points_count\"]}')
print(f'Indexed vectors : {d[\"indexed_vectors_count\"]}')
print(f'Status          : {d[\"status\"]}')
"

# 刪除 collection（需要重建時）
curl -X DELETE http://localhost:6333/collections/financial_documents

# 測試語義搜尋（Python）
python3 - <<'EOF'
import requests

# 先 embed query
resp = requests.post(
    'http://localhost:11434/api/embeddings',
    json={'model': 'nomic-embed-text', 'prompt': 'Apple iPhone sales revenue 2025'}
)
query_vector = resp.json()['embedding']

# 搜尋最相似嘅 5 條新聞
result = requests.post(
    'http://localhost:6333/collections/financial_documents/points/search',
    json={
        'vector': query_vector,
        'limit': 5,
        'with_payload': True
    }
).json()

for hit in result['result']:
    print(f"Score: {hit['score']:.4f} | Ticker: {hit['payload'].get('ticker_symbol')} | {hit['payload'].get('title', '')[:80]}")
EOF
```

> 📌 **關於 `indexed_vectors_count: 0` 嘅說明：**
> 呢個係正常現象。Qdrant 嘅預設 `indexing_threshold` 係 20,000 KB。當前 ~655 個 vectors × 768 維 ≈ 2 MB，低於門檻，所以 Qdrant 用 **brute-force full-scan** 搜尋，準確度 100%，而且對小型 collection 更快。HNSW index 會喺 collection 增大後自動建立。

---

## 9. Health Check

```bash
# 啟動 virtual environment
source .venv/bin/activate

# 執行完整 health check（檢查 File System + PostgreSQL + Neo4j + Qdrant）
python ingestion/inspect_data.py
```

**正常狀態下應該見到：**

```
✅ Connected to localhost:5432/airflow
✅ raw_timeseries           516,197 rows
✅ Connected to bolt://localhost:7687
✅ :Company                 5 nodes
✅ Status: green  |  Points: 655
```

**需要關注嘅警告：**

| 訊息 | 原因 | 需要處理? |
|---|---|---|
| `⚠️ :Fact 0 nodes` | FMP API 未接入，只有 EODHD 數據 | 否 |
| `⚠️ No relationships found` | 同上 | 否 |
| `Vectors: 0` | Collection 細過 indexing threshold | 否（正常）|
| `Empty file for sentiment_trends` | 舊 DAG run 嘅遺留，新版已修復 | 否 |

---

## 10. 日常操作指南

### 每日正常流程（全自動）

```
Ollama 保持運行中
    ↓
Airflow scheduler 每日 01:00 UTC 自動觸發 DAG
    ↓
自動 scrape → load PostgreSQL + Neo4j + Qdrant
    ↓
如果 Ollama 未回應，Qdrant task 會自動 retry 3 次（間隔 2 分鐘）
```

### 手動觸發完整 ingestion

```bash
docker exec fyp-airflow-scheduler airflow dags trigger eodhd_complete_ingestion
```

### 增加新 ticker

喺 `.env` 檔案加入新 ticker：

```bash
TRACKED_TICKERS=AAPL,GOOGL,MSFT,NVDA,TSLA,AMZN  # 加入 AMZN
```

然後重新啟動 scheduler：

```bash
docker compose restart fyp-airflow-scheduler
```

### 查看 Airflow logs

```bash
# 查看 scheduler logs（實時）
docker logs -f fyp-airflow-scheduler

# 查看特定 task 嘅 log
docker exec fyp-airflow-scheduler airflow tasks logs \
  eodhd_complete_ingestion \
  eodhd_load_qdrant_business_analyst_AAPL \
  2026-02-24
```

---

## 11. 故障排除

### Ollama 連接失敗（Qdrant embedding 出錯）

```bash
# 確認 Ollama 正在運行
curl http://localhost:11434/api/tags

# 如果唔通，重新啟動 Ollama
ollama serve

# 確認 nomic-embed-text 已載入
ollama list
# 如果未有，重新拉取
ollama pull nomic-embed-text
```

### Docker containers 未能啟動

```bash
# 查看啟動錯誤
docker compose logs --tail=50

# 強制重建
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### PostgreSQL 連接失敗

```bash
# 確認 container 正在運行
docker ps | grep postgres

# 直接測試連接
docker exec fyp-postgres pg_isready -U airflow
```

### Neo4j 連接失敗

```bash
# 確認 container 正在運行
docker ps | grep neo4j

# 查看 Neo4j logs
docker logs fyp-neo4j --tail=30

# 測試連接
docker exec fyp-neo4j cypher-shell -u neo4j -p changeme_neo4j_password \
  "RETURN 'connected' AS status;"
```

### Qdrant collection 唔見咗

```bash
# 重建 collection（執行所有 5 個 tickers 嘅 embed loop）
for ticker in AAPL GOOGL MSFT NVDA TSLA; do
  docker exec fyp-airflow-scheduler airflow tasks test \
    eodhd_complete_ingestion \
    eodhd_load_qdrant_business_analyst_${ticker} \
    2026-02-24
done
```

### DAG 唔出現喺 Airflow UI

```bash
# 確認 DAG 無語法錯誤
docker exec fyp-airflow-scheduler python \
  /opt/airflow/dags/dag_eodhd_ingestion_unified.py

# 強制 reload DAGs
docker exec fyp-airflow-scheduler airflow dags reserialize
```

---

## 附錄：ETL Loaders 技術細節

詳細嘅 ETL loader 技術文件請參閱 [`etl/README_etl_loaders.md`](etl/README_etl_loaders.md)。

| Loader | 檔案 | 主要功能 |
|---|---|---|
| `load_postgres.py` | PostgreSQL loader | 自動辨識數據類型，upsert 到對應 table |
| `load_neo4j.py` | Neo4j loader | 建立/更新 Company node，flatten 巢狀 JSON |
| `load_qdrant.py` | Qdrant + Ollama | 清洗文字、embed、upsert vectors，Ollama warmup probe |

---

*最後更新：2026-02-24 | 作者：hck717*
