# 🏗️ Project Setup & Infrastructure Guide

This document details the repository structure and infrastructure configuration for **The Agentic Investment Analyst**.

## 1. Repository Structure

The project follows a modular "Plan-Execute-Synthesize" architecture. Create the following directory structure in your root folder:

```text
/FYP
├── /agents                 # Core Agent Logic (The "Brains")
│   ├── /supervisor         # GPT-4o Supervisor (Planner)
│   ├── /critic             # Critic Agent (Verification)
│   └── /summarizer         # Synthesis Agent (Reporting)
│
├── /ingestion              # Data Pipeline (The "Worker")
│   ├── /dags               # Airflow DAGs (Scheduled Tasks)
│   ├── /etl                # Extraction & Transformation scripts
│   └── /schema             # Database schemas (SQL & Qdrant)
│
├── /skills                 # Deterministic Capabilities (The "Hands")
│   ├── /fundamental        # Skill 1: Fundamental Health
│   ├── /technical          # Skill 2: Price & Momentum
│   ├── /valuation          # Skill 3: Valuation Reality
│   ├── /sentiment          # Skill 4: Sentiment Analysis
│   └── /risk               # Skill 5: Risk Assessment
│
├── /rag                    # Retrieval System
│   ├── /embeddings         # Embedding generation logic
│   └── /retrieval          # Hybrid search implementation
│
├── /ui                     # User Interface
│   ├── /components         # Streamlit widgets (Charts, Chat)
│   └── app.py              # Main Streamlit entry point
│
├── /data                   # Local Data Storage (Git-ignored)
│   ├── /raw                # Unprocessed downloads
│   └── /logs               # System logs
│
├── /docker                 # Infrastructure Configuration
│   ├── airflow.Dockerfile  # Custom Airflow image
│   └── requirements.txt    # Python dependencies
│
├── .env                    # Environment secrets (Git-ignored)
└── docker-compose.yml      # Service orchestration
```

### Module Descriptions

- **`/agents`**: Contains the reasoning engines. The `supervisor` routes tasks, `critic` validates claims, and `summarizer` compiles the final markdown report.
- **`/ingestion`**: Manages the "LLM-ETL" pipeline. Airflow DAGs here trigger Llama 3.2 for extraction and load data into Postgres/Qdrant.
- **`/skills`**: Each subdirectory (e.g., `/valuation`) should contain a `SKILL.md` manifest and a Python entry point (e.g., `tool.py`) that the Supervisor can call.
- **`/rag`**: Encapsulates the self-improving retrieval logic, including relevance feedback loops and dynamic chunking strategies.

---

## 2. Docker Infrastructure Configuration

We use **Docker Compose** to orchestrate the three core infrastructure components:
1.  **Airflow**: For scheduling daily data ingestion and model retraining.
2.  **Qdrant**: For vector storage (semantic search).
3.  **PostgreSQL**: For Airflow metadata and structured financial data.

### Prerequisites
- Docker Desktop installed
- 4GB+ RAM allocated to Docker

### A. Environment Variables (`.env`)
Create a `.env` file in the project root:

```ini
# Project Settings
PROJECT_NAME=AgenticAnalyst
AIRFLOW_UID=50000

# Postgres Credentials
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Qdrant Settings
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### B. Docker Compose File (`docker-compose.yml`)
Create this file in the project root. It sets up the "Ingestion Layer" backend.

```yaml
version: '3.8'

x-airflow-common:
  &airflow-common
  build: ./docker
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW_UID: ${AIRFLOW_UID}
  volumes:
    - ./ingestion/dags:/opt/airflow/dags
    - ./data/logs:/opt/airflow/logs
    - ./ingestion/etl:/opt/airflow/etl
  depends_on:
    &airflow-common-depends-on
    postgres:
      condition: service_healthy

services:
  # 1. Vector Database (Qdrant)
  qdrant:
    image: qdrant/qdrant:latest
    container_name: fyp-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
    restart: always

  # 2. Structured Database (PostgreSQL)
  postgres:
    image: postgres:15
    container_name: fyp-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - ./data/postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    restart: always

  # 3. Airflow Webserver (UI)
  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always

  # 4. Airflow Scheduler (Task Runner)
  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    restart: always

  # 5. Airflow Initialization (Run once)
  airflow-init:
    <<: *airflow-common
    command: version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_UPGRADE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: admin
      _AIRFLOW_WWW_USER_PASSWORD: admin
    depends_on:
      postgres:
        condition: service_healthy
```

### C. Custom Dockerfile (`docker/airflow.Dockerfile`)
Create a folder named `docker` and add this file to install necessary Python libraries for your agents.

```dockerfile
FROM apache/airflow:2.8.1

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

### D. Requirements (`docker/requirements.txt`)
Add the core libraries mentioned in your architecture:

```text
apache-airflow-providers-postgres
langchain
langgraph
qdrant-client
openai
pandas
numpy
yfinance
plotly
requests
```

---

## 3. Initialization Steps

Run the following commands in your terminal to start the infrastructure:

1.  **Create Directories:**
    ```bash
    mkdir -p ingestion/dags ingestion/etl data/logs data/postgres_data data/qdrant_storage docker
    ```

2.  **Build & Start:**
    ```bash
    docker-compose up --build -d
    ```

3.  **Verify Access:**
    - **Airflow UI:** [http://localhost:8080](http://localhost:8080) (User: `admin`, Pass: `admin`)
    - **Qdrant API:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
