# Complete Deployment Guide: FYP (Agentic Investment Analyst)

> **For new users deploying FYP from a fresh clone**

This comprehensive guide covers local setup, cloud deployment, and public sharing options. Choose your deployment path below.

## Table of Contents

1. [Quick Navigation](#quick-navigation)
2. [Prerequisites](#prerequisites)
3. [Deployment Paths](#deployment-paths)
4. [Local Setup (Step-by-Step)](#local-setup-step-by-step)
5. [Verification & Testing](#verification--testing)
6. [Cloud Deployment](#cloud-deployment)
7. [Public Sharing with ngrok](#public-sharing-with-ngrok)
8. [Troubleshooting](#troubleshooting)

---

## Quick Navigation

### I want to start immediately (5 minutes)
→ Skip to [Automated Setup](#automated-setup) or [One-Command Setup](#one-command-setup)

### I want to understand each step (30 minutes)
→ Follow [Local Setup (Step-by-Step)](#local-setup-step-by-step)

### I want to deploy to production
→ Jump to [Cloud Deployment](#cloud-deployment)

### I want to share publicly from my machine
→ See [Public Sharing with ngrok](#public-sharing-with-ngrok)

### Something's broken
→ Check [Troubleshooting](#troubleshooting) or [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

## Prerequisites

### System Requirements

Ensure you have the following installed:

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Docker & Docker Compose** - [Install](https://docs.docker.com/engine/install/)
  - Minimum 8GB RAM, 20GB disk space recommended
  - On macOS/Windows, enable Docker Desktop with sufficient resources
- **Git** - [Install](https://git-scm.com/)
- **curl** (usually pre-installed)

### Verify Prerequisites

```bash
python3 --version      # Should be 3.11+
docker --version       # Should be 24.0+
docker compose version # Should be v2.x
git --version          # Any recent version
```

### Storage & Memory

```bash
# Check available disk space (should have 20GB+)
df -h /

# Check available memory (should have 8GB+)
free -h  # Linux/macOS
# or: vm_stat  # macOS
```

---

## Deployment Paths

### Path 1: Fastest (Automated) ⭐ Recommended

```bash
# Clone and run bootstrap
git clone https://github.com/your-org/FYP.git
cd FYP
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

**Time: 15 minutes | Difficulty: Easy**
- Fully automated
- Interactive prompts
- Auto-validates deployment

### Path 2: Step-by-Step (Manual) 🎓 Learning

Follow the [Local Setup (Step-by-Step)](#local-setup-step-by-step) section below.

**Time: 30 minutes | Difficulty: Medium**
- Detailed explanations
- Full control
- Learn each step

### Path 3: Minimal (For Experienced Users) ⚡ Quick

```bash
git clone https://github.com/your-org/FYP.git
cd FYP
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d --build
```

**Time: 5-10 minutes | Difficulty: Hard**
- No guidance
- Assumes Docker/Python knowledge

---

## Local Setup (Step-by-Step)

### Step 1: Clone the Repository

```bash
# Choose a working directory
cd ~/projects  # or your preferred location

# Clone the repository
git clone https://github.com/your-org/FYP.git
cd FYP

# Verify structure
ls -la
```

Expected structure:
```
FYP/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env                 ← Configuration with defaults
├── .env.example         ← Optional: template reference
├── scripts/
├── docker/
├── agents/
├── orchestration/
├── ingestion/
├── ui/
├── data/
└── tests/
```

### Step 2: Configure Environment Variables

The `.env` file includes **working defaults** for local development. For production, customize as needed:

```bash
# View current configuration
cat .env

# Edit if needed (optional - defaults work for testing)
nano .env    # or vim, code, etc.
```

**Key variables to customize:**

```env
# Financial Data APIs (optional - defaults work, better with own keys)
EODHD_API_KEY=your-key-here           # https://eodhd.com
FMP_API_KEY=your-key-here             # https://financialmodelingprep.com
DEEPSEEK_API_KEY=your-key-here        # https://platform.deepseek.com

# Database credentials (defaults work for local)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=SecureNeo4jPass2025!
```

### Step 3: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows (PowerShell):
# .venv\Scripts\Activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat

# Verify activation (prompt should show (.venv))
python --version  # Should show 3.11+
```

### Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list | head -20
```

This installs:
- **LangChain, LangGraph** - Orchestration framework
- **Streamlit** - Web interface
- **FastAPI** - API layer
- **SQLAlchemy, Neo4j driver** - Database clients
- **pytest** - Testing framework
- Plus 10+ data science dependencies

### Step 5: Start Docker Services

The `docker-compose.yml` defines all services:
- **PostgreSQL** (with pgvector)
- **Neo4j** (knowledge graph)
- **Ollama** (embeddings/LLM)
- **Airflow** (DAG orchestration)
- **Streamlit** (web UI)

#### Build and Start

```bash
# Build and start in background (-d = detached)
docker compose up -d --build

# This will:
# 1. Build Airflow image (5-10 mins first time)
# 2. Start PostgreSQL → Neo4j → Ollama → Airflow
# 3. Initialize Airflow DB
# 4. Pull Ollama models

# Monitor progress
docker compose logs -f

# Exit log monitoring (Ctrl+C)
```

#### Verify All Services Are Running

```bash
# Check container status
docker compose ps

# Expected output:
NAME              STATUS          PORTS
fyp-postgres      Up (healthy)    5432
fyp-neo4j         Up (healthy)    7474, 7687
fyp-ollama        Up              11434
fyp-airflow-*     Up              8080
fyp-streamlit     Up              8501
```

### Step 6: Verify Data Layer

Ensure databases are initialized and accessible:

```bash
# PostgreSQL: List tables
docker exec fyp-postgres psql -U airflow -d airflow -c "\dt"

# Expected: Tables like raw_timeseries, raw_fundamentals, sentiment_trends

# Neo4j: Check connectivity
curl -X GET http://localhost:7474/db/data/ \
  -u neo4j:SecureNeo4jPass2025!

# Expected: JSON response with neo4j version info
```

### Step 7: Run Orchestration Test

Test the core workflow without UI:

```bash
# Activate venv if not already active
source .venv/bin/activate

# Run a simple query
python - <<'EOF'
from orchestration.graph import run

# This runs the full orchestration pipeline
# May take 30-60s depending on model availability
result = run("What is Apple's recent stock performance?")

print("\n" + "="*60)
print("ORCHESTRATION RESULT")
print("="*60)
print(result.get("final_summary", "No summary"))
print("="*60)
EOF
```

**Expected output:** Multi-agent analysis report with citations and metrics.

### Step 8: Launch Streamlit UI

```bash
# From FYP root directory with venv active
streamlit run ui/app.py

# Output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://xxx.xxx.xxx.xxx:8501
```

**Access the UI:**
- Open browser → `http://localhost:8501`
- Input a ticker (e.g., `AAPL`)
- Click "Analyze"
- View results with interactive charts

To stop: Press `Ctrl+C` in terminal.

---

## Verification & Testing

### Health Check System

Use our automated validation script:

```bash
# Run comprehensive health checks
./scripts/validate-deployment.sh

# Expected output:
# ════════════════════════════════════════════════════════════
# Summary: 45+ passed | 3 warned | 0 failed
# ════════════════════════════════════════════════════════════
# ✓ All systems operational
```

### Manual Verification

```bash
# Check all services running
docker compose ps

# Check specific service logs
docker compose logs fyp-postgres
docker compose logs fyp-neo4j
docker compose logs fyp-ollama

# Test database connections
docker exec fyp-postgres pg_isready -U airflow

# Test Neo4j API
curl -s http://localhost:7474/db/data/ | jq .

# Test Ollama
curl -s http://localhost:11434/api/tags | jq .
```

### Access Local Dashboards

| Service | URL | Login |
|---------|-----|-------|
| **Streamlit UI** | http://localhost:8501 | None |
| **Airflow DAGs** | http://localhost:8080 | admin / admin |
| **Neo4j Browser** | http://localhost:7474 | neo4j / SecureNeo4jPass2025! |
| **PostgreSQL** | localhost:5432 | airflow / airflow |
| **Ollama API** | http://localhost:11434 | None |

### Run Test Suite

```bash
# Install test dependencies (if not in requirements.txt)
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/integration/ -v -m integration
pytest tests/prompts/ -v -m prompt

# Expected: ~141 tests, 100% pass rate
```

---

## Cloud Deployment

### AWS, GCP, DigitalOcean Setup

For production deployment on a cloud VM:

#### Requirements

- **Compute:** Linux VM (Ubuntu 22.04+) with 4-8 vCPUs, 16GB RAM
- **Storage:** 50GB+ SSD
- **Network:** Ports 8501 (Streamlit), 8080 (Airflow), 7474 (Neo4j) open in firewall

#### Step-by-Step

1. **Provision VM** on your cloud provider
2. **Install Docker:**
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose git
   sudo usermod -aG docker $USER
   ```

3. **Clone Repository:**
   ```bash
   git clone https://github.com/your-org/FYP.git
   cd FYP
   ```

4. **Create .env for Production:**
   ```bash
   # Copy template
   cp .env.example .env  # or keep existing .env with test keys
   
   # Edit with your production API keys
   nano .env
   ```

5. **Start Infrastructure:**
   ```bash
   docker compose up -d --build
   
   # Monitor startup
   docker compose logs -f
   ```

6. **Verify Deployment:**
   ```bash
   docker compose ps
   ./scripts/validate-deployment.sh
   ```

7. **Access Application:**
   Navigate to `http://<your-vm-public-ip>:8501`

#### Configure SSL/TLS (Optional but Recommended)

```bash
# Install certbot for Let's Encrypt
sudo apt install -y certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Use certificate in nginx reverse proxy
# (See your cloud provider's documentation)
```

#### Scaling Considerations

- **Load Balancing:** Use cloud provider's load balancer
- **Database Replication:** Use managed services (RDS, AuraDB)
- **Container Orchestration:** Consider Kubernetes for production
- **Monitoring:** Set up CloudWatch, Datadog, or similar

### Streamlit Cloud (Frontend Only)

Host just the frontend on Streamlit Cloud with external databases:

1. **Set up external databases:**
   - PostgreSQL: AWS RDS, Azure Database, or similar
   - Neo4j: AuraDB (neo4j.com)
   - Update `.env` with remote connection strings

2. **Push to GitHub:** Ensure repo is public

3. **Deploy to Streamlit Cloud:**
   - Visit https://share.streamlit.io
   - Connect GitHub repo
   - Set environment variables in Streamlit Cloud Secrets
   - Deploy

**Limitations:** Requires managed database services (not local containers)

---

## Public Sharing with ngrok

Share your local development environment publicly without cloud deployment:

### Prerequisites

1. **ngrok Account:** https://dashboard.ngrok.com (free tier available)
2. **ngrok Auth Token:** From your dashboard
3. **Docker & Services Running:** From Steps 5 above

### Quick Start

```bash
# Set your ngrok token (get from https://dashboard.ngrok.com)
export NGROK_AUTHTOKEN=your_token_here

# Start all services (ngrok configured in docker-compose.yml)
docker compose up -d

# Check services started
docker compose ps

# Get public URL from logs
docker logs fyp-streamlit | grep "External URL"

# Or check ngrok dashboard
open http://localhost:4040
```

### Manual ngrok Setup

```bash
# Terminal 1: Start ngrok tunnel
ngrok http 8501

# Terminal 2: Start Streamlit
source .venv/bin/activate
streamlit run ui/app.py

# Access ngrok dashboard
open http://localhost:4040
```

### Share Your Public URL

1. Open http://localhost:4040
2. Copy the public URL (e.g., `https://abc123.ngrok.io`)
3. Share with anyone!

### Security Considerations

⚠️ **Important when sharing:**

1. **API Keys Exposed** - Anyone with the URL can use your DeepSeek API key
   - Use a separate key with rate limits
   - Monitor API usage closely
   - Delete tunnel when done

2. **No Authentication** - Consider adding password protection:
   ```bash
   # Add to streamlit config
   streamlit config set client.requireUserInput false
   ```

3. **Data Accessible** - Your local databases are accessible through the tunnel
   - Don't share sensitive data
   - Use test datasets only

4. **Monitoring** - Set up logging and alerts for API usage

### Disable ngrok When Done

```bash
# Stop all services
docker compose down

# Kill any remaining ngrok processes
pkill ngrok
```

---

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker version

# Check logs for specific service
docker compose logs fyp-postgres
docker compose logs fyp-neo4j
docker compose logs fyp-ollama

# Restart all services
docker compose down --volumes  # Warning: deletes data!
docker compose up -d --build
```

### Python Import Errors

```bash
# Ensure venv is activated
source .venv/bin/activate

# Verify PYTHONPATH
export PYTHONPATH=/path/to/FYP:$PYTHONPATH

# Try importing
python - <<'EOF'
from orchestration.graph import run
EOF
```

### Ollama Models Not Pulling

```bash
# Manually pull models
docker exec fyp-ollama ollama pull nomic-embed-text:v1.5
docker exec fyp-ollama ollama pull llama3.2:3b

# Verify models exist
docker exec fyp-ollama ollama list
```

### Port Conflicts

```bash
# Find what's using port 8501 (example)
lsof -i :8501

# Use different port
streamlit run ui/app.py --server.port 8502
```

### Database Connection Issues

```bash
# Test PostgreSQL
docker exec fyp-postgres pg_isready -U airflow

# Test Neo4j
curl -X GET http://localhost:7474/db/data/ \
  -u neo4j:SecureNeo4jPass2025!

# Check connection strings in .env
cat .env | grep -E "(POSTGRES|NEO4J)"
```

### Disk Space Issues

```bash
# Check available space
df -h

# Clean up Docker
docker system prune -a

# Remove old containers/images
docker compose down -v
```

### Memory Issues

```bash
# Check memory usage
docker stats

# Increase Docker memory allocation
# In Docker Desktop: Settings → Resources → Memory

# Reduce parallelism
docker compose down
# Edit docker-compose.yml: AIRFLOW__CORE__PARALLELISM=2
docker compose up -d
```

### Getting More Help

- **Specific Error?** Search [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- **Docker Issues?** Check [Docker docs](https://docs.docker.com/)
- **LangGraph Issues?** See [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- **Create Issue?** [GitHub Issues](https://github.com/your-org/FYP/issues)

---

## Common Scenarios

### Scenario 1: Fresh Install (Zero Config)

```bash
git clone https://github.com/your-org/FYP.git
cd FYP
./scripts/bootstrap.sh          # Automated setup
./scripts/validate-deployment.sh # Verify
# Access http://localhost:8501
```

**Time: 15 minutes**

### Scenario 2: Learning Setup

```bash
# Follow steps 1-7 in "Local Setup (Step-by-Step)"
# Execute test query (Step 7)
# Launch UI (Step 8)
# Run tests to understand system
pytest tests/ -v
```

**Time: 30 minutes**

### Scenario 3: Production Deployment

1. Provision cloud VM
2. Follow "Cloud Deployment" section
3. Configure SSL/TLS
4. Set up monitoring
5. Deploy with custom API keys

**Time: 1-2 hours**

### Scenario 4: Share with Team

1. Set up ngrok account
2. Start services locally
3. Run: `export NGROK_AUTHTOKEN=...; docker compose up -d`
4. Share public URL from `http://localhost:4040`

**Time: 10 minutes**

### Scenario 5: Troubleshoot Failures

1. Run: `./scripts/validate-deployment.sh`
2. Note any failures
3. Search [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
4. Apply fix
5. Rerun validation

---

## Next Steps

1. **Explore Agents:** Review code in `agents/` folder
2. **Customize Prompts:** Edit agent instructions
3. **Review Architecture:** Read `README.md` and `orchestration/README.md`
4. **Run Full Tests:** `pytest tests/ -v`
5. **Deploy to Production:** Follow Cloud Deployment section

---

## Support Resources

- **Quick Questions:** See [QUICKSTART.md](../QUICKSTART.md)
- **Errors/Bugs:** Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- **Architecture:** Review [orchestration/README.md](../orchestration/README.md)
- **Community:** [GitHub Issues](https://github.com/your-org/FYP/issues)

---

**Last Updated:** May 3, 2026  
**Version:** 2.0 (Consolidated)  
**Status:** ✓ Production Ready  
**Tested On:** Python 3.11+, Docker 24.0+, macOS/Linux
