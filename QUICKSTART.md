# Quick Start: Deploy FYP in 2 Minutes

> **For new users deploying FYP for the first time**

This is a TL;DR overview. For complete details, see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

## Prerequisites (Verify)

```bash
python3 --version      # Should be 3.11+
docker --version       # Should be 24.0+
git --version          # Should be present
```

Need to install? → [Full guide here](docs/DEPLOYMENT_GUIDE.md#prerequisites)

## One-Command Setup ⭐ (Recommended)

```bash
# 1. Clone
git clone https://github.com/your-org/FYP.git
cd FYP

# 2. Run bootstrap (handles EVERYTHING)
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh

# 3. Access at http://localhost:8501
```

**Total time: 10-15 minutes** (first run with downloads)

That's it! The bootstrap script will:
- ✓ Validate prerequisites
- ✓ Set up Python virtual environment
- ✓ Install dependencies
- ✓ Start all Docker services
- ✓ Run health checks
- ✓ Display access URLs

## Manual Setup (If Preferred)

```bash
# 1. Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Start services
docker compose up -d --build

# 4. Verify
docker compose ps
./scripts/validate-deployment.sh
```

---

## Test the Deployment

### Option A: Test with Python CLI

```bash
source .venv/bin/activate
python - <<'EOF'
from orchestration.graph import run

result = run("What is Apple's recent stock performance?")
print(result["final_summary"])
EOF
```

### Option B: Use Streamlit Web UI

```bash
streamlit run ui/app.py
# Open: http://localhost:8501
```

### Option C: Check Health Status

```bash
./scripts/validate-deployment.sh
# Expected: 40+ PASS, minimal WARN, 0 FAIL
```

---

## Local Dashboards (When Running)

| Service | URL | Login |
|---------|-----|-------|
| **Streamlit UI** ⭐ | http://localhost:8501 | None |
| **Airflow DAGs** | http://localhost:8080 | admin / admin |
| **Neo4j Browser** | http://localhost:7474 | neo4j / SecureNeo4jPass2025! |
| **PostgreSQL** | localhost:5432 | airflow / airflow |

---

## Configuration (Optional)

The `.env` file has **working defaults** for local development. To customize:

```bash
# Edit configuration
nano .env
```

Key settings:
```env
# Financial APIs (optional - defaults work for testing)
EODHD_API_KEY=your-key           # https://eodhd.com
FMP_API_KEY=your-key             # https://financialmodelingprep.com
DEEPSEEK_API_KEY=your-key        # https://platform.deepseek.com

# Databases (defaults work for local)
POSTGRES_HOST=localhost
NEO4J_URI=bolt://localhost:7687
```

Then restart services:
```bash
docker compose down
docker compose up -d
```

---

## Troubleshooting Quick Fixes

```bash
# Services stuck?
docker compose down
docker compose up -d

# See what's happening
docker compose logs -f

# Reset everything (deletes data!)
docker compose down -v
docker compose up -d --build

# Full diagnostics
./scripts/validate-deployment.sh

# Need more help?
cat TROUBLESHOOTING.md
```

---

## What's Running?

```
┌─ Streamlit UI (port 8501) ─┐
│  Web interface for analysis │
└────────────────┬────────────┘
                 │
┌────────────────▼──────────────────────┐
│  Orchestration Layer (LangGraph)      │
│  7 specialized AI agents in parallel  │
└────────────────┬───────────────────────┘
                 │
     ┌────────────┼────────────┐
     │            │            │
  ┌──▼──┐   ┌────▼────┐   ┌───▼──┐
  │ SQL │   │ Neo4j   │   │Ollama│
  │ DB  │   │ Graph   │   │ LLM  │
  └─────┘   └─────────┘   └──────┘
```

---

## Next Steps

1. **Explore agents** - Check `agents/` folder
2. **Customize prompts** - Edit agent instructions
3. **Review docs** - Read [README.md](README.md)
4. **Run tests** - `pytest tests/ -v`
5. **Deploy to cloud** - See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md#cloud-deployment)

---

## Full Documentation

- **Complete Deployment Guide**: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Architecture & Design**: [README.md](README.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Agent Details**: [orchestration/README.md](orchestration/README.md)

---

## Need Help?

- **Setup Issues** → [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Errors/Bugs** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Questions** → Create [GitHub issue](https://github.com/your-org/FYP/issues)

---

**Ready?** Run `./scripts/bootstrap.sh` to get started! 🚀

**Last Updated**: May 3, 2026 | **Status**: ✓ Production Ready
