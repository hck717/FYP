# Troubleshooting Guide for FYP Deployment

This guide covers common issues and solutions for deploying and running the Agentic Investment Analyst (AIPM) platform.

## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)
2. [Python & Virtual Environment](#python--virtual-environment)
3. [Docker Issues](#docker-issues)
4. [Database Connectivity](#database-connectivity)
5. [API & LLM Issues](#api--llm-issues)
6. [Orchestration & Agents](#orchestration--agents)
7. [Streamlit UI](#streamlit-ui)
8. [Performance & Optimization](#performance--optimization)
9. [Advanced Debugging](#advanced-debugging)

---

## Prerequisites & Installation

### Issue: "Python 3 command not found"

**Symptoms:**
```bash
$ python3 --version
command not found: python3
```

**Solution:**
1. Install Python 3.11+ from [python.org](https://www.python.org/)
2. On macOS, use Homebrew: `brew install python@3.11`
3. On Ubuntu/Debian: `sudo apt-get install python3.11 python3.11-venv`
4. Verify: `python3 --version` (should show 3.11+)

### Issue: "Docker not installed or not running"

**Symptoms:**
```bash
$ docker ps
Cannot connect to Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
1. Install Docker from [docker.com](https://www.docker.com/products/docker-desktop)
2. Start Docker Desktop (or Docker daemon on Linux)
3. Test: `docker ps` (should show running containers or be empty)

### Issue: "Insufficient disk space"

**Symptoms:**
```
Docker build fails with: No space left on device
```

**Solution:**
1. Check available space: `df -h`
2. Minimum required: 20GB
3. Clean up Docker: `docker system prune -a` (removes unused images)
4. Or run on external drive: `docker -D info` to check storage path

---

## Python & Virtual Environment

### Issue: "venv creation fails"

**Symptoms:**
```bash
$ python3 -m venv .venv
Error: Command '[...] -m venv [...]' returned non-zero exit status 1
```

**Solution:**
```bash
# On Ubuntu/Debian
sudo apt-get install python3.11-venv

# On macOS
brew install python@3.11

# Retry venv creation
python3 -m venv .venv
```

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Symptoms:**
```bash
$ python -c "import langchain"
ModuleNotFoundError: No module named 'langchain'
```

**Solution:**
```bash
# Verify venv is activated
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\Scripts\activate

# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import langchain; print(langchain.__version__)"
```

### Issue: "venv path contains spaces or special characters"

**Symptoms:**
```bash
$ source ./My Project/.venv/bin/activate
bash: ./My Project/.venv/bin/activate: No such file or directory
```

**Solution:**
- Use proper quoting: `source "./My Project/.venv/bin/activate"`
- Or better: use path without spaces: `mv "My Project" my_project`

### Issue: "pip install hangs or times out"

**Symptoms:**
```
Collecting package... (hangs for >5 minutes)
```

**Solution:**
```bash
# Increase timeout and use faster index
pip install --default-timeout=1000 -i https://pypi.org/simple/ -r requirements.txt

# Or use a mirror
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

---

## Docker Issues

### Issue: "docker compose: command not found"

**Symptoms:**
```bash
$ docker compose up
docker: command not found
```

**Solution:**
```bash
# Check Docker Compose version
docker compose version

# If not found, upgrade Docker Desktop (includes Compose v2)
# Or install standalone: https://docs.docker.com/compose/install/

# Use legacy `docker-compose` (with hyphen) if needed
docker-compose up -d
```

### Issue: "Docker build fails with 'network timeout'"

**Symptoms:**
```
ERROR: Service 'airflow-webserver' failed to build
... temporary failure in name resolution ...
```

**Solution:**
```bash
# Check network connectivity
ping google.com

# For macOS/Linux:
docker run --rm busybox nslookup docker.io

# Try again with --build-arg
docker compose build --no-cache

# Or specify DNS
docker run --dns 8.8.8.8 --rm -it ubuntu bash
```

### Issue: "Port already in use"

**Symptoms:**
```
ERROR: bind: address already in use
... port 8501 is already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill it
kill -9 <PID>

# Or use different port
streamlit run ui/app.py --server.port 8502
```

### Issue: "Service keeps restarting"

**Symptoms:**
```bash
$ docker compose ps
fyp-postgres       Up 1 second (restarting)
```

**Solution:**
```bash
# Check logs
docker compose logs fyp-postgres

# Common cause: Health check failed
# Stop and inspect
docker compose down
docker compose up -d
sleep 30  # Wait longer for startup
docker compose logs
```

### Issue: "Docker volume not mounting correctly"

**Symptoms:**
```
data/ directory appears empty inside container
```

**Solution:**
```bash
# Check volume mounts
docker inspect fyp-postgres | grep -A 10 "Mounts"

# Remove and recreate volumes
docker compose down -v  # Warning: deletes data!
docker compose up -d

# Or manually recreate directory
mkdir -p data/postgres_data
chmod 777 data/postgres_data
```

---

## Database Connectivity

### Issue: "PostgreSQL connection refused"

**Symptoms:**
```bash
$ psql -h localhost -U airflow -d airflow
psql: error: could not connect to server: Connection refused
```

**Solution:**
```bash
# Check container is running
docker compose ps | grep postgres

# Check healthcheck status
docker exec fyp-postgres pg_isready -U airflow

# View logs
docker compose logs fyp-postgres

# Restart
docker compose restart fyp-postgres
```

### Issue: "PostgreSQL password authentication failed"

**Symptoms:**
```
FATAL: password authentication failed for user "airflow"
```

**Solution:**
```bash
# Verify credentials in .env
grep POSTGRES_PASSWORD .env

# Reset (warning: deletes data)
docker compose down -v
docker compose up -d

# Or connect with correct password
PGPASSWORD=airflow psql -h localhost -U airflow -d airflow
```

### Issue: "Neo4j connection timeout"

**Symptoms:**
```
java.io.IOException: Failed to connect to server address: localhost:7687
```

**Solution:**
```bash
# Check Neo4j is running
docker compose ps | grep neo4j

# Check port
curl http://localhost:7474/db/data/

# Verify credentials
curl -u neo4j:SecureNeo4jPass2025! http://localhost:7474/db/data/

# Check logs
docker compose logs fyp-neo4j
```

### Issue: "pgvector extension not available"

**Symptoms:**
```
ERROR: relation "text_chunks" does not exist
```

**Solution:**
```bash
# Verify pgvector image is being used
docker compose config | grep -A 2 "postgres:"

# Should show: pgvector/pgvector:pg15

# Reset database
docker compose down -v
docker compose up -d fyp-postgres
sleep 30

# Verify extension
docker exec fyp-postgres psql -U airflow -d airflow -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## API & LLM Issues

### Issue: "EODHD API rate limit exceeded"

**Symptoms:**
```
Error: Rate limit exceeded. Current limit: 120 calls/minute
```

**Solution:**
1. Check your API key tier at [eodhd.com](https://eodhd.com)
2. Upgrade to higher tier for more calls
3. Add delay between requests in ingestion code
4. Use cached data instead of live API calls

### Issue: "DeepSeek API key invalid or expired"

**Symptoms:**
```
Error: 401 Unauthorized - Invalid API key
```

**Solution:**
```bash
# Verify key in .env
grep DEEPSEEK_API_KEY .env

# Get fresh key from https://platform.deepseek.com/

# Update .env
sed -i 's/DEEPSEEK_API_KEY=.*/DEEPSEEK_API_KEY=sk-your-new-key/' .env

# Restart containers
docker compose down
docker compose up -d
```

### Issue: "Ollama models not pulling"

**Symptoms:**
```
Error: Failed to pull model nomic-embed-text:v1.5
```

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Manually pull models
docker exec fyp-ollama ollama pull nomic-embed-text:v1.5
docker exec fyp-ollama ollama pull llama3.2:3b

# Check available models
docker exec fyp-ollama ollama list

# View Ollama logs
docker compose logs fyp-ollama
```

### Issue: "Embedding generation is very slow"

**Symptoms:**
```
Taking >30 seconds per embedding
```

**Solution:**
```bash
# Check Ollama resource usage
docker stats fyp-ollama

# For GPU support, uncomment in docker-compose.yml:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]

# Or use lighter embedding model
# In .env: EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Issue: "Web search (Perplexity) not working"

**Symptoms:**
```
Error: Web search agent failed - API error
```

**Solution:**
```bash
# Verify API key
grep PERPLEXITY_API_KEY .env

# Test connection
curl -H "Authorization: Bearer YOUR_KEY" \
  "https://api.perplexity.ai/api/v1/list_models"

# If error, get fresh key from https://www.perplexity.ai/api/
```

---

## Orchestration & Agents

### Issue: "ImportError: cannot import name 'run' from 'orchestration.graph'"

**Symptoms:**
```bash
from orchestration.graph import run
ImportError: cannot import name 'run'
```

**Solution:**
```bash
# Verify file exists
ls orchestration/graph.py

# Check PYTHONPATH
echo $PYTHONPATH

# Set correct path
export PYTHONPATH=/path/to/FYP:$PYTHONPATH

# Try import in Python
cd /path/to/FYP
source .venv/bin/activate
python -c "from orchestration.graph import run; print('OK')"
```

### Issue: "Agent timeout or hanging"

**Symptoms:**
```
Command hangs, never returns
```

**Solution:**
```bash
# Set timeout limits in .env
BUSINESS_ANALYST_MAX_TOKENS=16000  # Lower from 32000
BA_MAX_REWRITE_LOOPS=2              # Lower from 3
QUANT_AGENT_SQL_TIMEOUT=15          # Lower from 30

# Restart and try again
docker compose restart
```

### Issue: "Neo4j graph queries returning no results"

**Symptoms:**
```
Business Analyst agent returns no retrieved chunks
```

**Solution:**
```bash
# Check if data was ingested
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! \
  "MATCH (n) RETURN COUNT(n)"

# If 0, run ingestion DAG
# In Airflow UI (http://localhost:8080):
# - Find DAG: dag_eodhd_ingestion_unified
# - Trigger manually
# - Wait for completion

# Or manually check
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! \
  "MATCH (c:Chunk) RETURN c.ticker, COUNT(c) GROUP BY c.ticker"
```

### Issue: "Quant Fundamental agent calculation errors"

**Symptoms:**
```
ValueError: Cannot calculate metric - missing data
```

**Solution:**
```bash
# Verify PostgreSQL has financial data
docker exec fyp-postgres psql -U airflow -d airflow -c \
  "SELECT DISTINCT ticker_symbol FROM raw_fundamentals LIMIT 5;"

# If empty, run FMP ingestion DAG first
# Or manually load test data
```

---

## Streamlit UI

### Issue: "Streamlit container crashes with 'cat: /tmp/ngrok.log: No such file or directory'"

**Symptoms:**
```
fyp-streamlit | cat: /tmp/ngrok.log: No such file or directory
Container restarting (1) repeatedly
```

**Solution (Applied in v2.0+):**
The ngrok tunnel requirement has been removed from the Streamlit startup command. The container now runs in headless mode locally without external tunnel.

If you encounter this issue on an older version:
```bash
# Edit docker-compose.yml and find the streamlit service command (line ~287)
# Replace the full command with:
sh -c "pip install -q streamlit && streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0"

# Remove these lines from the command:
# - pip install -q pyngrok streamlit (remove pyngrok)
# - ngrok http 8501 --log=stdout > /tmp/ngrok.log 2>&1 &
# - sleep 3 &&
# - cat /tmp/ngrok.log &&

# Then restart
docker compose restart fyp-streamlit
```

### Issue: "Streamlit app crashes on startup"

**Symptoms:**
```
StreamlitAPIException: connection failed
```

**Solution:**
```bash
# Check Streamlit container logs
docker compose logs fyp-streamlit

# Verify database connections
docker exec fyp-streamlit python -c \
  "import psycopg2; psycopg2.connect('dbname=airflow user=airflow host=postgres')"

# Restart UI
docker compose restart fyp-streamlit
docker compose logs -f fyp-streamlit
```

### Issue: "Streamlit page not loading"

**Symptoms:**
```
ERR! Could not connect to Streamlit server
```

**Solution:**
```bash
# Verify Streamlit is running
docker compose ps | grep streamlit

# Check port
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 8501

# Access directly
curl http://localhost:8501

# If still not working, restart
docker compose restart fyp-streamlit
sleep 5
# Try again at http://localhost:8501
```

### Issue: "Charts/graphs not displaying"

**Symptoms:**
```
Empty Plotly charts or missing visualizations
```

**Solution:**
```bash
# Check if data is being returned
# In Streamlit app, add debug output

# Clear browser cache: Ctrl+Shift+Delete
# Or hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)

# Restart Streamlit
docker compose restart fyp-streamlit
```

---

## Performance & Optimization

### Issue: "Orchestration queries are slow (>2 minutes)"

**Symptoms:**
```
Agent execution taking very long
```

**Solution:**
1. **Database indices**: Ensure PostgreSQL indices exist
   ```bash
   docker exec fyp-postgres psql -U airflow -d airflow -c "\di"
   ```

2. **Reduce max chunks**:
   ```bash
   # In .env
   BUSINESS_ANALYST_MAX_CHUNKS=30  # Lower from 50
   ```

3. **Use database caching** for frequently queried tickers

4. **Check system resources**:
   ```bash
   docker stats
   ```

### Issue: "High memory usage"

**Symptoms:**
```
Docker containers using >80% available RAM
```

**Solution:**
```bash
# Check memory limits
docker inspect fyp-postgres | grep -i memory

# Set memory limits in docker-compose.yml:
# services:
#   postgres:
#     deploy:
#       resources:
#         limits:
#           memory: 4G

# Reduce Neo4j heap:
NEO4J_server_memory_heap_max__size=512m  # In docker-compose.yml

# Restart
docker compose down
docker compose up -d
```

---

## Advanced Debugging

### Enable verbose logging

```bash
# In .env
LOG_LEVEL=DEBUG

# Restart
docker compose restart
```

### Capture full logs

```bash
# Save all logs to file
docker compose logs > deployment.log 2>&1

# View specific service
docker compose logs fyp-postgres > postgres.log

# Real-time monitoring
docker compose logs -f --tail=50
```

### Connect to containers directly

```bash
# Python shell in Airflow
docker exec -it fyp-airflow-webserver python

# PostgreSQL client
docker exec -it fyp-postgres psql -U airflow -d airflow

# Neo4j shell
docker exec -it fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025!

# Bash shell
docker exec -it fyp-postgres /bin/bash
```

### Reset everything (nuclear option)

```bash
# WARNING: This deletes all data!
docker compose down -v
rm -rf data/
docker system prune -a

# Recreate from scratch
docker compose up -d --build
```

---

## Getting Help

1. **Check logs first**: `docker compose logs -f`
2. **Run validation**: `./scripts/validate-deployment.sh`
3. **Review documentation**: `docs/FRESH_DEPLOYMENT.md`
4. **Search GitHub issues**: https://github.com/your-org/FYP/issues
5. **Create new issue** with:
   - Error message (full output)
   - Output of: `docker compose ps`
   - Output of: `docker compose logs`
   - Your OS and Docker version

---

**Last Updated**: 2026-05-03
**Tested On**: macOS (M1/M2/Intel), Ubuntu 22.04, Docker 24.0+
