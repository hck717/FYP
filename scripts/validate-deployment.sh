#!/bin/bash

################################################################################
# FYP Deployment Validator & Health Check Script
# 
# This script validates that a fresh clone & deploy of FYP is working correctly.
# It checks prerequisites, service health, database connectivity, and runs tests.
#
# Usage:
#   chmod +x scripts/validate-deployment.sh
#   ./scripts/validate-deployment.sh
#
# Or with verbose output:
#   ./scripts/validate-deployment.sh -v
#
# Exit codes:
#   0 = All checks passed
#   1 = Warning (non-critical issues)
#   2 = Critical failure (deployment not ready)
################################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

################################################################################
# Helper Functions
################################################################################

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
  echo -e "${GREEN}[✓ PASS]${NC} $1"
  ((CHECKS_PASSED++))
}

log_fail() {
  echo -e "${RED}[✗ FAIL]${NC} $1"
  ((CHECKS_FAILED++))
}

log_warn() {
  echo -e "${YELLOW}[⚠ WARN]${NC} $1"
  ((CHECKS_WARNED++))
}

verbose_output() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${BLUE}      → $1${NC}"
  fi
}

print_header() {
  echo ""
  echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_footer() {
  echo ""
  echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
  echo -e "Summary: ${GREEN}$CHECKS_PASSED passed${NC} | ${YELLOW}$CHECKS_WARNED warned${NC} | ${RED}$CHECKS_FAILED failed${NC}"
  echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

################################################################################
# SECTION 1: Prerequisites Check
################################################################################

print_header "1. Checking Prerequisites"

# Python
if command -v python3 &> /dev/null; then
  PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
  if [[ "$PY_VERSION" > "3.11" ]] || [[ "$PY_VERSION" == "3.11"* ]]; then
    log_pass "Python 3.11+ found: $PY_VERSION"
    verbose_output "$(command -v python3)"
  else
    log_fail "Python 3.11+ required, found: $PY_VERSION"
  fi
else
  log_fail "Python 3 not installed"
fi

# Docker
if command -v docker &> /dev/null; then
  DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
  log_pass "Docker found: $DOCKER_VERSION"
  verbose_output "$(command -v docker)"
else
  log_fail "Docker not installed"
fi

# Docker Compose
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
  COMPOSE_VERSION=$(docker compose version | awk '{print $NF}')
  log_pass "Docker Compose found: $COMPOSE_VERSION"
else
  log_fail "Docker Compose not installed or Docker not running"
fi

# Git
if command -v git &> /dev/null; then
  GIT_VERSION=$(git --version | awk '{print $3}')
  log_pass "Git found: $GIT_VERSION"
else
  log_fail "Git not installed"
fi

# Disk space
DISK_AVAILABLE=$(df . | awk 'NR==2 {print int($4/1024/1024)}'  ) # MB
if [ "$DISK_AVAILABLE" -gt 20480 ]; then
  log_pass "Disk space: ${DISK_AVAILABLE}MB available (>20GB required)"
else
  log_warn "Low disk space: ${DISK_AVAILABLE}MB available (<20GB recommended)"
fi

# Memory (if available)
if command -v free &> /dev/null; then
  MEM_AVAILABLE=$(free -m | awk 'NR==2 {print $7}')
  if [ "$MEM_AVAILABLE" -gt 8192 ]; then
    log_pass "Memory: ${MEM_AVAILABLE}MB available (>8GB)"
  else
    log_warn "Low memory: ${MEM_AVAILABLE}MB available (<8GB recommended)"
  fi
fi

################################################################################
# SECTION 2: Repository Structure
################################################################################

print_header "2. Checking Repository Structure"

# Check critical directories
REQUIRED_DIRS=(
  "agents"
  "orchestration"
  "ingestion"
  "ui"
  "docker"
  "data"
  "tests"
  "docs"
)

for dir in "${REQUIRED_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    log_pass "Directory exists: $dir/"
  else
    log_fail "Missing directory: $dir/"
  fi
done

# Check critical files
REQUIRED_FILES=(
  "README.md"
  "docker-compose.yml"
  "requirements.txt"
  ".env"
  "pyproject.toml"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [ -f "$file" ]; then
    log_pass "File exists: $file"
  else
    log_fail "Missing file: $file"
  fi
done

################################################################################
# SECTION 3: Environment & Python Setup
################################################################################

print_header "3. Checking Python Environment"

# Check venv
if [ -d ".venv" ]; then
  log_pass "Virtual environment exists: .venv/"
  
  # Check if activated
  if [ -n "$VIRTUAL_ENV" ] && [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
    log_pass "Virtual environment is ACTIVATED"
  else
    log_warn "Virtual environment exists but NOT activated"
    verbose_output "Activate with: source .venv/bin/activate"
  fi
else
  log_warn "Virtual environment not created (.venv/)"
  verbose_output "Create with: python3.11 -m venv .venv && source .venv/bin/activate"
fi

# Check requirements installed
if [ -n "$VIRTUAL_ENV" ]; then
  if python -c "import langchain, langchain_community, langgraph" 2>/dev/null; then
    log_pass "Core dependencies installed (langchain, langgraph)"
  else
    log_warn "Some dependencies may not be installed"
    verbose_output "Install with: pip install -r requirements.txt"
  fi
else
  log_warn "Cannot check dependencies (venv not activated)"
fi

################################################################################
# SECTION 4: Docker Services Health
################################################################################

print_header "4. Checking Docker Services"

# Check if Docker is running
if ! docker ps &> /dev/null; then
  log_fail "Docker daemon not running or no permissions"
  echo -e "${YELLOW}Please start Docker Desktop or Docker daemon and try again${NC}"
else
  log_pass "Docker daemon is running"
  
  # Check running containers
  CONTAINERS=(
    "fyp-postgres:PostgreSQL"
    "fyp-neo4j:Neo4j"
    "fyp-ollama:Ollama"
    "fyp-airflow-webserver:Airflow Webserver"
    "fyp-airflow-scheduler:Airflow Scheduler"
    "fyp-streamlit:Streamlit UI"
  )
  
  for container_info in "${CONTAINERS[@]}"; do
    IFS=':' read -r container_name service_name <<< "$container_info"
    
    if docker ps --filter "name=$container_name" --format "{{.Names}}" | grep -q "$container_name"; then
      STATUS=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "running")
      
      if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "running" ]; then
        log_pass "$service_name is running (status: $STATUS)"
      else
        log_warn "$service_name is running but status: $STATUS"
      fi
    else
      log_fail "$service_name not running"
      verbose_output "Start services with: docker compose up -d --build"
    fi
  done
fi

################################################################################
# SECTION 5: Database Connectivity
################################################################################

print_header "5. Checking Database Connectivity"

# PostgreSQL
if command -v psql &> /dev/null || docker ps --filter "name=fyp-postgres" --format "{{.Names}}" | grep -q "fyp-postgres"; then
  if docker exec fyp-postgres pg_isready -U airflow &> /dev/null; then
    log_pass "PostgreSQL is accessible"
    
    # Check tables
    TABLES=$(docker exec fyp-postgres psql -U airflow -d airflow -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null)
    if [ -n "$TABLES" ] && [ "$TABLES" -gt 0 ]; then
      log_pass "PostgreSQL has $TABLES tables initialized"
    else
      log_warn "PostgreSQL tables may not be initialized"
    fi
  else
    log_fail "PostgreSQL is not responding"
  fi
else
  log_fail "PostgreSQL container not found or psql not installed"
fi

# Neo4j
if docker ps --filter "name=fyp-neo4j" --format "{{.Names}}" | grep -q "fyp-neo4j"; then
  if curl -s -u neo4j:SecureNeo4jPass2025! http://localhost:7474/db/data/ &> /dev/null; then
    log_pass "Neo4j is accessible"
    
    # Try to get node count
    NODE_COUNT=$(docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "MATCH (n) RETURN COUNT(n)" 2>/dev/null | tail -1)
    if [ -n "$NODE_COUNT" ]; then
      log_pass "Neo4j has $NODE_COUNT nodes"
    else
      log_warn "Could not query Neo4j node count"
    fi
  else
    log_fail "Neo4j is not responding on port 7474"
  fi
else
  log_fail "Neo4j container not found"
fi

# Ollama
if docker ps --filter "name=fyp-ollama" --format "{{.Names}}" | grep -q "fyp-ollama"; then
  if curl -s http://localhost:11434/api/tags &> /dev/null; then
    log_pass "Ollama API is responsive"
    
    # Check models
    if curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
      log_pass "Embedding model (nomic-embed-text) is available"
    else
      log_warn "Embedding model (nomic-embed-text) not found - will be pulled on demand"
    fi
  else
    log_fail "Ollama is not responding on port 11434"
  fi
else
  log_fail "Ollama container not found"
fi

################################################################################
# SECTION 6: Environment Variables
################################################################################

print_header "6. Checking Environment Variables"

# Check .env file
if [ -f ".env" ]; then
  log_pass ".env file exists"
  
  # Check critical env vars
  CRITICAL_VARS=(
    "DEEPSEEK_API_KEY"
    "EODHD_API_KEY"
    "POSTGRES_HOST"
    "NEO4J_URI"
  )
  
  for var in "${CRITICAL_VARS[@]}"; do
    if grep -q "^${var}=" .env; then
      VALUE=$(grep "^${var}=" .env | cut -d'=' -f2 | head -c 20)...
      log_pass "Environment variable found: $var=$VALUE"
    else
      log_warn "Environment variable missing or commented: $var"
    fi
  done
else
  log_fail ".env file not found"
fi

################################################################################
# SECTION 7: Orchestration Tests
################################################################################

print_header "7. Running Orchestration Tests"

if [ -n "$VIRTUAL_ENV" ]; then
  # Test imports
  if python -c "from orchestration.graph import run" 2>/dev/null; then
    log_pass "Orchestration module imports successfully"
  else
    log_fail "Orchestration module import failed"
    verbose_output "Check: ls orchestration/graph.py"
  fi
  
  # Test agents import
  if python -c "import agents" 2>/dev/null; then
    log_pass "Agents module imports successfully"
  else
    log_fail "Agents module import failed"
  fi
else
  log_warn "Cannot test orchestration (venv not activated)"
fi

################################################################################
# SECTION 8: Port Availability
################################################################################

print_header "8. Checking Port Availability"

PORTS=(
  "8501:Streamlit"
  "8080:Airflow"
  "7474:Neo4j HTTP"
  "7687:Neo4j Bolt"
  "5432:PostgreSQL"
  "11434:Ollama"
)

for port_info in "${PORTS[@]}"; do
  IFS=':' read -r port service <<< "$port_info"
  
  if netstat -tuln 2>/dev/null | grep -q ":$port " || lsof -i ":$port" 2>/dev/null | grep -q LISTEN; then
    log_pass "Port $port ($service) is accessible"
  else
    log_warn "Port $port ($service) may not be in use (service may be down)"
  fi
done

################################################################################
# SECTION 9: Test Suite Check
################################################################################

print_header "9. Checking Test Suite"

if [ -d "tests" ]; then
  TEST_COUNT=$(find tests -name "test_*.py" -o -name "*_test.py" | wc -l)
  if [ "$TEST_COUNT" -gt 0 ]; then
    log_pass "Found $TEST_COUNT test files"
    
    # Check if pytest is available
    if python -c "import pytest" 2>/dev/null; then
      log_pass "pytest is installed and available"
    else
      log_warn "pytest not found - install with: pip install pytest"
    fi
  else
    log_warn "No test files found in tests/"
  fi
else
  log_fail "tests/ directory not found"
fi

################################################################################
# SECTION 10: Documentation Check
################################################################################

print_header "10. Checking Documentation"

DOC_FILES=(
  "docs/README.md"
  "docs/DEPLOYMENT.md"
  "docs/FRESH_DEPLOYMENT.md"
  "TROUBLESHOOTING.md"
)

for doc in "${DOC_FILES[@]}"; do
  if [ -f "$doc" ]; then
    log_pass "Found documentation: $doc"
  else
    log_warn "Documentation not found: $doc"
  fi
done

################################################################################
# Summary & Recommendations
################################################################################

print_footer

# Overall status
if [ $CHECKS_FAILED -eq 0 ]; then
  if [ $CHECKS_WARNED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your deployment is ready.${NC}"
    EXIT_CODE=0
  else
    echo -e "${YELLOW}⚠ Deployment ready with warnings. Review above.${NC}"
    EXIT_CODE=1
  fi
else
  echo -e "${RED}✗ Critical issues detected. Cannot proceed.${NC}"
  EXIT_CODE=2
fi

echo ""
echo "Next steps:"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
  echo "1. Fix the failures above"
  echo "2. Re-run this script to validate"
  echo "3. Refer to TROUBLESHOOTING.md for help"
else
  echo "1. Start services: docker compose up -d"
  echo "2. Run orchestration test: python orchestration/test.py"
  echo "3. Launch UI: streamlit run ui/app.py"
  echo "4. Access dashboards:"
  echo "   - Streamlit: http://localhost:8501"
  echo "   - Airflow: http://localhost:8080"
  echo "   - Neo4j: http://localhost:7474"
fi

echo ""

exit $EXIT_CODE
