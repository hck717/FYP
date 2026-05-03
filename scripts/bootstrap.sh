#!/bin/bash

################################################################################
# FYP Bootstrap & Initial Setup Script
#
# Automates the complete setup of FYP for new users:
# 1. Validates prerequisites
# 2. Sets up Python virtual environment
# 3. Configures environment variables
# 4. Starts Docker services
# 5. Validates deployment
#
# Usage:
#   chmod +x scripts/bootstrap.sh
#   ./scripts/bootstrap.sh
#
# With options:
#   ./scripts/bootstrap.sh --skip-venv     # Skip venv setup
#   ./scripts/bootstrap.sh --skip-docker   # Skip Docker startup
#   ./scripts/bootstrap.sh --help          # Show this message
#
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
SKIP_VENV=false
SKIP_DOCKER=false
SKIP_VALIDATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-venv)
      SKIP_VENV=true
      shift
      ;;
    --skip-docker)
      SKIP_DOCKER=true
      shift
      ;;
    --skip-validation)
      SKIP_VALIDATION=true
      shift
      ;;
    --help)
      cat << EOF
FYP Bootstrap Script

Usage: ./scripts/bootstrap.sh [OPTIONS]

Options:
  --skip-venv          Skip Python virtual environment setup
  --skip-docker        Skip Docker service startup
  --skip-validation    Skip final validation checks
  --help              Show this help message

Example (fresh setup):
  ./scripts/bootstrap.sh

Example (skip Docker, only setup Python):
  ./scripts/bootstrap.sh --skip-docker

EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

################################################################################
# Helper Functions
################################################################################

log_header() {
  echo ""
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║ $1${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

log_info() {
  echo -e "${BLUE}[→]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
  echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[!]${NC} $1"
}

confirm() {
  local prompt="$1"
  local response
  
  read -p "$(echo -e ${BLUE}${prompt}${NC})" response
  [[ "$response" =~ ^[Yy]$ ]]
}

################################################################################
# STEP 1: Validate Prerequisites
################################################################################

log_header "STEP 1: Validating Prerequisites"

# Check we're in FYP directory
if [ ! -f "docker-compose.yml" ] || [ ! -f "requirements.txt" ]; then
  log_error "Not in FYP root directory!"
  log_info "Current directory: $(pwd)"
  log_info "Run this script from the FYP project root"
  exit 1
fi

log_success "Found FYP project structure"

# Check Python
if ! command -v python3 &> /dev/null; then
  log_error "Python 3 not found!"
  log_info "Install Python 3.11+ from https://www.python.org/downloads/"
  exit 1
fi

PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_success "Python $PY_VERSION found"

# Check Docker (optional if skipping)
if [ "$SKIP_DOCKER" = false ]; then
  if ! command -v docker &> /dev/null; then
    log_error "Docker not found!"
    log_info "Install Docker from https://docs.docker.com/get-docker/"
    exit 1
  fi
  
  if ! docker ps &> /dev/null; then
    log_error "Docker daemon not running!"
    log_info "Start Docker Desktop or Docker daemon"
    exit 1
  fi
  
  DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
  log_success "Docker $DOCKER_VERSION is running"
fi

# Check Git
if ! command -v git &> /dev/null; then
  log_warning "Git not found (optional, for version control)"
fi

log_success "All prerequisites validated"

################################################################################
# STEP 2: Configure Environment Variables
################################################################################

log_header "STEP 2: Configuring Environment Variables"

if [ -f ".env" ]; then
  log_warning ".env already exists"
  if confirm "Do you want to keep existing .env? (Y/n) "; then
    log_info "Keeping existing .env configuration"
  else
    log_info "Backing up existing .env to .env.backup"
    cp .env .env.backup
    
    if [ -f ".env.example" ]; then
      log_info "Creating new .env from .env.example"
      cp .env.example .env
    fi
  fi
else
  if [ -f ".env.example" ]; then
    log_info "Creating .env from .env.example template"
    cp .env.example .env
  else
    log_warning ".env.example not found, will use default .env"
  fi
fi

log_success ".env configuration ready"

# Optionally prompt for API keys
if confirm "Do you want to configure API keys now? (Y/n) "; then
  echo ""
  echo "Enter your API keys (press Enter to skip):"
  echo ""
  
  read -p "EODHD_API_KEY (leave blank to skip): " EODHD_KEY
  if [ ! -z "$EODHD_KEY" ]; then
    sed -i '' "s/EODHD_API_KEY=.*/EODHD_API_KEY=$EODHD_KEY/" .env
    log_success "Updated EODHD_API_KEY"
  fi
  
  read -p "DEEPSEEK_API_KEY (leave blank to skip): " DEEPSEEK_KEY
  if [ ! -z "$DEEPSEEK_KEY" ]; then
    sed -i '' "s/DEEPSEEK_API_KEY=.*/DEEPSEEK_API_KEY=$DEEPSEEK_KEY/" .env
    log_success "Updated DEEPSEEK_API_KEY"
  fi
  
  read -p "FMP_API_KEY (leave blank to skip): " FMP_KEY
  if [ ! -z "$FMP_KEY" ]; then
    sed -i '' "s/FMP_API_KEY=.*/FMP_API_KEY=$FMP_KEY/" .env
    log_success "Updated FMP_API_KEY"
  fi
fi

################################################################################
# STEP 3: Setup Python Virtual Environment
################################################################################

if [ "$SKIP_VENV" = false ]; then
  log_header "STEP 3: Setting Up Python Virtual Environment"
  
  if [ -d ".venv" ]; then
    log_warning "Virtual environment already exists (.venv/)"
    if confirm "Recreate venv? (y/N) "; then
      log_info "Removing existing venv..."
      rm -rf .venv
      log_info "Creating new venv..."
      python3 -m venv .venv
      log_success "Virtual environment created"
    else
      log_info "Using existing venv"
    fi
  else
    log_info "Creating Python virtual environment..."
    python3 -m venv .venv
    log_success "Virtual environment created"
  fi
  
  # Activate venv
  log_info "Activating virtual environment..."
  source .venv/bin/activate
  log_success "Virtual environment activated"
  
  # Upgrade pip
  log_info "Upgrading pip..."
  pip install --quiet --upgrade pip setuptools wheel
  log_success "pip upgraded"
  
  # Install requirements
  log_info "Installing Python dependencies (this may take 2-5 minutes)..."
  pip install -q -r requirements.txt
  log_success "Dependencies installed"
  
  # Quick import test
  log_info "Testing imports..."
  python -c "import langchain, langchain_community, langgraph" && \
    log_success "Core dependencies working" || \
    log_error "Some dependencies failed to import"
  
else
  log_header "STEP 3: Skipping Virtual Environment Setup"
  log_warning "Make sure to activate venv: source .venv/bin/activate"
fi

################################################################################
# STEP 4: Start Docker Services
################################################################################

if [ "$SKIP_DOCKER" = false ]; then
  log_header "STEP 4: Starting Docker Services"
  
  log_info "This will build Docker images (first time takes 5-10 minutes)..."
  log_info "Starting PostgreSQL, Neo4j, Ollama, Airflow, and Streamlit..."
  echo ""
  
  docker compose down --remove-orphans 2>/dev/null || true
  
  docker compose up -d --build
  
  log_success "Docker services started"
  
  # Wait for services to be healthy
  log_info "Waiting for services to become healthy (this takes ~2-3 minutes)..."
  
  WAIT_TIME=0
  MAX_WAIT=180
  
  while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    echo -ne "  Checking service health... ${WAIT_TIME}s\r"
    
    POSTGRES_OK=$(docker exec fyp-postgres pg_isready -U airflow &>/dev/null && echo "true" || echo "false")
    NEO4J_OK=$(curl -s http://localhost:7474/db/data/ &>/dev/null && echo "true" || echo "false")
    OLLAMA_OK=$(curl -s http://localhost:11434/api/tags &>/dev/null && echo "true" || echo "false")
    
    if [ "$POSTGRES_OK" = "true" ] && [ "$NEO4J_OK" = "true" ] && [ "$OLLAMA_OK" = "true" ]; then
      echo -ne "  ✓ All services healthy!                \n"
      break
    fi
    
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
  done
  
  if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    log_warning "Services took longer than expected to start"
    log_info "Check status with: docker compose ps"
    log_info "View logs with: docker compose logs -f"
  fi
  
  log_success "Docker services ready"
  
else
  log_header "STEP 4: Skipping Docker Startup"
  log_warning "Make sure to start Docker services: docker compose up -d --build"
fi

################################################################################
# STEP 5: Quick Validation
################################################################################

if [ "$SKIP_VALIDATION" = false ]; then
  log_header "STEP 5: Running Validation Checks"
  
  if [ -f "scripts/validate-deployment.sh" ]; then
    chmod +x scripts/validate-deployment.sh
    ./scripts/validate-deployment.sh
  else
    log_warning "Validation script not found at scripts/validate-deployment.sh"
  fi
fi

################################################################################
# Complete
################################################################################

log_header "BOOTSTRAP COMPLETE ✓"

echo ""
echo "Next steps:"
echo ""
echo "1. Verify services are running:"
echo -e "   ${YELLOW}docker compose ps${NC}"
echo ""
echo "2. Access the dashboards:"
echo -e "   ${YELLOW}Streamlit UI:${NC}     http://localhost:8501"
echo -e "   ${YELLOW}Airflow:${NC}         http://localhost:8080 (admin/admin)"
echo -e "   ${YELLOW}Neo4j:${NC}           http://localhost:7474 (neo4j/SecureNeo4jPass2025!)"
echo ""
echo "3. Test the orchestration:"
echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
echo -e "   ${YELLOW}python - <<'EOF'${NC}"
echo "   from orchestration.graph import run"
echo "   result = run('What is Apple\\'s stock price?')"
echo "   print(result['final_summary'])"
echo -e "   ${YELLOW}EOF${NC}"
echo ""
echo "4. Or launch the Streamlit UI:"
echo -e "   ${YELLOW}streamlit run ui/app.py${NC}"
echo ""
echo "For documentation and troubleshooting:"
echo -e "   ${YELLOW}See docs/FRESH_DEPLOYMENT.md${NC}"
echo -e "   ${YELLOW}See TROUBLESHOOTING.md${NC}"
echo ""

exit 0
