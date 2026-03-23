#!/bin/bash
#
# start_with_ngrok.sh - Start Streamlit with ngrok tunnel
#
# This script starts:
# 1. PostgreSQL (via docker-compose)
# 2. Neo4j (via docker-compose)  
# 3. ngrok tunnel on port 8501 (Streamlit default)
# 4. Streamlit app
#
# Usage:
#   ./start_with_ngrok.sh [ngrok_token]
#
# Environment:
#   NGROK_AUTHTOKEN - Your ngrok auth token (can pass as argument)
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting FYP with ngrok tunnel ===${NC}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD="python3"
STREAMLIT_CMD="streamlit"
PIP_CMD="python3 -m pip"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ] && [ -x "$SCRIPT_DIR/.venv/bin/streamlit" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
    STREAMLIT_CMD="$SCRIPT_DIR/.venv/bin/streamlit"
    PIP_CMD="$SCRIPT_DIR/.venv/bin/python -m pip"
fi

# Check if ngrok token is provided
NGROK_TOKEN=${1:-${NGROK_AUTHTOKEN:-}}

if [ -z "$NGROK_TOKEN" ]; then
    echo -e "${YELLOW}Warning: NGROK_AUTHTOKEN not set${NC}"
    echo "Get your free token at: https://dashboard.ngrok.com/signup"
    echo "Usage: ./start_with_ngrok.sh YOUR_NGROK_TOKEN"
    echo ""
    read -p "Enter your ngrok token (or press Enter to skip): " NGROK_TOKEN
fi

# Pick a free local port (prefers 8501)
pick_port() {
    local candidate
    for candidate in 8501 8502 8503 8504 8505; do
        if ! lsof -nP -iTCP:"$candidate" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

STREAMLIT_PORT="$(pick_port || true)"
if [ -z "$STREAMLIT_PORT" ]; then
    echo -e "${RED}Error: no free port found in 8501-8505${NC}"
    exit 1
fi

if [ "$STREAMLIT_PORT" != "8501" ]; then
    echo -e "${YELLOW}Port 8501 is busy, using port ${STREAMLIT_PORT} instead${NC}"
fi

# Ensure local dependencies exist (installs once if missing)
echo -e "${GREEN}Checking Python dependencies...${NC}"
if ! "$PYTHON_CMD" -c "import streamlit, dotenv, langchain_openai, neo4j, psycopg2" >/dev/null 2>&1; then
    echo -e "${YELLOW}Missing dependencies detected. Installing requirements...${NC}"
    $PIP_CMD install -r "$SCRIPT_DIR/requirements.txt"
fi

# Start PostgreSQL and Neo4j
echo -e "${GREEN}Starting PostgreSQL and Neo4j...${NC}"

# Check if docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    docker compose up -d postgres neo4j
    echo -e "${GREEN}✓ PostgreSQL and Neo4j started${NC}"
else
    echo -e "${YELLOW}Warning: docker-compose.yml not found, skipping database start${NC}"
fi

# Wait for databases to be ready
echo -e "${GREEN}Waiting for databases to be ready...${NC}"
sleep 5

# Configure ngrok
if [ -n "$NGROK_TOKEN" ]; then
    echo -e "${GREEN}Configuring ngrok...${NC}"
    ngrok config add-authtoken "$NGROK_TOKEN" 2>/dev/null || true
fi

# Kill any existing ngrok process to avoid 4040 conflicts
pkill -f "ngrok http" 2>/dev/null || true

# Start ngrok tunnel in background
echo -e "${GREEN}Starting ngrok tunnel on port ${STREAMLIT_PORT}...${NC}"
ngrok http "$STREAMLIT_PORT" --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start
sleep 3

# Get the public URL
if [ -f "/tmp/ngrok.log" ]; then
    echo -e "${GREEN}ngrok log:${NC}"
    cat /tmp/ngrok.log
fi

# Try to extract the URL
NGROK_URL=$(curl -s localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('tunnels',[{}])[0].get('public_url','NOT_READY'))" 2>/dev/null || echo "NOT_READY")

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🚀 Your FYP App is starting!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Start streamlit
echo -e "${GREEN}Starting Streamlit...${NC}"
export STREAMLIT_SERVER_PORT="$STREAMLIT_PORT"
export STREAMLIT_SERVER_HEADLESS=true

cd "$(dirname "$0")/POC/streamlit"
"$STREAMLIT_CMD" run app.py --server.port "$STREAMLIT_PORT" --server.headless true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait for streamlit to start and fail fast on startup errors
for i in {1..20}; do
    if lsof -nP -iTCP:"$STREAMLIT_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo -e "${RED}Streamlit failed to start. Last logs:${NC}"
        tail -n 120 /tmp/streamlit.log || true
        kill "$NGROK_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  📱 Access URLs:${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo -e "  Local:   ${GREEN}http://localhost:${STREAMLIT_PORT}${NC}"
echo ""

# Try to get ngrok URL
for i in {1..10}; do
    NGROK_URL=$(curl -s localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('tunnels',[{}])[0].get('public_url','NOT_READY'))" 2>/dev/null || echo "NOT_READY")
    if [ "$NGROK_URL" != "NOT_READY" ] && [ -n "$NGROK_URL" ]; then
        echo -e "  Public:  ${GREEN}$NGROK_URL${NC}"
        echo ""
        echo -e "${YELLOW}Share this URL with others!${NC}"
        break
    fi
    sleep 2
done

if [ "$NGROK_URL" == "NOT_READY" ] || [ -z "$NGROK_URL" ]; then
    echo -e "${YELLOW}  Public:  Check ngrok logs at http://localhost:4040${NC}"
fi

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  🔧 Management URLs:${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo -e "  ngrok dashboard: ${GREEN}http://localhost:4040${NC}"
echo -e "  PostgreSQL:      ${GREEN}localhost:5432${NC}"
echo -e "  Neo4j:           ${GREEN}localhost:7474 (browser)${NC}"
echo -e "  Neo4j Bolt:      ${GREEN}localhost:7687${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
trap "echo 'Stopping services...'; kill $NGROK_PID 2>/dev/null; kill $STREAMLIT_PID 2>/dev/null; docker compose down 2>/dev/null; exit" INT TERM

# Wait for streamlit process (and keep ngrok alive)
wait "$STREAMLIT_PID"
