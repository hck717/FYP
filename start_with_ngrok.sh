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

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting FYP with ngrok tunnel ===${NC}"

# Check if ngrok token is provided
NGROK_TOKEN=${1:-${NGROK_AUTHTOKEN:-}}

if [ -z "$NGROK_TOKEN" ]; then
    echo -e "${YELLOW}Warning: NGROK_AUTHTOKEN not set${NC}"
    echo "Get your free token at: https://dashboard.ngrok.com/signup"
    echo "Usage: ./start_with_ngrok.sh YOUR_NGROK_TOKEN"
    echo ""
    read -p "Enter your ngrok token (or press Enter to skip): " NGROK_TOKEN
fi

# Start PostgreSQL and Neo4j
echo -e "${GREEN}Starting PostgreSQL and Neo4j...${NC}"
cd "$(dirname "$0")"

# Check if docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    docker-compose up -d postgres neo4j
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

# Kill any existing ngrok processes on port 8501
pkill -f "ngrok.*8501" 2>/dev/null || true

# Start ngrok tunnel in background
echo -e "${GREEN}Starting ngrok tunnel on port 8501...${NC}"
ngrok http 8501 --log=stdout > /tmp/ngrok.log 2>&1 &
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
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true

cd "$(dirname "$0")/POC/streamlit"
streamlit run app.py --server.port 8501 --server.headless true &

# Wait for streamlit to start
sleep 3

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  📱 Access URLs:${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo -e "  Local:   ${GREEN}http://localhost:8501${NC}"
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
trap "echo 'Stopping services...'; kill $NGROK_PID 2>/dev/null; pkill -f 'streamlit run'; docker-compose down 2>/dev/null; exit" INT TERM

# Wait for ngrok
wait $NGROK_PID
