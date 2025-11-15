#!/bin/bash
# Safe restart script for LangGraph Studio development
# Ensures clean rebuild with code changes

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker stop $(docker ps -q --filter "name=langgraph") 2>/dev/null || true

echo -e "${YELLOW}⏳ Waiting for cleanup...${NC}"
sleep 2

echo -e "${YELLOW}🔨 Rebuilding with latest code...${NC}"
# Run in background to capture PID
langgraph up --port 2024 > /tmp/langgraph_startup.log 2>&1 &
LANGGRAPH_PID=$!

echo -e "${YELLOW}⏳ Waiting for server to initialize (15s)...${NC}"
sleep 15

echo -e "${YELLOW}🔍 Checking server health...${NC}"
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:2024/info > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is ready!${NC}"
        echo -e "${GREEN}🌐 Open http://127.0.0.1:2024 to test${NC}"
        echo -e "${GREEN}📊 LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024${NC}"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -e "${YELLOW}   Attempt $RETRY_COUNT/$MAX_RETRIES... (waiting 3s)${NC}"
        sleep 3
    fi
done

echo -e "${RED}❌ Server failed to start after $MAX_RETRIES attempts${NC}"
echo -e "${RED}📋 Last 30 lines of startup log:${NC}"
tail -30 /tmp/langgraph_startup.log

echo -e "${RED}📋 Container logs:${NC}"
docker logs --tail 50 noahsaiassistant--langgraph-api-1 2>/dev/null || echo "Container not found"

exit 1
