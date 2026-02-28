#!/bin/bash
# Migration Verification and Testing Script
# This script helps verify that database migrations are applied correctly

set -e  # Exit on error

echo "======================================"
echo "Supabase Migration Verification"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
    exit 1
fi

# Load environment variables
source .env

# Check required variables
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo -e "${RED}Error: Missing required environment variables${NC}"
    echo "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment variables loaded"
echo ""

# Function to run SQL query
run_query() {
    local query=$1
    curl -s -X POST "${SUPABASE_URL}/rest/v1/rpc/exec_sql" \
        -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"${query}\"}"
}

echo "Step 1: Checking which tables exist..."
echo "--------------------------------------"

# Check for each required table
tables=("kb_chunks" "messages" "retrieval_logs" "links" "feedback" "confessions" "sms_logs")

for table in "${tables[@]}"; do
    # Use Supabase REST API to check table
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        "${SUPABASE_URL}/rest/v1/${table}?limit=0" \
        -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓${NC} ${table} exists"
    else
        echo -e "${RED}✗${NC} ${table} MISSING (HTTP ${response})"
        missing_tables="${missing_tables}${table} "
    fi
done

echo ""

if [ -n "$missing_tables" ]; then
    echo -e "${RED}Missing tables detected!${NC}"
    echo "Please run the following migrations in Supabase SQL Editor:"
    echo ""
    
    if [[ "$missing_tables" == *"kb_chunks"* ]] || [[ "$missing_tables" == *"messages"* ]]; then
        echo "1. supabase/migrations/001_initial_schema.sql"
    fi
    
    if [[ "$missing_tables" == *"confessions"* ]] || [[ "$missing_tables" == *"sms_logs"* ]]; then
        echo "2. supabase/migrations/002_add_confessions_and_sms.sql"
    fi
    
    echo ""
    echo "How to run migrations:"
    echo "  1. Go to https://app.supabase.com"
    echo "  2. Select your project"
    echo "  3. Click SQL Editor → New Query"
    echo "  4. Copy/paste migration file contents"
    echo "  5. Click Run"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓${NC} All required tables exist!"
fi

echo ""
echo "Step 2: Testing API endpoints..."
echo "--------------------------------------"

# Test /api/chat
echo -n "Testing /api/chat... "
chat_response=$(curl -s -X POST https://noahsaiassistant.vercel.app/api/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "role": "Just looking around"}')

if echo "$chat_response" | grep -q '"success": true'; then
    echo -e "${GREEN}✓${NC} Working"
else
    echo -e "${RED}✗${NC} Failed"
    echo "Response: $chat_response"
fi

# Test /api/confess
echo -n "Testing /api/confess... "
confess_response=$(curl -s -X POST https://noahsaiassistant.vercel.app/api/confess \
    -H "Content-Type: application/json" \
    -d '{"message": "Test confession from verification script", "is_anonymous": true}')

if echo "$confess_response" | grep -q '"success": true'; then
    echo -e "${GREEN}✓${NC} Working"
elif echo "$confess_response" | grep -q "404"; then
    echo -e "${RED}✗${NC} 404 - Run migration 002"
else
    echo -e "${YELLOW}?${NC} Unexpected response"
    echo "Response: $confess_response"
fi

# Test /api/feedback
echo -n "Testing /api/feedback... "
feedback_response=$(curl -s -X POST https://noahsaiassistant.vercel.app/api/feedback \
    -H "Content-Type: application/json" \
    -d '{"message_id": "test", "rating": 5, "comment": "Test", "contact_requested": false}')

if echo "$feedback_response" | grep -q '"success": true'; then
    echo -e "${GREEN}✓${NC} Working"
elif echo "$feedback_response" | grep -q "column.*does not exist"; then
    echo -e "${RED}✗${NC} Missing columns - Run migration 002"
else
    echo -e "${YELLOW}?${NC} Unexpected response"
    echo "Response: $feedback_response"
fi

echo ""
echo "======================================"
echo "Verification Complete!"
echo "======================================"
