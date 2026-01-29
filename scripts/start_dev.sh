#!/bin/bash
# Quick dev server start (no tunnel, optimized for iteration)
# This keeps the URL stable so you don't need to reconnect in LangSmith Studio

cd "$(dirname "$0")/.."

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Ensure LangSmith tracing is enabled
export LANGCHAIN_TRACING_V2=true

echo "🚀 Starting LangGraph dev server (localhost mode)"
echo "================================================"
echo ""
echo "🌐 Server URL: http://127.0.0.1:2024"
echo "📊 LangSmith Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
echo ""
echo "💡 This URL stays stable - no need to reconnect after restarts!"
echo "💡 Server auto-reloads on code changes - just save and test!"
echo ""
echo "📝 First time setup:"
echo "   1. Connect to http://127.0.0.1:2024 in LangSmith Studio"
echo "   2. Add 'https://smith.langchain.com' to Allowed Origins"
echo "   3. Keep connected while iterating!"
echo ""
echo "================================================"
echo ""

# Start server (no tunnel for faster iteration)
langgraph dev
