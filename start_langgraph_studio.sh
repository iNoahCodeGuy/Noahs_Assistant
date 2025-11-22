#!/bin/bash
# Quick LangGraph Studio Startup (No Docker - for testing)

echo "🚀 Starting LangGraph Studio (No Docker)..."
echo "==============================================="

# Load environment variables
if [ -f .env ]; then
    echo "📊 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment loaded"
else
    echo "⚠️  No .env file found - continuing anyway..."
fi

# Ensure LangSmith tracing is enabled
export LANGCHAIN_TRACING_V2=true
echo "🔗 LangSmith tracing enabled"

# Start LangGraph dev server (no Docker)
echo "🎯 Starting LangGraph dev server..."
echo "🌐 Server will be at: http://127.0.0.1:2024"
echo "📊 LangSmith Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
echo "📈 View traces: https://smith.langchain.com/o/project/${LANGCHAIN_PROJECT:-noahs-ai-assistant}"
echo "==============================================="
echo ""
echo "⏳ Starting server... (will open browser in 5 seconds)"
echo ""

# Start langgraph dev in background and capture PID
langgraph dev &
LANGGRAPH_PID=$!

# Wait a few seconds for server to start, then open browser
sleep 5

# Open LangSmith Studio in browser
LANGSMITH_URL="https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
echo "🌐 Opening LangSmith Studio in browser..."
open "$LANGSMITH_URL" 2>/dev/null || xdg-open "$LANGSMITH_URL" 2>/dev/null || echo "⚠️  Please manually open: $LANGSMITH_URL"

echo ""
echo "✅ Server running (PID: $LANGGRAPH_PID)"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Wait for langgraph process
wait $LANGGRAPH_PID
