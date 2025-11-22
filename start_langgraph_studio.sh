#!/bin/bash
# Quick LangGraph Studio Startup (No Docker - for testing)

set -e  # Exit on error

echo "🚀 Starting LangGraph Studio (No Docker)..."
echo "==============================================="

# Check if port 2024 is already in use
if lsof -Pi :2024 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 2024 is already in use!"
    echo "   Either stop the existing server or use a different port"
    echo ""
    echo "   To stop existing server:"
    echo "   lsof -ti:2024 | xargs kill -9"
    exit 1
fi

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
echo "🌐 Server will be at: http://127.0.0.1:2024 (or secure tunnel URL)"
echo "📊 LangSmith Studio: https://smith.langchain.com/studio/?baseUrl=<server-url>"
echo "📈 View traces: https://smith.langchain.com/o/project/${LANGCHAIN_PROJECT:-noahs-ai-assistant}"
echo ""
echo "💡 Using localhost by default (faster for testing)"
echo "   To enable tunnel (for Safari/Brave), set: USE_TUNNEL=true ./start_langgraph_studio.sh"
echo "==============================================="
echo ""

# Check if user wants to use tunnel (fixes Safari/Brave browser issues)
USE_TUNNEL=${USE_TUNNEL:-false}
if [ "$USE_TUNNEL" = "true" ]; then
    echo "🔒 Using secure tunnel (fixes browser connection issues)"
    TUNNEL_FLAG="--tunnel"
    TUNNEL_NOTE=" (via secure tunnel)"
else
    TUNNEL_FLAG=""
    TUNNEL_NOTE=""
fi

# Start langgraph dev in background and capture PID
echo "⏳ Starting server..."
if [ -n "$TUNNEL_FLAG" ]; then
    # With tunnel, we need to capture the tunnel URL from output
    langgraph dev $TUNNEL_FLAG > /tmp/langgraph_dev.log 2>&1 &
    LANGGRAPH_PID=$!
else
    langgraph dev > /tmp/langgraph_dev.log 2>&1 &
    LANGGRAPH_PID=$!
fi

# Wait for server to be ready
echo "⏳ Waiting for server to start..."
MAX_WAIT=45
WAIT_COUNT=0
SERVER_READY=false
TUNNEL_URL=""

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Check if process is still running
    if ! kill -0 $LANGGRAPH_PID 2>/dev/null; then
        echo ""
        echo "❌ Server failed to start!"
        echo "📋 Last 20 lines of log:"
        tail -20 /tmp/langgraph_dev.log
        exit 1
    fi

    # Check for tunnel URL in logs (if using tunnel)
    if [ -n "$TUNNEL_FLAG" ] && [ -z "$TUNNEL_URL" ]; then
        # Try multiple patterns for tunnel URL (macOS compatible)
        TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com[^[:space:]]*' /tmp/langgraph_dev.log 2>/dev/null | head -1 || echo "")
        if [ -z "$TUNNEL_URL" ]; then
            # Alternative pattern matching
            TUNNEL_URL=$(grep -i 'tunnel\|cloudflare' /tmp/langgraph_dev.log 2>/dev/null | grep -oE 'https://[^[:space:]]+' | head -1 || echo "")
        fi
        if [ -n "$TUNNEL_URL" ]; then
            echo ""
            echo "🔗 Tunnel URL detected: $TUNNEL_URL"
        fi
    fi

    # Check if server is responding
    if [ -n "$TUNNEL_URL" ]; then
        # Test tunnel URL
        if curl -s "${TUNNEL_URL}/info" > /dev/null 2>&1 || curl -s "${TUNNEL_URL}/" > /dev/null 2>&1; then
            SERVER_READY=true
            break
        fi
    else
        # Test local URL
        if curl -s http://127.0.0.1:2024/info > /dev/null 2>&1 || curl -s http://127.0.0.1:2024/ > /dev/null 2>&1; then
            SERVER_READY=true
            break
        fi
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
    sleep 1
done

echo ""

if [ "$SERVER_READY" = true ]; then
    echo "✅ Server is ready!"
    echo ""

    # Determine the base URL to use
    if [ -n "$TUNNEL_URL" ]; then
        BASE_URL="$TUNNEL_URL"
        echo "🔒 Using secure tunnel: $TUNNEL_URL"
    else
        BASE_URL="http://127.0.0.1:2024"
        echo "🌐 Using local server: $BASE_URL"

        # If using tunnel but URL not detected, show logs
        if [ -n "$TUNNEL_FLAG" ]; then
            echo ""
            echo "⚠️  Tunnel URL not auto-detected. Checking logs..."
            echo "📋 Recent log output (look for tunnel URL):"
            tail -10 /tmp/langgraph_dev.log | grep -i "tunnel\|cloudflare\|https://" || tail -5 /tmp/langgraph_dev.log
            echo ""
            echo "💡 If you see a tunnel URL above, use it in LangSmith Studio"
            echo "   Format: https://smith.langchain.com/studio/?baseUrl=<tunnel-url>"
        else
            echo "✅ Using localhost for faster testing"
            echo "💡 If you have browser connection issues (Safari/Brave), set USE_TUNNEL=true"
        fi
    fi

    # Open LangSmith Studio in browser
    LANGSMITH_URL="https://smith.langchain.com/studio/?baseUrl=${BASE_URL}"
    echo "🌐 Opening LangSmith Studio in browser..."
    open "$LANGSMITH_URL" 2>/dev/null || xdg-open "$LANGSMITH_URL" 2>/dev/null || echo "⚠️  Please manually open: $LANGSMITH_URL"

    echo ""
    echo "✅ Server running (PID: $LANGGRAPH_PID)"
    echo "🌐 Server URL: $BASE_URL"
    echo "📊 LangSmith Studio: $LANGSMITH_URL"
    if [ -n "$TUNNEL_FLAG" ] && [ -z "$TUNNEL_URL" ]; then
        echo ""
        echo "💡 To see the tunnel URL, check: tail -f /tmp/langgraph_dev.log"
    fi
    echo "🛑 Press Ctrl+C to stop"
    echo ""

    # Wait for langgraph process
    wait $LANGGRAPH_PID
else
    echo ""
    echo "❌ Server did not become ready after ${MAX_WAIT} seconds"
    echo "📋 Last 30 lines of log:"
    tail -30 /tmp/langgraph_dev.log
    echo ""
    echo "💡 Try checking:"
    echo "   - Is langgraph-cli installed? (pip install langgraph-cli)"
    echo "   - Are all dependencies installed?"
    echo "   - Check the full log: cat /tmp/langgraph_dev.log"
    echo "   - If using Safari/Brave, tunnel is enabled by default"
    kill $LANGGRAPH_PID 2>/dev/null || true
    exit 1
fi
