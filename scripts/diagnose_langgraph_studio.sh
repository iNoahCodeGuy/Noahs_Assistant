#!/bin/bash
# Diagnostic script for LangGraph Studio connection issues

echo "🔍 LangGraph Studio Connection Diagnostics"
echo "=========================================="
echo ""

# Check 1: Port availability
echo "1️⃣  Checking port 2024..."
if lsof -Pi :2024 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "   ✅ Port 2024 is in use"
    PID=$(lsof -ti:2024)
    echo "   📋 Process ID: $PID"
    ps -p $PID -o command= 2>/dev/null || echo "   ⚠️  Process not found"
else
    echo "   ❌ Port 2024 is not in use (server not running)"
fi
echo ""

# Check 2: Server connectivity
echo "2️⃣  Testing server connectivity..."
if curl -s -f http://127.0.0.1:2024/info > /dev/null 2>&1; then
    echo "   ✅ Server is responding at /info"
    curl -s http://127.0.0.1:2024/info | head -5
elif curl -s -f http://127.0.0.1:2024/ > /dev/null 2>&1; then
    echo "   ✅ Server is responding at /"
else
    echo "   ❌ Server is not responding"
    echo "   💡 Make sure 'langgraph dev' is running"
fi
echo ""

# Check 3: LangGraph CLI
echo "3️⃣  Checking LangGraph CLI..."
if command -v langgraph &> /dev/null; then
    echo "   ✅ LangGraph CLI is installed"
    langgraph --version 2>/dev/null || echo "   ⚠️  Could not get version"
else
    echo "   ❌ LangGraph CLI not found"
    echo "   💡 Install with: pip install langgraph-cli"
fi
echo ""

# Check 4: Configuration file
echo "4️⃣  Checking configuration..."
if [ -f langgraph.json ]; then
    echo "   ✅ langgraph.json exists"
    if python3 -c "import json; json.load(open('langgraph.json'))" 2>/dev/null; then
        echo "   ✅ langgraph.json is valid JSON"
    else
        echo "   ❌ langgraph.json is invalid JSON"
    fi
else
    echo "   ❌ langgraph.json not found"
fi
echo ""

# Check 5: Graph import
echo "5️⃣  Checking graph import..."
if [ -f langgraph.json ]; then
    GRAPH_PATH=$(python3 -c "import json; print(json.load(open('langgraph.json'))['graphs'].get('conversation_flow', ''))" 2>/dev/null)
    if [ -n "$GRAPH_PATH" ]; then
        echo "   📋 Graph path: $GRAPH_PATH"
        if python3 -c "import sys; sys.path.insert(0, '.'); exec('from ${GRAPH_PATH%:*} import ${GRAPH_PATH##*:}')" 2>/dev/null; then
            echo "   ✅ Graph can be imported"
        else
            echo "   ❌ Graph cannot be imported"
            echo "   💡 Check if the path is correct and dependencies are installed"
        fi
    fi
fi
echo ""

# Check 6: Environment variables
echo "6️⃣  Checking environment variables..."
if [ -f .env ]; then
    echo "   ✅ .env file exists"
    if grep -q "LANGCHAIN_API_KEY" .env 2>/dev/null; then
        echo "   ✅ LANGCHAIN_API_KEY is set"
    else
        echo "   ⚠️  LANGCHAIN_API_KEY not found in .env"
    fi
    if grep -q "LANGCHAIN_TRACING_V2" .env 2>/dev/null; then
        echo "   ✅ LANGCHAIN_TRACING_V2 is set"
    else
        echo "   ⚠️  LANGCHAIN_TRACING_V2 not found in .env (will default to false)"
    fi
else
    echo "   ⚠️  .env file not found"
fi
echo ""

# Check 7: Network connectivity
echo "7️⃣  Testing network connectivity..."
if curl -s -f https://smith.langchain.com > /dev/null 2>&1; then
    echo "   ✅ Can reach smith.langchain.com"
else
    echo "   ❌ Cannot reach smith.langchain.com"
    echo "   💡 Check your internet connection"
fi
echo ""

# Summary
echo "=========================================="
echo "📋 Summary:"
echo ""
if lsof -Pi :2024 -sTCP:LISTEN -t >/dev/null 2>&1 && curl -s -f http://127.0.0.1:2024/info > /dev/null 2>&1; then
    echo "✅ Server appears to be running correctly"
    echo ""
    echo "💡 If you're still getting 'Failed to fetch' errors:"
    echo "   1. Try refreshing the LangSmith Studio page"
    echo "   2. Check browser console for detailed errors (F12)"
    echo "   3. Try connecting directly: http://127.0.0.1:2024"
    echo "   4. Verify CORS is enabled (LangGraph dev should handle this)"
else
    echo "❌ Server is not running or not responding"
    echo ""
    echo "💡 To start the server:"
    echo "   ./start_langgraph_studio.sh"
fi
echo ""
