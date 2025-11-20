#!/bin/bash
# Start LangGraph server locally (NO DOCKER) for rapid development

echo "🚀 Starting LangGraph Local Server (NO DOCKER)..."
echo "==============================================="

# Load environment variables
if [ -f .env ]; then
    echo "📊 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment loaded"
else
    echo "⚠️  No .env file found (using system env)"
fi

# Check if langgraph CLI is available
if command -v langgraph &> /dev/null; then
    echo "✅ LangGraph CLI found"

    # Check if langgraph dev command is available (newer versions)
    if langgraph dev --help &> /dev/null; then
        echo "🎯 Using 'langgraph dev' (official local mode)"
        echo "💡 This runs locally with hot-reload - no Docker needed!"
        echo "💡 Code changes are picked up automatically!"
        echo ""
        echo "🔗 LangGraph Studio will be available at the URL shown below"
        echo "💡 Press Ctrl+C to stop"
        echo ""
        langgraph dev
    else
        echo "⚠️  'langgraph dev' not available (older CLI version)"
        echo "📦 To upgrade: pip install -U 'langgraph-cli[inmem]'"
        echo ""
        echo "🔄 Falling back to FastAPI server..."
        echo ""
        python3 scripts/run_langgraph_local_fastapi.py
    fi
else
    echo "⚠️  LangGraph CLI not found"
    echo "📦 Install with: pip install -U 'langgraph-cli[inmem]'"
    echo ""
    echo "🔄 Falling back to FastAPI server..."
    echo ""

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ python3 not found!"
        exit 1
    fi

    # Check dependencies
    echo "🔍 Checking dependencies..."
    python3 -c "import fastapi, uvicorn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  Missing dependencies. Installing..."
        pip install fastapi uvicorn[standard] python-dotenv
    fi

    echo ""
    echo "🎯 Starting FastAPI server..."
    echo "💡 Press Ctrl+C to stop"
    echo ""

    # Run the server
    python3 scripts/run_langgraph_local_fastapi.py
fi
