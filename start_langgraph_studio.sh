#!/bin/bash
# Easy LangGraph Studio Startup Script

echo "🚀 Starting LangGraph Studio..."
echo "==============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Starting Docker Desktop..."
    open /Applications/Docker.app
    echo "⏳ Waiting for Docker to start..."
    while ! docker info > /dev/null 2>&1; do
        sleep 2
        echo "   Still waiting for Docker..."
    done
    echo "✅ Docker is ready!"
fi

# Load environment variables
if [ -f .env ]; then
    echo "📊 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment loaded"
else
    echo "❌ .env file not found!"
    exit 1
fi

# Start LangGraph Studio
echo "🎯 Starting LangGraph Studio on port 2024..."
echo "🌐 Connect in LangSmith Studio: http://127.0.0.1:2024"
echo "📊 Dashboard: https://smith.langchain.com/"
echo "==============================================="

langgraph up --port 2024
