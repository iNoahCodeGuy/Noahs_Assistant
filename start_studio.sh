#!/bin/bash
# Quick LangGraph Studio Startup - Just run this!

echo "🚀 Starting LangGraph Studio..."

# Start Docker if needed
if ! docker info > /dev/null 2>&1; then
    echo "⏳ Starting Docker..."
    open /Applications/Docker.app
    sleep 10
fi

# Load environment and start
cd "$(dirname "$0")"
export $(cat .env | grep -v '^#' | xargs)
langgraph up --port 2024
