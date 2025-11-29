# LangSmith Studio - Local Server Setup

Quick guide to connect LangSmith Studio to your local LangGraph server for testing.

## 🚀 Quick Start

### Option 1: Python Script (Recommended)

```bash
python3 start_langgraph_local.py
```

This will:
- Load your `.env` file automatically
- Check if port 2024 is available
- Start the LangGraph dev server
- Provide the connection URL for LangSmith Studio

### Option 2: Bash Script

```bash
./start_langgraph_studio.sh
```

### Option 3: Manual Start

```bash
# 1. Load environment variables
export $(cat .env | grep -v '^#' | xargs)
export LANGCHAIN_TRACING_V2=true

# 2. Start LangGraph dev server
langgraph dev
```

## 📋 Prerequisites

1. **Install langgraph-cli:**
   ```bash
   pip install langgraph-cli
   ```

2. **Ensure .env file exists** with required variables:
   - `LANGSMITH_API_KEY`
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `LANGCHAIN_PROJECT` (optional, defaults to "noahs-ai-assistant")

## 🔗 Connecting LangSmith Studio

Once the server is running (you'll see "Ready. Listening on http://0.0.0.0:2024"):

1. **Open LangSmith Studio** in your browser:
   - Go to: https://smith.langchain.com/studio/
   - Or use the direct link provided by the script

2. **Connect to your local server:**
   - Server URL: `http://127.0.0.1:2024`
   - Or use the full URL: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

3. **Select your graph:**
   - You should see `conversation_flow` graph available
   - This corresponds to `assistant/flows/conversation_flow.py:graph`

## 🧪 Testing Your Graph

Once connected, you can:
- **Visualize the graph** - See all nodes and edges
- **Step through execution** - Run queries and watch state flow
- **Inspect state** - View state at each node
- **Debug issues** - See exactly where problems occur
- **View traces** - All runs are automatically traced to LangSmith

## 🔍 Troubleshooting

### Port 2024 Already in Use

```bash
# Find and kill the process
lsof -ti:2024 | xargs kill -9

# Or use a different port (modify langgraph.json)
```

### "langgraph: command not found"

```bash
pip install langgraph-cli
```

### "Connection failed" in LangSmith Studio

1. **Verify server is running:**
   ```bash
   curl http://127.0.0.1:2024/info
   # Should return JSON response
   ```

2. **Check server logs** - The script shows output in real-time

3. **Browser issues (Safari/Brave):**
   - Use the tunnel option: `USE_TUNNEL=true ./start_langgraph_studio.sh`
   - Or use Chrome/Firefox

### Graph Not Showing

1. **Verify langgraph.json:**
   ```json
   {
     "graphs": {
       "conversation_flow": "./assistant/flows/conversation_flow.py:graph"
     }
   }
   ```

2. **Check graph export:**
   - Ensure `assistant/flows/conversation_flow.py` exports `graph`
   - Test import: `python3 -c "from assistant.flows.conversation_flow import graph"`

## 📊 Viewing Traces

All runs are automatically traced to LangSmith:
- **Project Dashboard:** https://smith.langchain.com/o/project/{LANGCHAIN_PROJECT}
- **Default project:** `noahs-ai-assistant`

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## 📝 Configuration

The server uses `langgraph.json` for configuration:
- **Graph location:** `./assistant/flows/conversation_flow.py:graph`
- **Environment:** `.env` file
- **Python version:** 3.12

## 💡 Tips

- **Keep server running** while testing - it auto-reloads on code changes
- **Use LangSmith Studio** for debugging complex state flows
- **Check traces** in LangSmith dashboard for production-like debugging
- **Test edge cases** by stepping through the graph manually
