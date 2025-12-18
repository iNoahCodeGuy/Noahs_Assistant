# LangSmith Studio - Quick Start Guide

## 🎯 Simple Setup (One Command)

To start LangGraph Studio testing:

```bash
./start_langgraph_studio.sh
```

That's it! The script will:
1. Load your environment variables from `.env`
2. Start LangGraph dev server on port 2024
3. Wait for server to be ready
4. Open LangSmith Studio in your browser automatically

## 🔗 Connect LangSmith Studio

Once the server is running, LangSmith Studio will open automatically. If not, use this URL:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Or manually connect to:
```
http://127.0.0.1:2024
```

## 🛠️ Manual Steps (If Needed)

If the script fails or you want more control:

### 1. Load Environment Variables
```bash
export $(cat .env | grep -v '^#' | xargs)
export LANGCHAIN_TRACING_V2=true
```

### 2. Start LangGraph Server
```bash
langgraph dev
```

The server will start on port 2024 by default.

### 3. Connect Studio
- Open [LangSmith Studio](https://smith.langchain.com/studio/)
- Connect to `http://127.0.0.1:2024`
- Or use the direct link: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
- You should see your `conversation_flow` graph available

## 📁 Configuration

**Directory Structure**: Main application code is in `assistant/`
- LangGraph flow is at `assistant/flows/conversation_flow.py`
- All imports use `assistant.*` namespace

**Configuration**: `langgraph.json`
```json
{
  "dependencies": ["."],
  "graphs": {"conversation_flow": "./assistant/flows/conversation_flow.py:graph"},
  "env": ".env",
  "python_version": "3.11"
}
```

## 🔍 Troubleshooting

### "Connection failed" in LangSmith Studio
- Check if server is actually running: `curl http://127.0.0.1:2024/info`
- Check server logs: `cat /tmp/langgraph_dev.log`
- Verify port 2024 is not in use: `lsof -i :2024`
- **Safari/Brave users**: Use `USE_TUNNEL=true ./start_langgraph_studio.sh`

### "langgraph command not found" error
- Install langgraph-cli: `pip install langgraph-cli`
- Verify installation: `langgraph --version`

### Server won't start / Module errors
- Verify `langgraph.json` has correct path: `./assistant/flows/conversation_flow.py:graph`
- Check `.env` file exists with required variables:
  - `LANGSMITH_API_KEY`
  - `OPENAI_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`

### Server takes time to start
- First run may take 10-30 seconds to initialize
- The script waits up to 45 seconds for the server to be ready
- Check logs if it takes longer: `cat /tmp/langgraph_dev.log`

## 📝 Notes

- **Port**: LangGraph dev server uses port 2024
- **Environment**: Server loads variables from `.env` automatically
- **Logs**: All LangSmith traces go to project "noahs-ai-assistant"
- **Cleanup**: Stop server with Ctrl+C or `lsof -ti:2024 | xargs kill -9`
- **Tunnel**: Use `USE_TUNNEL=true` for Safari/Brave browser compatibility

## 🚀 Next Steps

Once connected, you can:
- Visualize conversation flow graph
- Step through node execution
- Inspect state transformations
- Debug retrieval/generation steps
- Monitor LangSmith traces in real-time

See `docs/LANGSMITH_TRACING_SETUP.md` for advanced observability features.
