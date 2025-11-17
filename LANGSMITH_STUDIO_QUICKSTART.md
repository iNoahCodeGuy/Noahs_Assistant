# LangSmith Studio - Quick Start Guide

## 🎯 Simple Setup (One Command)

To start LangGraph Studio testing:

```bash
./start_studio.sh
```

That's it! The script will:
1. Check if Docker is running
2. Auto-start Docker Desktop if needed (waits 30 seconds for initialization)
3. Load your environment variables from `.env`
4. Start LangGraph server on port 2024

## 🔗 Connect LangSmith Studio

Once the server is running (you'll see "Ready. Listening on http://0.0.0.0:2024"), open LangSmith Studio and connect to:

```
http://127.0.0.1:2024
```

## 🛠️ Manual Steps (If Needed)

If the script fails or you want more control:

### 1. Ensure Docker is Running
```bash
open /Applications/Docker.app
# Wait ~30 seconds for Docker daemon to start
docker info  # Verify it's running
```

### 2. Start LangGraph Server
```bash
export $(cat .env | grep -v '^#' | xargs) && langgraph up --port 2024
```

### 3. Connect Studio
- Open LangSmith Studio desktop app
- Connect to `http://127.0.0.1:2024`
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
- Check if server is actually running: `docker ps` (should show langgraph containers)
- Check server logs: `docker logs <container_id>`
- Verify port 2024 is not in use: `lsof -i :2024`

### "Docker not installed" error
- Ensure Docker Desktop is in `/Applications/Docker.app`
- Run `open /Applications/Docker.app` manually
- Wait 30 seconds, then verify with `docker info`

### Server won't start / Module errors
- Verify `langgraph.json` has correct path: `./assistant/flows/conversation_flow.py:graph`
- Check `.env` file exists with required variables:
  - `LANGSMITH_API_KEY`
  - `OPENAI_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`

### Images take forever to build
- First run downloads ~140MB of Docker images (postgres, redis, langgraph)
- Subsequent runs are much faster (images are cached)
- If interrupted: Just run `./start_studio.sh` again to resume

## 📝 Notes

- **Port**: LangGraph Studio uses port 2024 (Streamlit uses 8501/8502)
- **Environment**: Server loads variables from `.env` automatically
- **Logs**: All LangSmith traces go to project "noahs-ai-assistant"
- **Cleanup**: Stop server with Ctrl+C, containers auto-removed

## 🚀 Next Steps

Once connected, you can:
- Visualize conversation flow graph
- Step through node execution
- Inspect state transformations
- Debug retrieval/generation steps
- Monitor LangSmith traces in real-time

See `docs/LANGSMITH_TRACING_SETUP.md` for advanced observability features.
