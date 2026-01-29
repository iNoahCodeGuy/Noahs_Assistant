# Ideal LangSmith Studio Workflow for Quick Iteration

## 🎯 Goal
Iterate quickly on graph behavior, traces, and turns without reconnecting each time you restart.

## ✅ Recommended Setup: Localhost + Allowed Origin

### Why This is Better Than Tunnel:
- ✅ **Stable URL**: `http://127.0.0.1:2024` never changes
- ✅ **One-time setup**: Add origin once, works forever
- ✅ **Auto-reload**: LangGraph dev server watches files and reloads automatically
- ✅ **No reconnection needed**: Keep LangSmith Studio connected while iterating
- ✅ **Faster**: No tunnel overhead

### Initial Setup (One Time)

1. **Start server without tunnel:**
   ```bash
   python3 connect_langsmith_studio.py
   # When asked "Use secure tunnel? (Y/n):", type: n
   ```

2. **In LangSmith Studio connection dialog:**
   - Base URL: `http://127.0.0.1:2024`
   - Expand "Advanced Settings"
   - Under "Allowed Origins", click "+ Allowed Origin"
   - Add: `https://smith.langchain.com`
   - Click "Connect"

3. **Keep LangSmith Studio open** - you won't need to reconnect!

## 🔄 Daily Workflow (Quick Iteration)

### Making Code Changes

1. **Edit your code** (nodes, flows, etc.)
   ```bash
   # Edit any file in assistant/flows/
   vim assistant/flows/node_logic/stage5_generation_nodes.py
   ```

2. **Save the file** - LangGraph dev server automatically detects changes and reloads!

3. **Test immediately** in LangSmith Studio - no restart needed!

### Restarting Server (Only When Needed)

If you need to restart (e.g., after changing `.env` or adding dependencies):

1. **Stop server**: `Ctrl+C` in terminal
2. **Restart**: `python3 connect_langsmith_studio.py` (choose `n` for tunnel)
3. **LangSmith Studio stays connected** - no need to reconnect!

The URL `http://127.0.0.1:2024` is stable, so your connection persists.

## 🚀 Quick Start Script

Create `scripts/start_dev.sh`:

```bash
#!/bin/bash
# Quick dev server start (no tunnel, for iteration)

cd "$(dirname "$0")/.."
export LANGCHAIN_TRACING_V2=true
langgraph dev
```

Usage:
```bash
chmod +x scripts/start_dev.sh
./scripts/start_dev.sh
```

## 📊 Monitoring Changes

LangGraph dev server shows file changes in logs:
```
2026-01-08T17:35:39.742771Z [info] 3 changes detected
2026-01-08T17:35:47.648607Z [info] 12 changes detected
```

This means it's watching and will reload automatically.

## ⚡ When to Use Tunnel

Only use tunnel (`--tunnel`) if:
- You're using Safari/Brave browser (they block HTTP localhost)
- You need to share the server with someone else
- You're testing from a different network

For local development and iteration, **localhost is faster and better**.

## 🔍 Verifying Auto-Reload Works

1. Start server: `langgraph dev` (no tunnel)
2. Make a small change to a node (add a print statement)
3. Save the file
4. Check terminal - you should see "X changes detected"
5. Test in LangSmith Studio - changes should be live!

## 🎨 Complete Workflow Example

```bash
# Terminal 1: Start server (keep running)
./scripts/start_dev.sh

# Terminal 2: Edit code
vim assistant/flows/node_logic/stage5_generation_nodes.py
# Make changes, save

# Browser: LangSmith Studio
# - Already connected to http://127.0.0.1:2024
# - Test your changes immediately
# - View traces in real-time
# - No reconnection needed!
```

## 🐛 Troubleshooting

### "Connection failed" after restart
- Make sure server is running: `curl http://127.0.0.1:2024/info`
- Check if origin is still in allowed list in LangSmith Studio

### Changes not appearing
- Check terminal for "X changes detected" messages
- If no messages, restart server (sometimes needed for major changes)
- Check for Python syntax errors in terminal

### Want to switch back to tunnel?
- Just restart with tunnel: `langgraph dev --tunnel`
- Update Base URL in LangSmith Studio to new tunnel URL
- But remember: tunnel URLs change each restart!

## 📝 Summary

**Best Practice for Iteration:**
1. ✅ Use localhost (`http://127.0.0.1:2024`)
2. ✅ Add origin once (`https://smith.langchain.com`)
3. ✅ Keep server running
4. ✅ Keep LangSmith Studio connected
5. ✅ Edit code → Save → Test (auto-reload handles the rest!)

This gives you the fastest iteration cycle possible! 🚀
