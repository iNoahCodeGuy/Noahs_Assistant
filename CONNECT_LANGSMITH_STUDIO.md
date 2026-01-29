# Connect LangSmith Studio to Local Server

## 🚀 Quick Start (Recommended)

Run the connection script:

```bash
python3 connect_langsmith_studio.py
```

This script will:
- ✅ Check prerequisites (langgraph-cli, environment variables)
- ✅ Start the LangGraph dev server on http://127.0.0.1:2024
- ✅ Provide the connection URL for LangSmith Studio
- ✅ Optionally open LangSmith Studio in your browser

## ✅ Server Status

Your local LangGraph server should be running on **http://127.0.0.1:2024**

## 🔗 Connect LangSmith Studio

### Option 1: Direct Browser Link (Recommended)

Open this URL in your browser:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

This will open LangSmith Studio and automatically connect to your local server.

### Option 2: Manual Connection

1. Go to [LangSmith Studio](https://smith.langchain.com/studio/)
2. In the connection dialog, enter:
   ```
   http://127.0.0.1:2024
   ```
3. Click "Connect"

## 📊 What You'll See

Once connected, you should see:
- Your `conversation_flow` graph available for testing
- Ability to visualize the graph structure
- Step-by-step execution debugging
- Real-time state inspection

## 🛑 Stop the Server

To stop the local server:
```bash
lsof -ti:2024 | xargs kill -9
```

Or if you started it with the script, press `Ctrl+C` in that terminal.

## 🔄 Restart the Server

To restart the server:
```bash
./start_langgraph_studio.sh
```

## 🌐 Using Secure Tunnel (for Safari/Brave)

If you're using Safari or Brave browser and encounter connection issues, use the secure tunnel:

```bash
USE_TUNNEL=true ./start_langgraph_studio.sh
```

This will create an HTTPS tunnel URL that works with all browsers.

## 📈 View Traces

All traces are automatically sent to LangSmith. View them at:

```
https://smith.langchain.com/o/project/noahs-ai-assistant
```

## 🔍 Verify Server is Running

Check if the server is responding:
```bash
curl http://127.0.0.1:2024/info
```

You should see a JSON response with server information.

## 🐛 Troubleshooting

### "Failed to fetch" Error
- **Safari/Brave users**: Use `USE_TUNNEL=true ./start_langgraph_studio.sh`
- **All browsers**: Make sure server is running: `curl http://127.0.0.1:2024/info`

### Port Already in Use
```bash
# Kill existing server
lsof -ti:2024 | xargs kill -9

# Or use a different port
langgraph dev --port 2025
# Then connect to: http://127.0.0.1:2025
```

### Server Won't Start
- Check if `langgraph-cli` is installed: `langgraph --version`
- Verify `.env` file exists with required variables
- Check logs: `cat /tmp/langgraph_dev.log`
