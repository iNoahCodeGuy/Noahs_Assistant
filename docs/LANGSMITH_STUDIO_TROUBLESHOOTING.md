# LangSmith Studio Troubleshooting Guide

## "TypeError: Failed to fetch" Error

This error occurs when the LangSmith Studio web interface cannot connect to your local LangGraph dev server.

### ⚡ Quick Fix (Most Common Issue)

**If you're using Safari or Brave browser**, they block non-HTTPS connections to localhost. The startup script now uses a secure tunnel by default:

```bash
./start_langgraph_studio.sh
```

The script automatically uses `--tunnel` which creates a secure HTTPS connection that works with all browsers.

### Quick Fixes

1. **Run the diagnostic script:**
   ```bash
   ./scripts/diagnose_langgraph_studio.sh
   ```
   This will check all common issues and provide specific guidance.

2. **Use the improved startup script (with tunnel):**
   ```bash
   ./start_langgraph_studio.sh
   ```
   The script now waits for the server to be ready and uses a secure tunnel by default.

3. **If tunnel doesn't work, try without tunnel:**
   ```bash
   USE_TUNNEL=false ./start_langgraph_studio.sh
   ```

### Common Causes & Solutions

#### 1. Server Not Running

**Symptoms:**
- "Failed to fetch" error immediately
- Port 2024 shows as not in use

**Solution:**
```bash
# Start the server
./start_langgraph_studio.sh

# Or manually:
langgraph dev
```

**Verify:**
```bash
curl http://127.0.0.1:2024/info
# Should return JSON response
```

#### 2. Server Not Ready Yet

**Symptoms:**
- Browser opens before server is ready
- Error appears briefly then works

**Solution:**
The improved startup script now waits up to 30 seconds for the server to be ready. If you're starting manually:

```bash
# Wait for server to be ready
while ! curl -s http://127.0.0.1:2024/info > /dev/null; do
    echo "Waiting for server..."
    sleep 2
done
echo "Server is ready!"
```

#### 3. Port Already in Use

**Symptoms:**
- "Port 2024 is already in use" error
- Server fails to start

**Solution:**
```bash
# Find and kill the process using port 2024
lsof -ti:2024 | xargs kill -9

# Or use a different port
langgraph dev --port 2025
# Then connect to: http://127.0.0.1:2025
```

#### 4. Browser Security (Safari/Brave) - MOST COMMON

**Symptoms:**
- "Failed to fetch" error in Safari or Brave
- Works in Chrome/Firefox but not Safari
- Console shows network errors

**Root Cause:**
Safari and Brave block non-HTTPS connections to localhost for security reasons.

**Solution:**
Use the secure tunnel option (enabled by default in startup script):

```bash
# Automatic (recommended)
./start_langgraph_studio.sh

# Manual with tunnel
langgraph dev --tunnel

# The tunnel URL will be displayed, use it in LangSmith Studio:
# https://smith.langchain.com/studio/?baseUrl=https://xxxxx.trycloudflare.com
```

**Alternative for Safari:**
1. Open Safari Preferences
2. Go to Advanced tab
3. Check "Show Develop menu in menu bar"
4. Develop > Disable Local File Restrictions (may not fully solve the issue)

#### 5. CORS Issues

**Symptoms:**
- Server is running but browser can't connect
- Console shows CORS errors

**Solution:**
LangGraph dev server should handle CORS automatically. If you're still having issues:

1. Make sure you're using the correct URL format:
   ```
   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
   # Or with tunnel:
   https://smith.langchain.com/studio/?baseUrl=https://xxxxx.trycloudflare.com
   ```

2. Try accessing the server directly first:
   ```bash
   curl http://127.0.0.1:2024/info
   ```

3. Check browser console (F12) for detailed error messages

#### 6. Network/Firewall Blocking

**Symptoms:**
- Server runs but browser can't connect
- Works on some networks but not others

**Solution:**
```bash
# Test local connectivity
curl http://127.0.0.1:2024/info

# Test if LangSmith website is accessible
curl https://smith.langchain.com

# Check firewall settings (macOS)
# System Preferences > Security & Privacy > Firewall
```

#### 7. Graph Import Errors

**Symptoms:**
- Server starts but fails to load graph
- Error in server logs about import failures

**Solution:**
```bash
# Test if graph can be imported
python3 -c "
import sys
sys.path.insert(0, '.')
from assistant.flows.conversation_flow import graph
print('✅ Graph imported successfully')
"

# Check langgraph.json
cat langgraph.json

# Verify dependencies are installed
pip install -r requirements.txt
```

#### 8. Environment Variables Missing

**Symptoms:**
- Server starts but graph fails to initialize
- Errors about missing API keys

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify required variables
grep -E "LANGCHAIN_API_KEY|OPENAI_API_KEY|SUPABASE" .env

# Load environment manually
export $(cat .env | grep -v '^#' | xargs)
export LANGCHAIN_TRACING_V2=true
langgraph dev
```

### Step-by-Step Debugging

1. **Check if server is running:**
   ```bash
   lsof -i :2024
   ```

2. **Test server connectivity:**
   ```bash
   curl http://127.0.0.1:2024/info
   ```

3. **Check server logs:**
   ```bash
   # If using the startup script, logs are in:
   tail -f /tmp/langgraph_dev.log

   # Or if running manually, check terminal output
   ```

4. **Verify browser can reach server:**
   - Open: http://127.0.0.1:2024/info
   - Should see JSON response

5. **Check browser console:**
   - Press F12 in browser
   - Look for detailed error messages
   - Check Network tab for failed requests

6. **Try alternative connection method:**
   - Instead of web-based Studio, try desktop app if available
   - Or use direct API access: http://127.0.0.1:2024

### Manual Server Start (For Debugging)

If the script isn't working, start manually to see errors:

```bash
# Load environment
export $(cat .env | grep -v '^#' | xargs)
export LANGCHAIN_TRACING_V2=true

# Start server in foreground to see errors
langgraph dev

# In another terminal, test connection
curl http://127.0.0.1:2024/info
```

### Alternative: Use LangSmith Dashboard Directly

If LangGraph Studio continues to have issues, you can still use LangSmith tracing:

1. **Run your assistant with tracing:**
   ```bash
   export $(cat .env | grep -v '^#' | xargs)
   export LANGCHAIN_TRACING_V2=true
   python3 langsmith_connect.py
   # Choose option 2 for Streamlit
   ```

2. **View traces in dashboard:**
   - https://smith.langchain.com/o/project/noahs-ai-assistant
   - All traces appear in real-time
   - Filter by session ID for debugging

### Getting Help

If none of these solutions work:

1. **Run full diagnostics:**
   ```bash
   ./scripts/diagnose_langgraph_studio.sh > diagnostics.txt
   ```

2. **Collect information:**
   - Server logs: `/tmp/langgraph_dev.log`
   - Browser console errors (F12)
   - Output from diagnostic script

3. **Check LangGraph documentation:**
   - https://langchain-ai.github.io/langgraph/
   - LangGraph CLI: `langgraph --help`

### Prevention

To avoid this error in the future:

1. Always use the startup script: `./start_langgraph_studio.sh`
2. Wait for "Server is ready!" message before connecting
3. Keep LangGraph CLI updated: `pip install --upgrade langgraph-cli`
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
