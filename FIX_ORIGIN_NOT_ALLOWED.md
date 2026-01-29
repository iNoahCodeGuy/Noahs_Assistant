# Fix: "Origin is not allowed" Error

## Problem
You're seeing this error in LangSmith Studio:
```
Error: Failed to connect to Agent Server because the origin is not allowed.
```

## Solution: Use Secure Tunnel (Recommended)

The LangGraph dev server blocks requests from external origins (like LangSmith Studio) by default. The easiest fix is to use the `--tunnel` flag.

### Quick Fix

**Option 1: Use the connection script (easiest)**
```bash
python3 connect_langsmith_studio.py
```
When prompted "Use secure tunnel? (Y/n):", press **Enter** or type **Y**

**Option 2: Start server manually with tunnel**
```bash
export LANGCHAIN_TRACING_V2=true
langgraph dev --tunnel
```

### What happens:
1. The server starts with a secure HTTPS tunnel
2. You'll see a tunnel URL like: `https://xxxxx.trycloudflare.com`
3. Use this URL in LangSmith Studio instead of `http://127.0.0.1:2024`

### Connect to LangSmith Studio:

1. Copy the tunnel URL from the terminal output
2. Go to: https://smith.langchain.com/studio/
3. When prompted for server URL, paste the tunnel URL
4. Or use this format: `https://smith.langchain.com/studio/?baseUrl=<your-tunnel-url>`

## Alternative: Add Origin to Allowed Origins (If not using tunnel)

If you're using `http://127.0.0.1:2024` (without tunnel), you need to add LangSmith Studio's origin:

1. In the LangSmith Studio connection dialog, expand **"Advanced Settings"**
2. Under **"Allowed Origins"**, click **"+ Allowed Origin"**
3. Add: `https://smith.langchain.com`
4. Click **"Connect"**

**Note:** The tunnel method (Solution 1) is recommended as it works automatically without manual configuration.

## Why This Happens

- LangGraph dev server blocks cross-origin requests by default for security
- LangSmith Studio runs on `https://smith.langchain.com` (different origin)
- The tunnel creates an HTTPS endpoint that works with all origins
- This is the official recommended approach from LangChain

## Still Having Issues?

1. **Make sure you're using the tunnel URL** (not localhost)
2. **Check the server is running**: Look for the tunnel URL in terminal output
3. **Try a different browser**: Chrome/Firefox work best
4. **Restart the server**: Stop (Ctrl+C) and restart with `--tunnel`
