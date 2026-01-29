# Fix: SSL Certificate Verification Error

## Problem
When running `langgraph dev --tunnel`, you get:
```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

This happens because Python can't verify SSL certificates when downloading cloudflared.

## Solution 1: Install Certificates (Recommended)

Run this command in your terminal:

```bash
# Install certificates for Python
/Applications/Python\ 3.12/Install\ Certificates.command
```

If that path doesn't exist, try:
```bash
# Find your Python installation
python3 -c "import sys; print(sys.executable)"

# Then run (adjust path as needed):
/Applications/Python\ 3.12/Install\ Certificates.command
```

Or use the Python installer:
```bash
# Download and run the certificate installer
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Solution 2: Manually Install cloudflared

If certificate installation doesn't work, install cloudflared manually:

```bash
# macOS (using Homebrew)
brew install cloudflared

# Or download directly
# Visit: https://github.com/cloudflare/cloudflared/releases
# Download for macOS and add to PATH
```

Then try the tunnel again - langgraph should detect the installed cloudflared.

## Solution 3: Use Localhost + Add Origin (No Tunnel)

If you can't fix the SSL issue, use localhost and add the origin:

1. Start server without tunnel:
   ```bash
   langgraph dev
   ```

2. In LangSmith Studio connection dialog:
   - Base URL: `http://127.0.0.1:2024`
   - Advanced Settings → Allowed Origins → Add: `https://smith.langchain.com`
   - Click "Connect"

## Solution 4: Fix Python SSL Context (Advanced)

If you're comfortable with Python, you can temporarily disable SSL verification:

```bash
# NOT RECOMMENDED for production, but works for local dev
export PYTHONHTTPSVERIFY=0
langgraph dev --tunnel
```

**Warning:** This disables SSL verification system-wide. Only use for local development.

## Verify Fix

After applying a solution, try:
```bash
python3 connect_langsmith_studio.py
```

Or manually:
```bash
langgraph dev --tunnel
```

You should see the tunnel URL appear without SSL errors.
