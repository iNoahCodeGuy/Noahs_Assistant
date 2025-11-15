# Docker Development Workflow for LangGraph Studio

## ⚠️ CRITICAL RULES

### Rule #1: Code Changes Require Rebuild
**Docker containers do NOT auto-reload Python code changes.**

```bash
# ❌ WRONG - This will NOT pick up your code changes
docker restart noahsaiassistant--langgraph-api-1

# ✅ CORRECT - This rebuilds with your latest code
langgraph up --port 2024
```

### Rule #2: Always Check Container Status Before Testing
```bash
# Check if containers are running
docker ps --filter "name=langgraph"

# Check if server is responding
curl -s http://localhost:2024/info | jq '.version'
```

### Rule #3: Use Proper Restart Sequence
```bash
# 1. Stop cleanly
docker stop noahsaiassistant--langgraph-api-1

# 2. Rebuild and restart (automatically rebuilds changed files)
langgraph up --port 2024
```

## 🔄 Common Development Scenarios

### Scenario A: Making Code Changes to Nodes/Flows
**When:** Editing `assistant/flows/`, `assistant/core/`, etc.

**Workflow:**
```bash
# 1. Edit your Python files
vim assistant/flows/node_logic/stage5_generation_nodes.py

# 2. Stop existing server (if running)
docker stop $(docker ps -q --filter "name=langgraph-api") 2>/dev/null || true

# 3. Wait 2 seconds for cleanup
sleep 2

# 4. Rebuild and start (picks up code changes automatically)
langgraph up --port 2024

# 5. Wait for server to be ready (~15 seconds)
sleep 15

# 6. Verify server is responding
curl http://localhost:2024/info

# 7. Test in LangGraph Studio
open http://127.0.0.1:2024
```

**Estimated time:** 30-45 seconds for full rebuild

### Scenario B: Quick Debug Logging Addition
**When:** Adding `logger.info()` or `logger.warning()` calls

**Workflow:**
```bash
# Same as Scenario A - NO SHORTCUTS!
# Docker cannot hot-reload Python changes
```

### Scenario C: Viewing Logs Without Restart
**When:** Server is running, just want to see logs

**Workflow:**
```bash
# Follow live logs
docker logs -f noahsaiassistant--langgraph-api-1

# Search recent logs
docker logs noahsaiassistant--langgraph-api-1 2>&1 | grep "DEBUG"

# Last 50 lines
docker logs --tail 50 noahsaiassistant--langgraph-api-1
```

### Scenario D: Environment Variable Changes
**When:** Changing `.env` file or environment variables

**Workflow:**
```bash
# 1. Stop containers
docker stop $(docker ps -q --filter "name=langgraph") 2>/dev/null || true

# 2. Remove containers to clear old env vars
docker rm $(docker ps -aq --filter "name=langgraph") 2>/dev/null || true

# 3. Rebuild from scratch
langgraph up --port 2024
```

## 🚫 Anti-Patterns to AVOID

### ❌ Don't: Restart without rebuild
```bash
# This keeps OLD code running!
docker restart noahsaiassistant--langgraph-api-1
```

### ❌ Don't: Expect hot reload
```python
# Changing this file and saving does NOT update running container
# assistant/flows/node_logic/stage5_generation_nodes.py
logger.info("New debug message")  # Won't appear until rebuild!
```

### ❌ Don't: Test immediately after `langgraph up`
```bash
langgraph up --port 2024
# Server not ready yet! Need to wait ~15 seconds
curl http://localhost:2024/info  # This will fail!
```

## ✅ Safe Testing Script

Create `scripts/safe_restart.sh`:
```bash
#!/bin/bash
set -e

echo "🛑 Stopping existing containers..."
docker stop $(docker ps -q --filter "name=langgraph") 2>/dev/null || true

echo "⏳ Waiting for cleanup..."
sleep 2

echo "🔨 Rebuilding with latest code..."
langgraph up --port 2024 &
LANGGRAPH_PID=$!

echo "⏳ Waiting for server to start (15s)..."
sleep 15

echo "🔍 Checking server health..."
if curl -s http://localhost:2024/info > /dev/null; then
    echo "✅ Server is ready!"
    echo "🌐 Open http://127.0.0.1:2024 to test"
else
    echo "❌ Server failed to start. Check logs:"
    echo "   docker logs noahsaiassistant--langgraph-api-1"
    exit 1
fi
```

Usage:
```bash
chmod +x scripts/safe_restart.sh
./scripts/safe_restart.sh
```

## 🐛 Debugging "Failed to Fetch" Errors

### Symptom
```
TypeError: Failed to fetch
```

### Possible Causes
1. **Container crashed** → Check `docker ps` to see if container is running
2. **Server not ready yet** → Wait 15 seconds after `langgraph up`
3. **Port conflict** → Something else using port 2024
4. **Network issue** → Docker network misconfigured

### Debug Steps
```bash
# Step 1: Check if container is running
docker ps --filter "name=langgraph-api"
# If not listed → container crashed, check logs below

# Step 2: Check container logs for errors
docker logs --tail 100 noahsaiassistant--langgraph-api-1

# Step 3: Check if port is accessible
curl http://localhost:2024/info
# If connection refused → server not started yet or crashed

# Step 4: Check Docker network
docker network inspect noahsaiassistant-_default

# Step 5: Nuclear option - full cleanup
docker stop $(docker ps -q --filter "name=langgraph") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=langgraph") 2>/dev/null || true
docker volume prune -f
langgraph up --port 2024
```

## 📊 Testing Checklist

Before submitting a query in LangGraph Studio:

- [ ] Container is running: `docker ps | grep langgraph-api`
- [ ] Server responds: `curl http://localhost:2024/info`
- [ ] Logs show "Ready": `docker logs noahsaiassistant--langgraph-api-1 | grep "Ready"`
- [ ] Code changes applied: Check container start time vs file modification time
- [ ] No error messages: `docker logs noahsaiassistant--langgraph-api-1 | grep ERROR`

## ⏱️ Expected Timing

| Action | Expected Duration |
|--------|------------------|
| `docker stop` | 2-5 seconds |
| `langgraph up` full rebuild | 20-30 seconds |
| Server ready after build | 10-15 seconds |
| **Total safe restart** | **~45 seconds** |

## 🔄 CI/CD Implications

When deploying to production (Vercel):
- Vercel automatically rebuilds on git push
- No manual restart needed
- Serverless functions cold start ~1-2 seconds
- Docker timing issues don't affect production

## 📝 Quick Reference Card

```bash
# SAFE WORKFLOW (copy-paste friendly)
docker stop $(docker ps -q --filter "name=langgraph-api") 2>/dev/null || true && \
sleep 2 && \
langgraph up --port 2024 & \
sleep 15 && \
curl -s http://localhost:2024/info && \
echo "✅ Ready to test at http://127.0.0.1:2024"
```
