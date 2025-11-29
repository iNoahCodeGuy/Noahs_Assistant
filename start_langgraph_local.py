#!/usr/bin/env python3
"""
LangGraph Local Server Launcher for LangSmith Studio
====================================================

Starts a local LangGraph dev server for testing with LangSmith Studio.

Usage:
    python3 start_langgraph_local.py

This will:
1. Load environment variables from .env
2. Start LangGraph dev server on http://127.0.0.1:2024
3. Provide the URL to connect LangSmith Studio
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found - continuing anyway...")
        return

    print("📊 Loading environment variables...")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("✅ Environment loaded")

def check_port(port=2024):
    """Check if port is already in use"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result == 0:
            return True
    except Exception:
        pass
    return False

def check_server_ready(url="http://127.0.0.1:2024", max_wait=45):
    """Wait for server to be ready"""
    from urllib.request import urlopen
    from urllib.error import URLError

    for i in range(max_wait):
        try:
            urlopen(f"{url}/info", timeout=2)
            return True
        except (URLError, Exception):
            time.sleep(1)
            if i % 5 == 0:
                print(".", end="", flush=True)
    return False

def main():
    print("="*60)
    print("  LangGraph Local Server for LangSmith Studio")
    print("="*60)
    print()

    # Load environment
    load_env()

    # Ensure LangSmith tracing is enabled
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    print("🔗 LangSmith tracing enabled")

    # Check port
    port = 2024
    if check_port(port):
        print(f"⚠️  Port {port} is already in use!")
        print(f"   To stop existing server: lsof -ti:{port} | xargs kill -9")
        response = input("   Kill existing process? (y/N): ").strip().lower()
        if response == 'y':
            try:
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, check=False)
                time.sleep(2)
            except Exception:
                pass
        else:
            sys.exit(1)

    # Check if langgraph-cli is installed
    try:
        result = subprocess.run(
            ["langgraph", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ langgraph-cli not found!")
        print("   Install with: pip install langgraph-cli")
        sys.exit(1)

    print("✅ langgraph-cli found")
    print()

    # Start server
    server_url = f"http://127.0.0.1:{port}"
    langsmith_url = f"https://smith.langchain.com/studio/?baseUrl={server_url}"
    project = os.getenv('LANGCHAIN_PROJECT', 'noahs-ai-assistant')

    print("🎯 Starting LangGraph dev server...")
    print(f"🌐 Server URL: {server_url}")
    print(f"📊 LangSmith Studio: {langsmith_url}")
    print(f"📈 View traces: https://smith.langchain.com/o/project/{project}")
    print()
    print("💡 The server will start in the background")
    print("   Press Ctrl+C to stop the server")
    print("="*60)
    print()

    # Start langgraph dev server
    try:
        process = subprocess.Popen(
            ["langgraph", "dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Wait for server to be ready
        print("⏳ Waiting for server to start", end="", flush=True)
        if check_server_ready(server_url):
            print()
            print()
            print("✅ Server is ready!")
            print()
            print(f"🌐 Local Server: {server_url}")
            print(f"📊 LangSmith Studio: {langsmith_url}")
            print()
            print("📝 Next steps:")
            print("   1. Open LangSmith Studio in your browser")
            print(f"   2. Connect to: {server_url}")
            print("   3. Or use the direct link above")
            print()
            print("🛑 Press Ctrl+C to stop the server")
            print()

            # Stream output
            try:
                for line in process.stdout:
                    print(line, end="")
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping server...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("✅ Server stopped")
        else:
            print()
            print()
            print("❌ Server did not become ready after 45 seconds")
            print("📋 Checking logs...")
            process.terminate()
            process.wait(timeout=2)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ Server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
