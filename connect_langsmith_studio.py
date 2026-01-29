#!/usr/bin/env python3
"""
Connect Local Server to LangSmith Studio
=========================================

This script will:
1. Check prerequisites (langgraph-cli, environment variables)
2. Start the LangGraph dev server on http://127.0.0.1:2024
3. Provide the connection URL for LangSmith Studio
4. Optionally open LangSmith Studio in your browser

Usage:
    python3 connect_langsmith_studio.py
"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
import re
import threading
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found")
        print("   Creating a template .env file...")
        return False

    print("📊 Loading environment variables from .env...")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
    print("✅ Environment loaded")
    return True

def check_prerequisites():
    """Check if all prerequisites are met"""
    issues = []

    # Check langgraph-cli
    try:
        result = subprocess.run(
            ["langgraph", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            issues.append("langgraph-cli is installed but not working properly")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("langgraph-cli not found. Install with: pip install langgraph-cli")

    # Check environment variables
    if not os.getenv("LANGSMITH_API_KEY"):
        issues.append("LANGSMITH_API_KEY not set in environment")

    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set in environment")

    return issues

def check_port(port=2024):
    """Check if port is already in use"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except Exception:
        return False

def extract_tunnel_url(text):
    """Extract tunnel URL from text using various patterns"""
    patterns = [
        r'https://[a-z0-9-]+\.trycloudflare\.com[^\s]*',
        r'https://[a-z0-9-]+\.cloudflaretunnel\.com[^\s]*',
        r'Tunnel URL[:\s]+(https://[^\s]+)',
        r'Public URL[:\s]+(https://[^\s]+)',
        r'https://[a-z0-9-]+\.trycloudflare\.com',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            url = matches[0] if isinstance(matches[0], str) else matches[0]
            # Clean up the URL
            url = url.rstrip('.,;:')
            if url.startswith('https://'):
                return url
    return None

def wait_for_server(url="http://127.0.0.1:2024", max_wait=45):
    """Wait for server to be ready"""
    print("⏳ Waiting for server to start", end="", flush=True)
    for i in range(max_wait):
        try:
            urlopen(f"{url}/info", timeout=2)
            print()
            return True
        except (URLError, Exception):
            time.sleep(1)
            if i % 5 == 0:
                print(".", end="", flush=True)
    print()
    return False

def main():
    print("="*70)
    print("  Connect Local Server to LangSmith Studio")
    print("="*70)
    print()

    # Load environment
    env_loaded = load_env()
    if not env_loaded:
        print("\n⚠️  Please create a .env file with required variables:")
        print("   LANGSMITH_API_KEY=lsv2_pt_...")
        print("   OPENAI_API_KEY=sk-...")
        print("   LANGCHAIN_PROJECT=noahs-ai-assistant")
        print("   LANGCHAIN_TRACING_V2=true")
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(1)

    # Ensure LangSmith tracing is enabled
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    print("🔗 LangSmith tracing enabled")

    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    issues = check_prerequisites()
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 Please fix these issues before continuing")
        sys.exit(1)
    print("✅ All prerequisites met")

    # Check port
    port = 2024
    if check_port(port):
        print(f"\n⚠️  Port {port} is already in use!")
        response = input("   Kill existing process and continue? (y/N): ").strip().lower()
        if response == 'y':
            try:
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, check=False)
                time.sleep(2)
                print("✅ Port cleared")
            except Exception:
                print("❌ Failed to clear port")
                sys.exit(1)
        else:
            print("❌ Cannot proceed with port in use")
            sys.exit(1)

    # Prepare server info
    server_url = f"http://127.0.0.1:{port}"
    langsmith_url = f"https://smith.langchain.com/studio/?baseUrl={server_url}"
    project = os.getenv('LANGCHAIN_PROJECT', 'noahs-ai-assistant')

    print("\n" + "="*70)
    print("🎯 Starting LangGraph dev server...")
    print("="*70)
    print(f"🌐 Server URL: {server_url}")
    print(f"📊 LangSmith Studio: {langsmith_url}")
    print(f"📈 View traces: https://smith.langchain.com/o/project/{project}")
    print()

    # Ask about tunnel (recommend localhost for iteration)
    print("\n🔒 Connection method:")
    print("   For quick iteration: Use localhost (stable URL, auto-reload)")
    print("   For Safari/Brave: Use tunnel (URL changes each restart)")
    print()
    print("   💡 Recommended: localhost for development (faster iteration)")
    print("   💡 See IDEAL_LANGSMITH_WORKFLOW.md for best practices")
    print()
    use_tunnel = input("   Use secure tunnel? (y/N): ").strip().lower()
    tunnel_flag = ["--tunnel"] if use_tunnel == 'y' else []

    if tunnel_flag:
        print("   ✅ Using secure tunnel (for Safari/Brave or sharing)")
        print("   ⚠️  Note: Tunnel URL changes each restart - you'll need to reconnect")
        print("   💡 If SSL errors occur, try: /Applications/Python\\ 3.12/Install\\ Certificates.command")
    else:
        print("   ✅ Using localhost (recommended for iteration)")
        print("   📝 First time: Add 'https://smith.langchain.com' to Allowed Origins in LangSmith Studio")
        print("   💡 URL stays stable - no reconnection needed after restarts!")
        print("   💡 Server auto-reloads on code changes - just save and test!")

    # Start langgraph dev server
    try:
        process = subprocess.Popen(
            ["langgraph", "dev"] + tunnel_flag,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Detect tunnel URL from output if using tunnel
        tunnel_url = None
        output_buffer = []

        if tunnel_flag:
            print("\n⏳ Starting server and detecting tunnel URL...")
            print("   (This may take 10-20 seconds)")
            print()

            # Read output line by line to detect tunnel URL
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.5)
                    continue

                output_buffer.append(line)
                print(line, end="", flush=True)

                # Try to extract tunnel URL
                tunnel_url = extract_tunnel_url(line)
                if tunnel_url:
                    print()
                    print("="*70)
                    print("🔗 TUNNEL URL DETECTED!")
                    print("="*70)
                    print()
                    print(f"📋 Copy this URL:")
                    print(f"   {tunnel_url}")
                    print()
                    actual_langsmith_url = f"https://smith.langchain.com/studio/?baseUrl={tunnel_url}"
                    print(f"📊 Or use this direct link:")
                    print(f"   {actual_langsmith_url}")
                    print()
                    print("="*70)
                    print()
                    break

                # Check if process died
                if process.poll() is not None:
                    break

            if not tunnel_url:
                print()
                print("⚠️  Tunnel URL not automatically detected")
                print("   Look for a URL like 'https://xxxxx.trycloudflare.com' in the output above")
        else:
            # No tunnel - wait for regular server
            if wait_for_server(server_url):
                pass

        # Check if server is ready (either tunnel or local)
        server_ready = tunnel_url is not None or (not tunnel_flag and wait_for_server(server_url))

        if server_ready or tunnel_flag:
            if not tunnel_flag or tunnel_url:
                print()
                print("="*70)
                print("✅ Server is ready!")
                print("="*70)
                print()

                if tunnel_flag and tunnel_url:
                    # Display tunnel URL prominently
                    print("🔒 Secure tunnel is active")
                    print()
                    print("📋 TUNNEL URL (copy this):")
                    print(f"   {tunnel_url}")
                    print()
                    actual_langsmith_url = f"https://smith.langchain.com/studio/?baseUrl={tunnel_url}"
                    print("📊 LangSmith Studio connection:")
                    print(f"   {actual_langsmith_url}")
                    print()
                    print("📝 Next steps:")
                    print("   1. Copy the tunnel URL above")
                    print("   2. Go to: https://smith.langchain.com/studio/")
                    print("   3. Paste the tunnel URL in the 'Base URL' field")
                    print("   4. Click 'Connect'")
                    print()

                    # Ask if user wants to open browser
                    response = input("🌐 Open LangSmith Studio in browser? (Y/n): ").strip().lower()
                    if response != 'n':
                        try:
                            webbrowser.open(actual_langsmith_url)
                            print("✅ Opened LangSmith Studio in browser")
                        except Exception:
                            print("⚠️  Could not open browser automatically")
                            print(f"   Please manually open: {actual_langsmith_url}")
                elif tunnel_flag and not tunnel_url:
                    # Tunnel enabled but URL not detected yet
                    print("🔒 Secure tunnel is starting...")
                    print("📋 Continue reading the output above to find the tunnel URL")
                    print("   It will look like: https://xxxxx.trycloudflare.com")
                    print()
                    print("💡 Once you see the tunnel URL:")
                    print("   1. Copy it from the output")
                    print("   2. Go to: https://smith.langchain.com/studio/")
                    print("   3. Paste it in the 'Base URL' field")
                else:
                    # No tunnel
                    print(f"🌐 Local Server: {server_url}")
                    print(f"📊 LangSmith Studio: {langsmith_url}")
                    print()
                    print("📝 Next steps:")
                    print("   1. Open LangSmith Studio in your browser")
                    print(f"   2. Connect to: {server_url}")
                    print("   3. Or use the direct link above")
                    print()

                    # Ask if user wants to open browser
                    response = input("🌐 Open LangSmith Studio in browser? (Y/n): ").strip().lower()
                    if response != 'n':
                        try:
                            webbrowser.open(langsmith_url)
                            print("✅ Opened LangSmith Studio in browser")
                        except Exception:
                            print("⚠️  Could not open browser automatically")
                            print(f"   Please manually open: {langsmith_url}")

            print()
            print("🛑 Press Ctrl+C to stop the server")
            print("="*70)
            print()

            # Continue streaming output (tunnel URL already detected if using tunnel)
            try:
                # If we already detected tunnel URL, continue reading
                if tunnel_flag and tunnel_url:
                    # Already printed the tunnel URL, now just stream remaining output
                    for line in process.stdout:
                        print(line, end="")
                elif tunnel_flag and not tunnel_url:
                    # Still looking for tunnel URL, continue reading
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        # Keep trying to detect tunnel URL
                        if not tunnel_url:
                            detected = extract_tunnel_url(line)
                            if detected:
                                tunnel_url = detected
                                print()
                                print("="*70)
                                print("🔗 TUNNEL URL DETECTED!")
                                print("="*70)
                                print()
                                print(f"📋 Copy this URL:")
                                print(f"   {tunnel_url}")
                                print()
                                actual_langsmith_url = f"https://smith.langchain.com/studio/?baseUrl={tunnel_url}"
                                print(f"📊 Or use this direct link:")
                                print(f"   {actual_langsmith_url}")
                                print("="*70)
                                print()
                else:
                    # No tunnel, just stream output
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
            print("❌ Server did not become ready after 45 seconds")
            print("📋 Checking logs...")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
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
