#!/usr/bin/env python3
"""
LangSmith Connection Launcher
============================

Simple launcher for LangSmith Studio testing with your AI assistant.

Usage:
    python3 langsmith_connect.py

Options:
    1) Test LangSmith integration
    2) Start Streamlit with tracing
    3) Show dashboard links
"""

import os
import subprocess
import sys

def main():
    print("="*50)
    print("  LangSmith Studio Connection")
    print("="*50)

    # Load environment
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        sys.exit(1)

    # Simple environment loading
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'

    # Check essential variables
    required = ['LANGSMITH_API_KEY', 'OPENAI_API_KEY', 'SUPABASE_URL']
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        sys.exit(1)

    print("✅ Environment loaded")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'noahs-ai-assistant')}")
    print()

    print("Choose option:")
    print("1) Test LangSmith integration")
    print("2) Start Streamlit with tracing")
    print("3) Show dashboard links")
    print("4) Exit")

    choice = input("\nEnter (1-4): ").strip()

    if choice == '1':
        print("\n🧪 Running LangSmith test...")
        try:
            subprocess.run([sys.executable, "scripts/test_langsmith_tracing.py"], check=True)
            print("✅ Test completed!")
        except subprocess.CalledProcessError:
            print("❌ Test failed (but LangSmith tracing likely worked)")
        except FileNotFoundError:
            print("❌ Test script not found")

    elif choice == '2':
        print("\n🌐 Starting Streamlit with LangSmith tracing...")
        print("🔗 URL: http://localhost:8501")
        print("📊 Traces: https://smith.langchain.com/")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/main.py", "--server.port=8501"])
        except KeyboardInterrupt:
            print("\n✅ Streamlit stopped")

    elif choice == '3':
        project = os.getenv('LANGCHAIN_PROJECT', 'noahs-ai-assistant')
        print(f"\n📊 LangSmith Dashboard:")
        print(f"🌐 Main: https://smith.langchain.com/")
        print(f"📂 Project: https://smith.langchain.com/o/project/{project}")
        print("\n💡 Tip: Filter traces by session ID for debugging")

    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
