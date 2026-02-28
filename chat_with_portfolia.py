#!/usr/bin/env python3
"""
🤖 Chat with Portfolia - Interactive Terminal Interface

Just run this script and start chatting!
Portfolia will greet you first, then you can have a conversation.

Usage:
    python3 chat_with_portfolia.py

Commands:
    /quit or /exit - Exit the chat
    /clear - Clear conversation history
    /debug - Toggle debug mode (show retrieval details)
"""

import os
import sys
import uuid
from dotenv import load_dotenv
from typing import Any, Dict, List

load_dotenv()

# Colors for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

def print_portfolia(message: str):
    """Print Portfolia's message in blue"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Portfolia:{Colors.END} {message}\n")

def print_user(message: str):
    """Print user's message in green"""
    print(f"{Colors.GREEN}You: {message}{Colors.END}")

def print_debug(message: str):
    """Print debug info in yellow"""
    print(f"{Colors.YELLOW}[DEBUG] {message}{Colors.END}")

def print_system(message: str):
    """Print system messages in cyan"""
    print(f"{Colors.CYAN}{message}{Colors.END}")

def print_separator():
    """Print a separator line"""
    print(f"{Colors.MAGENTA}{'=' * 80}{Colors.END}")

def main():
    # Initialize
    print_separator()
    print(f"{Colors.BOLD}{Colors.CYAN}🤖 PORTFOLIA - Noah's AI Assistant{Colors.END}")
    print_separator()
    print("\nInitializing...")

    try:
        from assistant.core.rag_engine import RagEngine
        from assistant.flows.conversation_flow import run_conversation_flow

        rag_engine = RagEngine()
        print_system("✅ Portfolia initialized and ready to chat!")
        print_system("\nCommands: /quit (exit) | /clear (reset) | /debug (toggle debug)\n")

    except ImportError as e:
        print(f"{Colors.RED}❌ Failed to import required modules: {e}{Colors.END}")
        print(f"{Colors.YELLOW}Make sure you're in the project directory and dependencies are installed.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Initialization failed: {e}{Colors.END}")
        sys.exit(1)

    # Chat state
    chat_history: List[Dict[str, str]] = []
    session_memory: Dict[str, Any] = {}
    session_id = f"terminal-{uuid.uuid4().hex[:8]}"
    role = "Just looking around"  # Default role
    debug_mode = False

    # Portfolia greets first (as per conversation examples)
    greeting = """Hey, I'm Portfolia -- Noah's AI assistant.

What brings you here?
1️⃣ Learn more about Noah
2️⃣ See what Noah has built
3️⃣ Just looking around
4️⃣ Confess a crush 💌"""

    print_portfolia(greeting)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}You: {Colors.END}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print_system("\n👋 Thanks for chatting! Goodbye!")
                break

            if user_input.lower() in ['/clear', 'clear']:
                chat_history = []
                session_memory = {}
                session_id = f"terminal-{uuid.uuid4().hex[:8]}"
                role = "Just looking around"
                print_system("\n🗑️  Conversation history cleared!\n")
                print_portfolia(greeting)
                continue

            if user_input.lower() in ['/debug', 'debug']:
                debug_mode = not debug_mode
                status = "ON" if debug_mode else "OFF"
                print_system(f"\n🔧 Debug mode: {status}\n")
                continue

            # Detect role selection (1, 2, 3, 4 or free-text menu labels)
            role_map = {
                "1": "Learn more about Noah",
                "2": "See what Noah has built",
                "3": "Just looking around",
                "4": "Looking to confess crush",
            }
            _text_role_map = {
                "learn more about noah": "Learn more about Noah",
                "see what noah has built": "See what Noah has built",
                "just looking around": "Just looking around",
                "confess a crush": "Looking to confess crush",
            }
            selection = role_map.get(user_input) or _text_role_map.get(user_input.lower())
            if selection and len(chat_history) == 0:
                role = selection
                if debug_mode:
                    print_debug(f"Role selected: {role}")

            # Show processing indicator
            if debug_mode:
                print_debug("Processing your message...")
                print_debug(f"Current role: {role}")
                print_debug(f"Chat history: {len(chat_history)} messages")

            # Run the full conversation pipeline (same path as production API)
            try:
                state: Dict[str, Any] = {
                    "role": role,
                    "query": user_input,
                    "chat_history": chat_history,
                    "session_id": session_id,
                    "session_memory": session_memory,
                }

                result = run_conversation_flow(state, rag_engine, session_id=session_id)

                response = result.get("answer", "")

                # Persist state that accumulates across turns
                chat_history = result.get("chat_history", chat_history)
                session_memory = result.get("session_memory", session_memory)
                role = result.get("role", role)

                # Show debug info
                if debug_mode:
                    print_debug(f"Intent: {result.get('message_intent', 'N/A')}")
                    print_debug(f"Phase: {result.get('conversation_phase', 'N/A')}")
                    print_debug(f"Chunks retrieved: {len(result.get('retrieved_chunks', []))}")
                    print_debug(f"Pipeline halt: {result.get('pipeline_halt', False)}")
                    print_debug("✅ Response generated")

                # Display response
                print_portfolia(response)

                # Keep history manageable (last 10 exchanges = 20 messages)
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]

            except Exception as e:
                print(f"{Colors.RED}❌ Error generating response: {e}{Colors.END}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
                print_system("\n💡 Try rephrasing your question or type /clear to start over.\n")

        except KeyboardInterrupt:
            print_system("\n\n👋 Interrupted. Thanks for chatting!")
            break
        except EOFError:
            print_system("\n\n👋 EOF received. Goodbye!")
            break
        except Exception as e:
            print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
            if debug_mode:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
