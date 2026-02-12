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
from dotenv import load_dotenv
from typing import List, Dict

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
        from assistant.retrieval.pgvector_retriever import PgVectorRetriever
        from assistant.flows.node_logic.stage1_intent_router import (
            classify_intent, handle_non_knowledge_intent, handle_crush_flow_continuation
        )

        rag_engine = RagEngine()
        retriever = PgVectorRetriever()
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
    role = "Just looking around"  # Default role
    debug_mode = False
    conversation_state = {}  # Track intent router state (crush flow, etc.)

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
                print_system("\n🗑️  Conversation history cleared!\n")
                print_portfolia(greeting)
                continue

            if user_input.lower() in ['/debug', 'debug']:
                debug_mode = not debug_mode
                status = "ON" if debug_mode else "OFF"
                print_system(f"\n🔧 Debug mode: {status}\n")
                continue

            # Detect role selection (1, 2, 3, 4)
            role_map = {
                "1": "Hiring Manager (nontechnical)",
                "2": "Software Developer",
                "3": "Just looking around",
                "4": "Looking to confess crush"
            }
            if user_input in role_map and len(chat_history) == 0:
                role = role_map[user_input]
                if debug_mode:
                    print_debug(f"Role selected: {role}")

            # Show processing indicator
            if debug_mode:
                print_debug("Processing your message...")
                print_debug(f"Current role: {role}")
                print_debug(f"Chat history: {len(chat_history)} messages")

                # Show retrieval process
                print_debug("\n🔍 Retrieving relevant chunks from knowledge base...")
                effective_query = conversation_state.get("query", user_input)
                chunks = retriever.retrieve(effective_query, top_k=3)
                chunks_list = chunks if isinstance(chunks, list) else chunks.get('chunks', [])

                if chunks_list:
                    print_debug(f"✅ Retrieved {len(chunks_list)} chunks:")
                    for i, chunk in enumerate(chunks_list, 1):
                        sim = chunk.get('similarity', 0)
                        doc = chunk.get('doc_id', 'unknown')
                        section = chunk.get('section', 'N/A')
                        content_preview = chunk.get('content', '')[:80].replace('\n', ' ')
                        print_debug(f"  {i}. [{doc}] Similarity: {sim:.3f}")
                        print_debug(f"     Section: {section}")
                        print_debug(f"     Preview: {content_preview}...")
                else:
                    print_debug("⚠️  No chunks retrieved (similarity below threshold)")

                print_debug("\n🤖 Generating response with LLM...")

            # Generate response
            try:
                # Update state for intent router
                # chat_history is set so _detect_crush_flow_from_history can
                # recover crush flow state even if fields were lost.
                conversation_state["query"] = user_input
                conversation_state["chat_history"] = chat_history
                # Clear volatile fields so they get re-classified each turn
                conversation_state.pop("message_intent", None)
                conversation_state.pop("skip_rag", None)
                conversation_state.pop("pipeline_halt", None)
                conversation_state.pop("is_self_referential", None)

                # Run intent classification BEFORE RAG
                conversation_state = classify_intent(conversation_state)
                intent = conversation_state.get("message_intent", "knowledge_query")

                if debug_mode:
                    print_debug(f"Intent classified as: {intent}")
                    print_debug(f"Crush flow state: awaiting={conversation_state.get('awaiting_crush_choice')}, step={conversation_state.get('crush_flow_step')}")

                # Handle non-knowledge intents (crush, small talk, off topic) OR crush flow continuation
                if (conversation_state.get("skip_rag") or
                    conversation_state.get("pipeline_halt") or
                    intent in ["crush_confession", "small_talk", "off_topic"]):
                    conversation_state = handle_non_knowledge_intent(conversation_state, rag_engine)
                    response = conversation_state.get("answer", "")

                    if debug_mode:
                        print_debug(f"Handled by intent router (no RAG)")
                else:
                    # Knowledge query - use RAG
                    # Use the (possibly expanded) query from intent router,
                    # NOT the raw user_input. This matters for continuations
                    # like "yes" which get expanded to the previous topic.
                    effective_query = conversation_state.get("query", user_input)
                    if debug_mode and effective_query != user_input:
                        print_debug(f"Query expanded: '{user_input}' → '{effective_query}'")
                    response = rag_engine.generate_response(
                        query=effective_query,
                        chat_history=chat_history
                    )

                # Show debug info
                if debug_mode:
                    print_debug("✅ Response generated")

                # Display response
                print_portfolia(response)

                # Update chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})

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
