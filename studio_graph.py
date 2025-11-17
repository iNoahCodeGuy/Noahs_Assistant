"""
LangGraph Studio compatible graph definition.
"""

# Import necessary modules
import sys
import os
from pathlib import Path

# Add assistant to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the graph
try:
    from assistant.flows.conversation_flow import graph
    print("✅ Successfully imported conversation flow graph")
except ImportError as e:
    print(f"❌ Failed to import graph: {e}")
    # Create a minimal fallback graph
    from langgraph.graph import StateGraph
    from typing import Dict, Any

    def dummy_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": "LangGraph Studio connection established!"}

    workflow = StateGraph(dict)
    workflow.add_node("dummy", dummy_node)
    workflow.set_entry_point("dummy")
    workflow.set_finish_point("dummy")
    graph = workflow.compile()
    print("✅ Using fallback graph for LangGraph Studio")
