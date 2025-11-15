"""
Simple wrapper for LangGraph Studio compatibility.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the graph from the actual location
from src.flows.conversation_flow import graph

# Export for LangGraph Studio
__all__ = ['graph']
