import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class Memory:
    def __init__(self, persistence_file: str = "data/session_memory.json"):
        self.persistence_file = persistence_file
        self.session_data = {}
        self.load_persistent_data()
        # Ephemeral role separate from persisted session contexts
        self._active_role: str | None = None

    def load_persistent_data(self):
        """Load persistent memory from file."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    self.session_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.session_data = {}

    def save_persistent_data(self):
        """Save memory to persistent storage."""
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except IOError:
            pass  # Fail silently if can't save

    def store_session_context(self, session_id: str, role: str, chat_history: List[Dict[str, str]]):
        """Store session context for persistence across refreshes."""
        self.session_data[session_id] = {
            "role": role,
            "chat_history": chat_history[-10:],  # Keep last 10 messages
            "timestamp": datetime.now().isoformat()
        }
        self.save_persistent_data()

    def retrieve_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored session context."""
        return self.session_data.get(session_id)

    def add_to_working_memory(self, key: str, value: Any):
        """Add data to working memory (current session)."""
        if "working_memory" not in self.session_data:
            self.session_data["working_memory"] = {}
        self.session_data["working_memory"][key] = value

    def get_from_working_memory(self, key: str, default=None):
        """Retrieve from working memory."""
        return self.session_data.get("working_memory", {}).get(key, default)

    def clear_session(self, session_id: str):
        """Clear specific session data."""
        if session_id in self.session_data:
            del self.session_data[session_id]
            self.save_persistent_data()

    # --- Role helpers expected by router/tests ---
    def set_role(self, role: str):
        """Set active role for current in-memory context (not auto-persisted)."""
        self._active_role = role

    def get_role(self) -> str | None:
        """Return active role if set."""
        return self._active_role
