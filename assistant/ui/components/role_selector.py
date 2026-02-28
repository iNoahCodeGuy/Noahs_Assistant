from streamlit import st

ROLE_OPTIONS = [
    "Hiring Manager (nontechnical)",
    "Hiring Manager (technical)",
    "Software Developer",
    "Just looking around",
    "Looking to confess crush"
]

class RoleSelector:
    """Handles initial role declaration and persistence in session state."""
    def __init__(self, session_key: str = "selected_role"):
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = None

    def select_role(self) -> str:
        st.write("Hello, I am Noah’s AI Assistant. To better provide assistance, which best describes you?")
        chosen = st.selectbox("Select your role:", ROLE_OPTIONS, index=0)
        st.session_state[self.session_key] = chosen
        return chosen

    def get_role(self) -> str:
        return st.session_state.get(self.session_key)

# Backward compatibility (optional)
def role_selector():
    selector = RoleSelector()
    return selector.select_role()
