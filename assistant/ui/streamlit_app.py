from streamlit import st
from components.role_selector import RoleSelector
from components.chat_interface import ChatInterface
from components.analytics_panel import AnalyticsPanel

def main():
    st.title("Noah's AI Assistant")

    # Role selection
    role = RoleSelector().select_role()

    # Chat interface
    chat_interface = ChatInterface(role)
    chat_interface.display_chat()

    # Analytics panel
    AnalyticsPanel().display_metrics()

if __name__ == "__main__":
    main()
