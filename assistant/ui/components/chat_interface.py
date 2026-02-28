from streamlit import st

class ChatInterface:
    def __init__(self, role: str):
        self.role = role
        self.user_input = ""
        self.chat_history = []

    def display_chat(self):
        st.title("Noah's AI Assistant Chat")
        st.caption(f"Active role: {self.role}")
        self.user_input = st.text_input("You:", "")

        if st.button("Send"):
            if self.user_input:
                self.chat_history.append({"user": self.user_input})
                response = self.get_response(self.user_input)
                self.chat_history.append({"assistant": response})
                self.user_input = ""

        self.show_chat_history()

    def show_chat_history(self):
        for chat in self.chat_history:
            if "user" in chat:
                st.write(f"You: {chat['user']}")
            elif "assistant" in chat:
                st.write(f"Assistant: {chat['assistant']}")

    def get_response(self, user_input):
        # Placeholder for the actual response generation logic
        return "This is a placeholder response."

if __name__ == "__main__":
    chat_interface = ChatInterface(role="default")
    chat_interface.display_chat()
