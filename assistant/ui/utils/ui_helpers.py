def format_response(response):
    # Formats the response for display in the UI
    return response.strip()

def validate_input(user_input):
    # Validates user input to ensure it meets criteria
    if not user_input:
        raise ValueError("Input cannot be empty.")
    return True

def create_role_prompt(roles):
    # Creates a prompt for role selection
    return f"Please select your role: {', '.join(roles)}"

def display_error_message(message):
    # Displays an error message in the UI
    return f"Error: {message}"

def format_timestamp(timestamp):
    # Formats a timestamp for display
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")
