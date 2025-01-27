import streamlit as st
import openai
import time

# Set page configuration
st.set_page_config(
    page_title="DeepSeek Reasoner Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for storing messages
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]

# Sidebar for configuration
st.sidebar.header("Configuration")

# Access API Key and Base URL from Streamlit Secrets
api_key = st.secrets["DEEPEEK_API_KEY"]
base_url = st.secrets.get("DEEPEEK_API_BASE_URL", "https://api.deepseek.com")

# Function to get assistant response
def get_assistant_response(messages, api_key, base_url):
    try:
        # Initialize OpenAI client with custom base_url
        openai.api_key = api_key
        openai.api_base = base_url

        # Create a chat completion with streaming
        response = openai.ChatCompletion.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )

        reasoning_content = ""
        content = ""
        assistant_message = ""

        # Iterate over streamed chunks
        for chunk in response:
            if 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                if 'reasoning_content' in delta:
                    reasoning_content += delta['reasoning_content']
                if 'content' in delta:
                    content += delta['content']
                    assistant_message += delta['content']
                    # Display the assistant's message in real-time
                    st.session_state.assistant_placeholder.markdown(assistant_message + "▌")
                    # Small delay to simulate streaming
                    time.sleep(0.05)

        # Replace the placeholder with the final message
        st.session_state.assistant_placeholder.markdown(assistant_message)
        return reasoning_content, content

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "", ""

# Main interface
st.title("🤖 DeepSeek Reasoner Chatbot for Coding")

# Display conversation history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    elif msg['role'] == 'assistant':
        st.markdown(f"**Assistant:** {msg['content']}")

st.markdown("---")

# Input area for user prompt
with st.form(key='user_input_form', clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    # Append user message to conversation
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    st.markdown(f"**You:** {user_input}")

    # Placeholder for assistant's response
    st.session_state.assistant_placeholder = st.empty()

    # Get assistant response
    reasoning, content = get_assistant_response(
        st.session_state.messages,
        api_key,
        base_url
    )

    # Append assistant message to conversation
    st.session_state.messages.append({"role": "assistant", "content": content})

    st.markdown("---")

# Footer
st.markdown("""
---
*Powered by DeepSeek Reasoner and Streamlit.*
""")
