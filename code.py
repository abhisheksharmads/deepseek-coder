import streamlit as st
from openai import OpenAI
from typing import Generator, Tuple

class DeepSeekReasonerChat:
    """
    A chat interface for the DeepSeek Reasoner model that separates reasoning from content.
    This implementation maintains the exact format of the DeepSeek API responses and handles
    the streaming of both reasoning and content separately.
    """
    def __init__(self):
        """Initialize the chat interface using the API key from Streamlit secrets."""
        self.client = OpenAI(
            api_key=st.secrets["deepseek"]["api_key"],
            base_url="https://api.deepseek.com"
        )
        
    def generate_response(self, messages: list) -> Tuple[str, str]:
        """
        Generate a response using the DeepSeek Reasoner model.
        
        Args:
            messages: List of conversation messages in the format 
                     [{"role": "user", "content": "message"}, ...]
        
        Returns:
            Tuple of (reasoning_content, content)
        """
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )
        
        reasoning_content = ""
        content = ""
        
        for chunk in response:
            # Extract reasoning content if present
            if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                if chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
            # Extract regular content
            if hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                    
        return reasoning_content, content

def initialize_session_state():
    """Initialize the session state variables for the chat history."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_reasoning' not in st.session_state:
        st.session_state.show_reasoning = False

def main():
    st.set_page_config(
        page_title="DeepSeek Reasoner Chat",
        page_icon=":brain:"
    )
    
    st.title("DeepSeek Reasoner Chat :brain:")
    st.write("An AI assistant that can show its reasoning process")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.session_state.show_reasoning = st.checkbox(
            "Show AI Reasoning Process",
            value=st.session_state.show_reasoning
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "reasoning" in message and st.session_state.show_reasoning:
                st.markdown("**Reasoning Process:**")
                st.markdown(message["reasoning"])
                st.divider()
                st.markdown("**Response:**")
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare messages for API
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ["user", "assistant"]
        ]
        api_messages.append({"role": "user", "content": prompt})
        
        # Add user message to session state
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                reasoner = DeepSeekReasonerChat()
                
                # Create placeholder for streaming response
                message_container = st.container()
                
                with message_container:
                    # Generate response
                    reasoning_content, content = reasoner.generate_response(api_messages)
                    
                    # Display reasoning if enabled
                    if st.session_state.show_reasoning:
                        st.markdown("**Reasoning Process:**")
                        st.markdown(reasoning_content)
                        st.divider()
                        st.markdown("**Response:**")
                    
                    # Display final response
                    st.markdown(content)
                
                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning_content
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
