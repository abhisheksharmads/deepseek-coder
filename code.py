import streamlit as st
from openai import OpenAI
import json
from typing import Generator

class ChatBot:
    def __init__(self, api_key: str):
        """Initialize the chatbot with the DeepSeek API key."""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.messages = []
        
    def get_system_prompt(self) -> str:
        """Return the system prompt for the coding assistant."""
        return """You are an expert coding assistant. When helping with code:
        1. Provide clear, well-documented solutions
        2. Explain your reasoning when asked
        3. Follow best practices and coding standards
        4. Include error handling where appropriate
        Please keep responses focused and practical."""
    
    def generate_response(self, user_input: str, show_reasoning: bool) -> Generator:
        """Generate a response from the model with optional reasoning."""
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        
        # If reasoning is requested, modify the prompt
        if show_reasoning:
            user_input = f"Please show your reasoning step by step, then provide the solution:\n{user_input}"
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    *self.messages
                ],
                stream=True
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("🤖 Coding Assistant")
    st.write("Your AI companion for coding questions and solutions")
    
    initialize_session_state()
    
    # Get API key from secrets
    api_key = st.secrets["deepseek"]["api_key"]
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Configuration")
        show_reasoning = st.checkbox("Show AI Reasoning Process", value=False)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your coding question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                chatbot = ChatBot(api_key)
                response = chatbot.generate_response(prompt, show_reasoning)
                
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
