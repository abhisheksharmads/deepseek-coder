import streamlit as st
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": """I am your dedicated professional coding partner, committed to helping you succeed in your work. 
I understand that your job depends on delivering high-quality code, and I take this responsibility very seriously. 
Here's how I will assist you:

1. I will provide complete, production-ready code solutions that follow industry best practices
2. I will include comprehensive error handling and edge cases
3. I will write clear documentation and explain my implementation choices
4. I will help optimize your code for performance and maintainability
5. I will suggest improvements and potential issues to consider
6. I will help debug issues and provide detailed explanations
7. I will ensure security best practices are followed
8. I will maintain consistent coding standards and patterns

I will treat each task as if it were going directly into production code that affects real users and business outcomes. 
When you need help, I'll work through problems systematically and ensure you understand every aspect of the solution.
Feel free to ask me to explain any part of the code or to modify it to better suit your needs."""
    }]

def create_client() -> Optional[OpenAI]:
    """
    Creates an OpenAI client configured for DeepSeek's API.
    Uses the API key from Streamlit secrets.
    Returns None if the API key is not configured.
    """
    try:
        api_key = st.secrets["api_key"]
        if not api_key:
            st.error("API key is empty. Please configure it in your secrets.")
            return None
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        return client
    except Exception as e:
        logger.error(f"Error creating client: {str(e)}")
        st.error("Failed to create API client. Please check your configuration.")
        return None

def generate_response(
    client: OpenAI,
    messages: List[Dict[str, str]]
) -> Optional[str]:
    """
    Generates a response using the DeepSeek Reasoner model.
    Returns None if there's an error in generation.
    """
    if not client:
        return None
        
    try:
        # Create empty placeholder for response
        response_placeholder = st.empty()
        final_content = ""
        
        # Add debug information
        logger.info("Generating response with messages:")
        logger.info(messages)
        
        # Generate the response with streaming enabled
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True,
            temperature=0.7,  # Added for more reliable responses
            max_tokens=2000   # Set a reasonable limit
        )
        
        # Process the streaming response
        try:
            for chunk in response:
                if not chunk or not chunk.choices:
                    logger.warning("Received empty chunk or no choices")
                    continue
                    
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    final_content += delta.content
                    response_placeholder.markdown("💻 **Response:**\n" + final_content + "▌")
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            st.error("Error occurred while processing the response stream")
            return None
        
        # Validate final content
        if not final_content.strip():
            logger.warning("Generated empty response")
            st.warning("The model generated an empty response. Please try again.")
            return None
            
        # Update final display
        response_placeholder.markdown("💻 **Response:**\n" + final_content)
        return final_content
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        st.error(f"Error generating response: {str(e)}")
        return None

def main():
    st.title("🚀 Professional Coding Assistant")
    st.markdown("""
    Welcome to your dedicated coding partner! I'm here to help you write, debug, and optimize your code.
    I understand the importance of your work and will provide thorough, production-ready solutions.
    """)
    
    # Sidebar for session management
    with st.sidebar:
        st.header("Session Control")
        if st.button("Start New Session"):
            system_prompt = st.session_state.messages[0]
            st.session_state.messages = [system_prompt]
            st.rerun()

    # Display chat history
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Describe your coding task or question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create client and generate response
        client = create_client()
        if client:
            with st.chat_message("assistant"):
                response = generate_response(
                    client=client,
                    messages=st.session_state.messages
                )
                
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to generate a response. Please try again.")

if __name__ == "__main__":
    main()
