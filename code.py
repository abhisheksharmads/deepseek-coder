import streamlit as st
from openai import OpenAI
from typing import Dict, List, Optional, Tuple

# Initialize session state for chat history
if "messages" not in st.session_state:
    # Initialize with our specialized system prompt
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
    
if "reasoning_history" not in st.session_state:
    st.session_state.reasoning_history = []

def create_client(api_key: str) -> OpenAI:
    """
    Creates an OpenAI client configured for DeepSeek's API.
    The client is specifically set up to work with the DeepSeek Reasoner model.
    """
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def generate_response(
    client: OpenAI,
    messages: List[Dict[str, str]]
) -> Tuple[str, str]:
    """
    Generates a response using the DeepSeek Reasoner model.
    Returns both the reasoning process and the final content.
    
    Args:
        client: The configured OpenAI client
        messages: List of conversation messages
        
    Returns:
        Tuple containing (reasoning_content, final_content)
    """
    try:
        # Create empty placeholders for streaming content
        reasoning_placeholder = st.empty()
        response_placeholder = st.empty()
        
        # Initialize content collectors
        reasoning_content = ""
        final_content = ""
        
        # Generate the response with streaming enabled
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )
        
        # Process the streaming response
        for chunk in response:
            # Handle reasoning content - This shows the model's thought process
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
                reasoning_placeholder.markdown("🤔 **Problem-Solving Process:**\n" + reasoning_content + "▌")
            
            # Handle final content - This contains the actual code and explanation
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                final_content += chunk.choices[0].delta.content
                response_placeholder.markdown("💻 **Solution:**\n" + final_content + "▌")
        
        # Update final display with complete content
        if reasoning_content:
            reasoning_placeholder.markdown("🤔 **Problem-Solving Process:**\n" + reasoning_content)
        if final_content:
            response_placeholder.markdown("💻 **Solution:**\n" + final_content)
            
        return reasoning_content, final_content
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "", ""

def main():
    st.title("🚀 Professional Coding Assistant")
    st.markdown("""
    Welcome to your dedicated coding partner! I'm here to help you write, debug, and optimize your code.
    I understand the importance of your work and will provide thorough, production-ready solutions.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="Enter your DeepSeek API key to begin"
        )
        
        if st.button("Start New Session"):
            # Preserve the system prompt while clearing the conversation
            system_prompt = st.session_state.messages[0]
            st.session_state.messages = [system_prompt]
            st.session_state.reasoning_history = []
            st.rerun()

    # Display chat history with reasoning
    for idx, (message, reasoning) in enumerate(zip(
        st.session_state.messages[1:],  # Skip system prompt in display
        st.session_state.reasoning_history + [None] * (len(st.session_state.messages) - 1 - len(st.session_state.reasoning_history))
    )):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if reasoning and message["role"] == "assistant":
                with st.expander("View Problem-Solving Process"):
                    st.markdown(reasoning)

    # Chat input
    if prompt := st.chat_input("Describe your coding task or question..."):
        if not api_key:
            st.error("Please enter your DeepSeek API key in the sidebar to begin.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create client and generate response
        client = create_client(api_key)
        
        # Display assistant response
        with st.chat_message("assistant"):
            reasoning, response = generate_response(
                client=client,
                messages=st.session_state.messages
            )
            
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.reasoning_history.append(reasoning)

if __name__ == "__main__":
    main()
