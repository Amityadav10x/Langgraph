import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="LangGraph Chat", layout="centered")
st.title("🤖 Local AI Chatbot")

# FIXED: 'configurable' spelling
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Initialize session state for UI history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display previous chat messages from history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input field
user_input = st.chat_input('Type your message here...')

if user_input:
    # 1. Add User message to UI and history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Get AI response from LangGraph backend
    # We pass the input as a list containing a HumanMessage
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]}, 
        config=CONFIG
    )
    
    # 3. Extract the text content from the last AIMessage object
    ai_message_obj = response['messages'][-1]
    ai_text = ai_message_obj.content

    # 4. Add AI response to history and UI
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_text})
    with st.chat_message("assistant"):
        st.markdown(ai_text)