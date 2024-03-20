from utils.prompts import ice_cream_assistant_prompt, api_response_prompt, api_url_prompt
from api_docs import scoopsie_api_docs

import streamlit as st
from openai import OpenAI
client = OpenAI()

st.title("💬 Store Spark")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-0125"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model = st.session_state["openai_model"],
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream = True,)
        response = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})