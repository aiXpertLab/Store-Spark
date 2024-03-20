import streamlit as st
import asyncio

from langchain_openai import OpenAI
from langchain.chains import LLMChain, APIChain
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv

from utils.prompts import ice_cream_assistant_prompt, api_response_prompt, api_url_prompt
from api_docs import scoopsie_api_docs

load_dotenv()

# In your Streamlit app: Initialize LLM and API chains on app startup
llm_model = "gpt-3.5-turbo-instruct"  # Replace with your desired model
temperature = 0


async def create_llm_chain(llm_model, temperature):
    llm = OpenAI(model=llm_model, temperature=temperature)
    conversation_memory = ConversationBufferMemory(memory_key="chat_history", max_len=200, return_messages=True)
    llm_chain = LLMChain(llm=llm, prompt=ice_cream_assistant_prompt, memory=conversation_memory)
    return llm_chain


async def create_api_chain(llm, api_docs):
    api_chain = APIChain.from_llm_and_api_docs(
        llm=llm,
        api_docs=api_docs,
        api_url_prompt=api_url_prompt,
        api_response_prompt=api_response_prompt,
        limit_to_domains=["http://hypech.com/"]
    )
    return api_chain


async def main():

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    st.title("💬 Chatbot")
    st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
    
    chat_history = []
    chat_container = st.container()  # Create a scrollable container

    llm_chain = await create_llm_chain(llm_model, temperature)
    api_chain = await create_api_chain(llm_chain.llm, scoopsie_api_docs)

    st.session_state["llm_chain"] = llm_chain
    st.session_state["api_chain"] = api_chain

    # Create a text input for user messages:
    # user_message = st.chat_input("Ask me anything about ice cream!")
    user_message = st.chat_input("Ask me anything about ice cream!", key="user_message") or ""

    # Clear container content (optional, for fresh conversation on refresh)
    # chat_container.empty()

    # Append user message to chat history
    chat_history.append({"role": "user", "content": user_message})
    
    if any(keyword in user_message.lower() for keyword in ["menu", "customization", "offer", "review"]):
        response = await api_chain.ainvoke(user_message)
        response_key = "output" if "output" in response else "text"
    else:
        response = await llm_chain.ainvoke(user_message)
        response_key = "output" if "output" in response else "text"
    
    chat_history.append({"role": "assistant", "content": response.get(response_key, "")})

    with chat_container:
            for msg in chat_history:
                st.chat_message(msg["role"]).write(msg["content"])    
                
    # for msg in chat_history:
    #     st.chat_message(msg["role"]).write(msg["content"])

    # st.write(response.get(response_key, ""))

if __name__ == "__main__":
    asyncio.run(main())
