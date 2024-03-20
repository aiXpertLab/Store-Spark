from openai import OpenAI
import streamlit as st
import asyncio, inspect

from langchain_openai import OpenAI
from langchain.chains import LLMChain, APIChain
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv

from utils.prompts import ice_cream_assistant_prompt, api_response_prompt, api_url_prompt
from api_docs import scoopsie_api_docs


llm_model = "gpt-3.5-turbo-0125"  # Replace with your desired model
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
    st.title("💬 Chatbot")
    st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

    llm_chain = await create_llm_chain(llm_model, temperature)
    api_chain = await create_api_chain(llm_chain.llm, scoopsie_api_docs)

    st.session_state["llm_chain"] = llm_chain
    st.session_state["api_chain"] = api_chain

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
        st.title("💬 Chatbot")
        st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

    # 1. The first code block creates an initial session state to store the LLM generated response as part of the chat message history.
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role"   : "assistant", 
                                        "content": "How can I help you?"}]

    # 2. The next code block displays messages (via st.chat_message()) from the chat history by iterating through the messages variable in the session state.
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        client = OpenAI(api_key=openai_api_key)

        st.session_state.messages.append({"role"   : "user", 
                                        "content": prompt})
        st.chat_message("user").write(prompt)
    else:
        prompt = 'd'
        
    if any(keyword in prompt.lower() for keyword in ["menu", "customization", "offer", "review"]):
        response = await api_chain.ainvoke(prompt)
        print('menu offer review')
    else:
        response = await llm_chain.ainvoke(prompt)
        print('llm')

    response_key = "output" if "output" in response else "text"

    if prompt != 'd':
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})

        st.chat_message("assistant").write(msg)

        st.write(response.get(response_key, ""))


if __name__ == "__main__":
    asyncio.run(main())
