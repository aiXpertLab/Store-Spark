import streamlit as st
import asyncio, inspect

from langchain_openai import OpenAI
from langchain.chains import LLMChain, APIChain
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv

from utils.prompts import ice_cream_assistant_prompt, api_response_prompt, api_url_prompt
from api_docs import scoopsie_api_docs

load_dotenv()


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
        verbose=True,
        limit_to_domains=["http://hypech.com/"]
    )
    return api_chain


async def main():
    # In your Streamlit app: Initialize LLM and API chains on app startup
    llm_model = "gpt-3.5-turbo-instruct"  # Replace with your desired model
    temperature = 0

    llm_chain = await create_llm_chain(llm_model, temperature)
    api_chain = await create_api_chain(llm_chain.llm, scoopsie_api_docs)

    st.session_state["llm_chain"] = llm_chain
    st.session_state["api_chain"] = api_chain

    # Create a text input for user messages:
    user_message = st.text_input("Ask me anything about ice cream!")

    if any(keyword in user_message.lower() for keyword in ["menu", "customization", "offer", "review"]):
        response = await api_chain.acall(user_message)
        response_key = "output" if "output" in response else "text"
    else:
        response = await llm_chain.acall(user_message)
        response_key = "output" if "output" in response else "text"

    st.write(response.get(response_key, ""))

if __name__ == "__main__":
    asyncio.run(main())
    # try:
    #     main()
    #     if inspect.iscoroutinefunction(main):
    #         asyncio.run(main())
    #     else:
    #         main()
    # except SystemExit:
    #     print('ha li kal sk;asjdf sja fsd f')