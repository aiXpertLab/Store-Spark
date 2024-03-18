from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain, APIChain

import chainlit as cl

from utils.LangChain_Prompt  import IceCreamPromptCreatorMemory, IceCreamPromptCreatorAPI
from utils.LangChain_Routine import llm

from api_docs import scoopsie_api_docs

# chainlit run spark.py -w --port 8001


@cl.on_chat_start
def setup_multiple_chains():
    llm= LlamaCpp(model_path = "e:/models/llama/llama-2-7b-chat.Q6_K.gguf",n_gpu_layers=40,n_batch=512,  )
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",max_len=250,return_messages=True,)
    assistant_prompt = IceCreamPromptCreatorMemory.create_prompt()
    api_url_prompt, api_response_prompt = IceCreamPromptCreatorAPI.create_prompt()

    llm_chain = LLMChain(llm=llm, prompt=assistant_prompt, memory=conversation_memory)
    cl.user_session.set("llm_chain", llm_chain)

    api_chain = APIChain.from_llm_and_api_docs(
            llm=llm,
            api_docs=scoopsie_api_docs,
            api_url_prompt=api_url_prompt,
            api_response_prompt=api_response_prompt,
            verbose=True,
            limit_to_domains=["http://127.0.0.1:5000"]
        )
    cl.user_session.set("api_chain", api_chain)

@cl.on_message
async def handle_message(message: cl.Message):
    user_message = message.content.lower()
    llm_chain = cl.user_session.get("llm_chain")
    api_chain = cl.user_session.get("api_chain")
    
    if any(keyword in user_message for keyword in ["menu", "customization", "offer", "review"]):
        # If any of the keywords are in the user_message, use api_chain
        print(f"-------------- {user_message}")
        print(f"============== {cl.AsyncLangchainCallbackHandler()}")
        response = await api_chain.acall(user_message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    else:
        # Default to llm_chain for handling general queries
        response = await llm_chain.acall(user_message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    response_key = "output" if "output" in response else "text"
    await cl.Message(response.get(response_key, "")).send()

# chainlit run spark.py -w --port 8001