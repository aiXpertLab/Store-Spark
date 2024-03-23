import requests
import streamlit as st
from streamlit_pills import pills
from streamlit_extras.app_logo import add_logo
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI
GPT_MODEL = "gpt-3.5-turbo-0125"

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def get_products():
    url = "https://hypech.com/StoreSpark/product_short.json" 
    # url = "https://hypech.com/StoreSpark/products.json"    
    response = requests.get(url)     
    if response.status_code == 200:  
        data = response.text                  
        return data
    else:
        print(f"The store is closed：{response.status_code}")    

st.title("👋 How can I help you today? 💬")
st.caption("🚀 Bridge the Gap: Chatbots for Every Store 🍨")
# st.sidebar.title("Store Spark")
st.sidebar.image("sslogo.png", use_column_width=True)

with st.sidebar:
    store_link = st.text_input("Enter Your Store URL:",   value="http://hypech.com/StoreSpark", disabled=True, key="store_link")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-0125"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system",      "content": "The product list in JSON format is available. You are an store assistant, help user to pick product and provide the pre-sale service."})
    st.session_state.messages.append({"role": "assistant",   "content": get_products()})

# Display chat messages from history on app rerun
for message in st.session_state.messages[2:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Looking for tees, drinkware, headgear, bag, accessories, or office supplies?🍦Ask me!"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    else: 
        try:
            client = OpenAI(api_key = openai_api_key)
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
        except:
            st.info("Invalid OpenAI API key. Please enter a valid key to proceed.")