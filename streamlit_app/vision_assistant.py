import base64
import requests

import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from PIL import Image

from src import CFG


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "llava_messages" not in st.session_state:
        # used in model
        st.session_state.llava_messages = [
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images.",
            }
        ]
        # used for displaying
        st.session_state.chv_messages = []


def get_answer(messages: list) -> str:
    api_url = f"http://{CFG.HOST}:{CFG.PORT_LLAVA}"
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json={"inputs": messages})
    try:
        return response.json()["choices"][0]["message"]
    except Exception as e:
        return f"Llava is probably not deployed: {e}"


def vision_assistant():
    st.sidebar.title("Vision Assistant")

    init_messages()

    uploaded_file = st.sidebar.file_uploader(
        "Upload your image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )
    if uploaded_file is None:
        st.info("Upload an image first.")
        return

    img = Image.open(uploaded_file)
    st.image(img)
    
    img_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

    # Display chat history
    for message in st.session_state.chv_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        if len(st.session_state.llava_messages) == 1:
            message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": user_input}
                ]
            }
        else:
            message = {"role": "user", "content": user_input}

        st.session_state.llava_messages.append(message)
        st.session_state.chv_messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                message = get_answer(st.session_state.llava_messages)
            st.markdown(message["content"])

        st.session_state.llava_messages.append(message)
        st.session_state.chv_messages.append(AIMessage(content=message["content"]))
