import base64
import requests

import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from PIL import Image

from src import CFG
from streamlit_app import get_http_status

API_URL = f"http://{CFG.HOST}:{CFG.PORT_LLAVA}"

# sliding window of the most recent interactions
MEMORY_BUFFER_WINDOW = 6


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="vision_assistant")
    if clear_button or "llava_messages" not in st.session_state:
        init_sess_state()


def init_sess_state() -> None:
    # llava_messages used in model
    st.session_state.llava_messages = [
        {
            "role": "system",
            "content": "You are an assistant who perfectly describes images.",
        }
    ]
    # chv_messages used for displaying
    st.session_state.chv_messages = []


def buffer_window_memory(messages: list) -> list:
    """
    Sliding window of the most recent interactions
    older interactions will not be sent to LLM except for
    system and first user and assistant interaction to retain context
    """
    return messages[:3] + messages[3:][-MEMORY_BUFFER_WINDOW:]


def get_output(messages: list) -> str:
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, json={"inputs": messages})
    return response.json()["choices"][0]["message"]


def vision_assistant():
    st.sidebar.title("Vision Assistant")
    st.sidebar.info(
        "Vision Assistant is powered by [LLaVA](https://llava-vl.github.io/)."
    )
    st.sidebar.info(f"Running on {CFG.DEVICE}")
    get_http_status(API_URL)

    uploaded_file = st.sidebar.file_uploader(
        "Upload your image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    init_messages()
    if uploaded_file is None:
        st.info("Upload an image first.")
        init_sess_state()
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_input},
                ],
            }
        else:
            message = {"role": "user", "content": user_input}

        st.session_state.llava_messages.append(message)
        st.session_state.llava_messages = buffer_window_memory(
            st.session_state.llava_messages
        )
        st.session_state.chv_messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                message = get_output(st.session_state.llava_messages)
            st.markdown(message["content"])

        st.session_state.llava_messages.append(message)
        st.session_state.chv_messages.append(AIMessage(content=message["content"]))
