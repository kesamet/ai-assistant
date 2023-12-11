import requests
from typing import Any, Union

import streamlit as st
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from src import CFG
from src.llama2chat import llama2_prompt
from streamlit_app import get_http_status

CHAT_MODELS = ["googlepalm", "gpt-4-0613", "llama-2"]


class Llama2:
    def __init__(self, model_name) -> None:
        if model_name.startswith("llama-2"):
            self._api_url = f"http://{CFG.HOST}:{CFG.PORT_LLAMA2CHAT}"
            get_http_status(self._api_url)
        else:
            raise NotImplementedError

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"inputs": llama2_prompt(messages)}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self._api_url, headers=headers, json=payload)
            return response.json()
        except Exception:
            return "Llama2 is not deployed"


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "ch_messages" not in st.session_state:
        st.session_state.ch_messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format."
            )
        ]


def select_llm() -> Union[ChatGooglePalm, ChatOpenAI, Llama2]:
    model_name = st.sidebar.radio("Select chat model", CHAT_MODELS)
    if model_name == "googlepalm":
        return ChatGooglePalm(temperature=CFG.TEMPERATURE)
    elif model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=CFG.TEMPERATURE, model_name=model_name)
    return Llama2(model_name)


def get_answer(llm, messages) -> str:
    if isinstance(llm, ChatGooglePalm):
        try:
            answer = llm(messages)
            return answer.content
        except Exception:
            return "GooglePalm is not available. Did you provide an API key?"

    if isinstance(llm, ChatOpenAI):
        try:
            # with get_openai_callback() as cb:
            answer = llm(messages)
            return answer.content
        except Exception:
            return "ChatGPT is not available. Did you provide an API key?"

    if isinstance(llm, Llama2):
        try:
            answer = llm(messages)
            return answer["content"]
        except Exception:
            return "Llama2 is not available. Did you deploy the model?"


def chatbot():
    st.sidebar.title("Chatbot")
    st.sidebar.info("A chatbot demo with different chat models.")

    llm = select_llm()
    init_messages()

    # Display chat history
    for message in st.session_state.ch_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.ch_messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer = get_answer(llm, st.session_state.ch_messages)
            st.markdown(answer)

        st.session_state.ch_messages.append(AIMessage(content=answer))
