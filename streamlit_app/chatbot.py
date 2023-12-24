import requests
from typing import Any

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src import CFG
from src.llama2 import llama2_prompt
from streamlit_app import get_http_status

CHAT_MODELS = ["gemini-pro", "gpt-4-0613", "llama-2", "mistral"]


class GeminiPro:
    """Wrapper of ChatGoogleGenerativeAI."""

    def __init__(self, model_name="gemini-pro") -> None:
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=CFG.TEMPERATURE)

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        """Converts messages to Human or AI (user/assistant) messages supported by Gemini first
        before calling the API
        """
        import langchain_core.messages as gm

        _messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                _messages.append(gm.HumanMessage(content=message.content))
            elif isinstance(message, AIMessage):
                _messages.append(gm.AIMessage(content=message.content))
        return self.llm.invoke(_messages)


class Llama2:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self._api_url = f"http://{CFG.HOST}:{CFG.PORT_LLAMA2}"
        get_http_status(self._api_url)

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"inputs": llama2_prompt(messages)}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self._api_url, headers=headers, json=payload)
            return response.json()
        except Exception:
            return "Llama2 is not deployed"

    def __str__(self):
        return self.model_name


class Mistral:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self._api_url = f"http://{CFG.HOST}:{CFG.PORT_MISTRAL}"
        get_http_status(self._api_url)

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"inputs": llama2_prompt(messages)}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self._api_url, headers=headers, json=payload)
            return response.json()
        except Exception:
            return "Mistral is not deployed"

    def __str__(self):
        return self.model_name


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "ch_messages" not in st.session_state:
        st.session_state.ch_messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format."
            )
        ]


def select_llm():
    model_name = st.sidebar.radio("Select chat model", CHAT_MODELS)
    if model_name.startswith("gemini"):
        return GeminiPro(model_name)
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=CFG.TEMPERATURE, model_name=model_name)
    if model_name.startswith("llama"):
        return Llama2(model_name)
    if model_name.startswith("mistral"):
        return Mistral(model_name)
    raise NotImplementedError


def get_answer(llm, messages) -> str:
    try:
        answer = llm(messages)
        if isinstance(answer, dict):
            return answer["content"]
        return answer.content
    except Exception:
        st.error(
            f"{llm} is not available. Did you provide an API key or deploy the model?"
        )
        return ""


def chatbot():
    st.sidebar.title("Chatbot playground")
    st.sidebar.info("A chatbot playground to test different chat models.")

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
