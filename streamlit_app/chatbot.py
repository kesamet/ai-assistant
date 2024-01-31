import requests
from typing import Any

import streamlit as st
from langchain.schema import ChatMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src import CFG

CHAT_MODELS = ["gemini-pro", "gpt-4-0613", "llama-2", "mistral", "llamacpp"]


class GeminiPro:
    """Wrapper of ChatGoogleGenerativeAI."""

    def __init__(self, model_name: str = "gemini-pro") -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=CFG.LLM_CONFIG.TEMPERATURE
        )

    def __call__(self, messages: list[ChatMessage], *args: Any, **kwds: Any) -> dict:
        # Converts messages to Human or AI messages first before calling the API
        _messages = list()
        for message in messages:
            if message.role == "user":
                _messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                _messages.append(AIMessage(content=message.content))
        return self.llm.invoke(_messages)

    def __str__(self):
        return self.model_name


class LocalChat:
    """Wrapper of API generated by server.py."""

    def __init__(self, model_name: str, api_url: str) -> None:
        self.model_name = model_name
        self.api_url = api_url

    def __call__(self, messages: list[ChatMessage], *args: Any, **kwds: Any) -> dict:
        response = requests.post(
            self.api_url + "/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"messages": self._convert_langchainschema_to_dict(messages)},
        )
        return response.json()["choices"][0]["message"]

    def __str__(self):
        return self.model_name

    @staticmethod
    def _convert_langchainschema_to_dict(messages: list[ChatMessage]) -> list[dict]:
        """Converts list of chat messages in langchain.schema format to list of dict."""
        return [
            {"role": message.role, "content": message.content} for message in messages
        ]


class LocalChatOpenAI:
    """
    Wrapper of API generated by llama_cpp.server or ollama

    ```
    python -m llama_cpp.server --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --chat_format chatml
    ```
    """

    def __init__(self, openai_api_base: str, **kwargs) -> None:
        self.openai_api_base = openai_api_base
        self.llm = ChatOpenAI(
            openai_api_base=openai_api_base,
            openai_api_key="sk-xxx",
            temperature=CFG.LLM_CONFIG.TEMPERATURE,
            streaming=True,
            **kwargs,
        )

    def __call__(self, messages: list[ChatMessage], *args: Any, **kwds: Any) -> dict:
        return self.llm.invoke(messages)

    def __str__(self):
        return self.openai_api_base


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "ch_messages" not in st.session_state:
        st.session_state.ch_messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful, respectful and honest assistant. "
                    "Always answer as helpfully as possible, while being safe."
                ),
            )
        ]


def select_llm():
    model_name = st.sidebar.radio("Select chat model", CHAT_MODELS)
    if model_name.startswith("gemini"):
        return GeminiPro(model_name)
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=CFG.LLM_CONFIG.TEMPERATURE, model_name=model_name)
    if model_name == "llama-2":
        return LocalChat(model_name, f"http://{CFG.HOST}:{CFG.PORT_LLAMA2}")
    if model_name == "mistral":
        return LocalChat(model_name, f"http://{CFG.HOST}:{CFG.PORT_MISTRAL}")
    if model_name == "llamacpp":
        return LocalChatOpenAI("http://localhost:8000/v1")
    raise NotImplementedError


def get_answer(llm, messages) -> str:
    try:
        answer = llm(messages)
        if isinstance(answer, dict):
            return answer["content"]
        return answer.content
    except Exception as e:
        st.error(e)
        return ""


def chatbot():
    st.sidebar.title("Chatbot playground")
    st.sidebar.info("A chatbot playground to test different chat models.")

    llm = select_llm()
    init_messages()

    # Display chat history
    for message in st.session_state.ch_messages:
        if message.role == "system":
            continue
        with st.chat_message(message.role):
            st.markdown(message.content)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.ch_messages.append(
            ChatMessage(role="user", content=user_input)
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer = get_answer(llm, st.session_state.ch_messages)
            st.markdown(answer)

        st.session_state.ch_messages.append(
            ChatMessage(role="assistant", content=answer)
        )
