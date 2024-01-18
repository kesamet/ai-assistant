import requests
from typing import Any, List, Union

import streamlit as st
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src import CFG
from src.prompt_format import Llama2Format, MistralFormat
from streamlit_app import get_http_status

CHAT_MODELS = ["gemini-pro", "gpt-4-0613", "llama-2", "mistral", "llamacpp"]

SYSTEM_PROMPT = "You are a helpful AI assistant. Reply your answer in markdown format."


class GeminiPro:
    """Wrapper of ChatGoogleGenerativeAI."""

    def __init__(self, model_name: str = "gemini-pro") -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=CFG.LLM_CONFIG.TEMPERATURE
        )

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        """Converts messages to Human or AI (user/assistant) messages supported by Gemini first
        before calling the API
        """
        import langchain_core.messages as gm

        _messages = []
        for message in messages:
            if message["role"] == "user":
                _messages.append(gm.HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                _messages.append(gm.AIMessage(content=message["content"]))
        return self.llm.invoke(_messages)

    def __str__(self):
        return self.model_name


class Llama2:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self._api_url = f"http://{CFG.HOST}:{CFG.PORT_LLAMA2}"
        get_http_status(self._api_url)
        self.prompt_format = Llama2Format()

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"messages": messages}
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._api_url + "/v1/chat/completions", headers=headers, json=payload
        )
        return response.json()["choices"][0]["message"]

    def __str__(self):
        return self.model_name


class Mistral:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self._api_url = f"http://{CFG.HOST}:{CFG.PORT_MISTRAL}"
        get_http_status(self._api_url)
        self.prompt_format = MistralFormat()

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"messages": messages}
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._api_url + "/v1/chat/completions", headers=headers, json=payload
        )
        return response.json()["choices"][0]["message"]

    def __str__(self):
        return self.model_name


class LocalChatOpenAI:
    def __init__(
        self,
        openai_api_base: str = "http://localhost:8000/v1",
        temperature: float = 0.2,
        **kwargs,
    ) -> None:
        self.openai_api_base = openai_api_base
        self.llm = ChatOpenAI(
            openai_api_base=openai_api_base,
            openai_api_key="sk-xxx",
            temperature=temperature,
            streaming=True,
            **kwargs,
        )

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        return self.llm.invoke(messages)

    def __str__(self):
        return self.openai_api_base


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "ch_messages" not in st.session_state:
        st.session_state.ch_messages = [{"content": SYSTEM_PROMPT, "role": "system"}]


def select_llm():
    model_name = st.sidebar.radio("Select chat model", CHAT_MODELS)
    if model_name.startswith("gemini"):
        return GeminiPro(model_name)
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=CFG.LLM_CONFIG.TEMPERATURE, model_name=model_name)
    if model_name == "llama-2":
        return Llama2(model_name)
    if model_name == "mistral":
        return Mistral(model_name)
    if model_name == "llamacpp":
        return LocalChatOpenAI()
    raise NotImplementedError


def _convert_langchainschema_to_dict(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
) -> List[dict]:
    """Converts list of chat messages in langchain.schema format to list of dict."""
    _messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            _messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            _messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            _messages.append({"role": "assistant", "content": message.content})
    return _messages


def get_answer(llm, messages) -> str:
    try:
        answer = llm(_convert_langchainschema_to_dict(messages))
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
