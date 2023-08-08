import requests
from typing import Any, Union

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from src import CFG
from src.llama2 import llama2_prompt

st.set_page_config(page_title="Personal Assistant")


class Llama2:
    def __init__(self) -> None:
        self._api_url = f"http://{CFG.HOST}:{CFG.PORT}"

    def __call__(self, messages, *args: Any, **kwds: Any) -> dict:
        payload = {"inputs": llama2_prompt(messages)}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self._api_url, headers=headers, json=payload)
        return response.json()


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format."
            )
        ]
        st.session_state.costs = []


def select_llm() -> Union[ChatGooglePalm, ChatOpenAI, Llama2]:
    model_name = st.sidebar.radio(
        "Choose LLM",
        ["llama-2-7b-chat-ggml", "googlepalm", "gpt-4", "gpt-3.5-turbo-0613"],
    )
    temperature = 0.01
    if model_name == "googlepalm":
        return ChatGooglePalm(temperature=temperature)
    elif model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name)
    elif model_name == "llama-2-7b-chat-ggml":
        return Llama2()


def get_answer(llm, messages) -> tuple[str, float]:
    if isinstance(llm, ChatGooglePalm):
        try:
            answer = llm(messages)
            return answer.content, 0.0
        except Exception:
            return "GooglePalm is not available", 0.0

    if isinstance(llm, ChatOpenAI):
        try:
            with get_openai_callback() as cb:
                answer = llm(messages)
            return answer.content, cb.total_cost
        except Exception:
            return "ChatGPT is not available", 0.0

    if isinstance(llm, Llama2):
        try:
            answer = llm(messages)
            return answer["content"], 0.0
        except Exception:
            return "Llama2 is not available", 0.0


def main():
    st.sidebar.title("Personal Assistant")

    llm = select_llm()

    st.sidebar.write("---")
    init_messages()

    if user_input := st.chat_input("Your input"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("AI is responding ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")


if __name__ == "__main__":
    main()
