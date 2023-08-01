import os
from typing import Union

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.llms import CTransformers
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from src.llama2 import build_llama2, llama2_prompt

st.set_page_config(page_title="Personal Assistant")

_ = load_dotenv("../code-conversion/.env")
os.environ["GOOGLE_API_KEY"] = os.environ["PALM_API_KEY"]


@st.cache_resource
def _load_llama2():
    return build_llama2()


LLM_LLAMA2 = _load_llama2()


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format."
            )
        ]
        st.session_state.costs = []


def select_llm() -> Union[ChatGooglePalm, ChatOpenAI, CTransformers]:
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
        return LLM_LLAMA2


def get_response(llm, messages) -> tuple[str, float]:
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

    if isinstance(llm, CTransformers):
        return llm(llama2_prompt(messages)), 0.0


def main():
    st.sidebar.title("Personal Assistant")

    llm = select_llm()

    st.sidebar.write("---")
    init_messages()

    if user_input := st.chat_input("Your input"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("AI is responding ..."):
            answer, cost = get_response(llm, st.session_state.messages)
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
