"""
Adapted from https://github.com/langchain-ai/langchain/blob/master/templates/gemini-functions-agent/gemini_functions_agent/agent.py
"""

from typing import List, Tuple

import streamlit as st
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents import create_gemini_functions_agent
from src.general import setup_tracing
from src.tools import (
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
)

# Define agent
tools = [
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
]

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_gemini_functions_agent(llm=llm, tools=tools, prompt=prompt)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)


def _setup_tracing():
    success, e = setup_tracing()
    if not success:
        st.sidebar.error(f"Tracing with Phoenix app not available: {e}")
        st.sidebar.info("To start the app, run `python3 -m phoenix.server.main serve`")
    else:
        st.sidebar.info("View traces with [Phoenix app](http://localhost:6006/)")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Chat", key="gf_agent")
    if clear_button or "gf_messages" not in st.session_state:
        st.session_state.gf_messages = []


def get_response(user_input: str, chat_history: List[Tuple[str, str]]) -> str:
    try:
        return agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history}
        )["output"]
    except Exception as e:
        st.error(e)


def agent_gemini_functions():
    st.sidebar.title("Gemini Functions Agent")
    _setup_tracing()

    init_messages()

    # Display chat history
    for human, ai in st.session_state.gf_messages:
        with st.chat_message("user"):
            st.markdown(human)
        with st.chat_message("assistant"):
            st.markdown(ai)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                response = get_response(user_input, st.session_state.gf_messages)
            st.markdown(response)

        st.session_state.gf_messages.append((user_input, response))
