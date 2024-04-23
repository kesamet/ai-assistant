import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain_google_genai import GoogleGenerativeAI

from src.agents import create_react_agent
from src.general import setup_tracing
from src.tools import (
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
)

# Define agent
model = GoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.2,
    max_tokens=512,
)

tools = [
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
]

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, return_intermediate_steps=True, verbose=True
)


def _setup_tracing():
    success, e = setup_tracing()
    if not success:
        st.sidebar.error(f"Tracing with Phoenix app not available: {e}")
        st.sidebar.info("To start the app, run `python3 -m phoenix.server.main serve`")
    else:
        st.sidebar.info("View traces with [Phoenix app](http://localhost:6006/)")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Chat", key="react_agent")
    if clear_button or "react_messages" not in st.session_state:
        st.session_state.react_messages = []


def get_response(user_input: str) -> str:
    try:
        return agent_executor.invoke({"input": user_input})
    except Exception as e:
        st.error(e)


def agent_react():
    st.sidebar.title("ReAct Agent")
    st.sidebar.info(
        "ReAct Agent is powered by gemini-pro and has access to Tavily, Wikipedia, "
        "News API, Wolfram and calculator tools."
    )
    _setup_tracing()

    init_messages()

    # Display chat history
    for human, ai in st.session_state.react_messages:
        with st.chat_message("user"):
            st.markdown(human)
        with st.chat_message("assistant"):
            st.markdown(ai)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                response = get_response(user_input)
            st.markdown(response["output"])
            st.markdown("**Sources**")
            for action, content in response["intermediate_steps"]:
                st.markdown(f"`{action.log}`")
                st.markdown(content)
                st.markdown("---")

        st.session_state.react_messages.append((user_input, response["output"]))
