import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain_google_genai import GoogleGenerativeAI

from src.agents import create_react_agent
from src.tools import (
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
    get_stock_price_history,
    get_stock_quantstats,
)

tools = [
    tavily_tool,
    wikipedia_tool,
    calculator_tool,
    newsapi_tool,
    wolfram_tool,
    get_stock_price_history,
    get_stock_quantstats,
]

llm = GoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.2,
    max_tokens=512,
)

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
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    max_execution_time=60,
    verbose=True,
)


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Chat", key="react_agent")
    if clear_button or "react_messages" not in st.session_state:
        st.session_state.react_messages = []


def get_response(user_input: str) -> str:
    try:
        return agent_executor.invoke({"input": user_input})
    except Exception as e:
        st.error(e)
        return ""


def agent_react():
    st.sidebar.title("ReAct Agent")
    st.sidebar.info(
        "ReAct Agent is powered by gemini-pro and has access to Tavily, Wikipedia, "
        "News API, Wolfram and calculator tools."
    )

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
            output = response["output"].replace("$", "\$")
            st.markdown(response["output"])

            with st.expander("Thoughts and Actions"):
                for action, content in response["intermediate_steps"]:
                    st.markdown(f"`{action.log}`")
                    st.markdown(content)
                    st.markdown("---")

        st.session_state.react_messages.append((user_input, output))
