# Setup tracing
from phoenix.trace.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()

import streamlit as st

from streamlit_app.agent_react import agent_react
from streamlit_app.agent_gemini_functions import agent_gemini_functions
from streamlit_app.chatbot import chatbot
from streamlit_app.code_assistant import code_assistant
from streamlit_app.vision_assistant import vision_assistant

# from streamlit_app.financial_assistant import financial_assistant

st.set_page_config(page_title="AI Assistants")


def main():
    dict_pages = {
        "ReAct Agent": agent_react,
        "Gemini Functions Agent": agent_gemini_functions,
        # "Financial Assistant": financial_assistant,
        "Chatbot Playground": chatbot,
        "Vision Assistant": vision_assistant,
        "Code Assistant": code_assistant,
    }

    select_page = st.sidebar.radio("Select assistant", list(dict_pages.keys()))
    st.sidebar.write("---")

    dict_pages[select_page]()


if __name__ == "__main__":
    main()
