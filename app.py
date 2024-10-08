import streamlit as st

from streamlit_app.agent_react import agent_react
from streamlit_app.chatbot import chatbot
from streamlit_app.code_assistant import code_assistant
from streamlit_app.vision_assistant import vision_assistant

# from phoenix.trace.langchain import LangChainInstrumentor

# Setup tracing
# LangChainInstrumentor().instrument()

st.set_page_config(page_title="AI Assistants")


def main():
    dict_pages = {
        "ReAct Agent": agent_react,
        "Chatbot Playground": chatbot,
        "Vision Assistant": vision_assistant,
        "Code Assistant": code_assistant,
    }

    select_page = st.sidebar.radio("Select assistant", list(dict_pages.keys()))
    st.sidebar.write("---")

    dict_pages[select_page]()


if __name__ == "__main__":
    main()
