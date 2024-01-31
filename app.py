import streamlit as st

from streamlit_app.ai_agents import ai_agents
from streamlit_app.chatbot import chatbot
from streamlit_app.code_assistant import code_assistant
from streamlit_app.vision_assistant import vision_assistant
from streamlit_app.financial_assistant import financial_assistant

st.set_page_config(page_title="AI Assistants")


def main():
    dict_pages = {
        "Vision Assistant": vision_assistant,
        "AI Agents": ai_agents,
        "Code Assistant": code_assistant,
        "Chatbot Playground": chatbot,
        "Financial Assistant": financial_assistant,
    }

    select_page = st.sidebar.radio("Select assistant", list(dict_pages.keys()))
    st.sidebar.write("---")

    dict_pages[select_page]()


if __name__ == "__main__":
    main()
