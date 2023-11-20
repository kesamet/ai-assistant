import streamlit as st

from streamlit_app.ai_assistant import ai_assistant
from streamlit_app.chatbot import chatbot
from streamlit_app.code_assistant import code_assistant

st.set_page_config(page_title="LLM-powered Assistant")


def main():
    dict_pages = {
        "AI Assistant": ai_assistant,
        "Code Assistant": code_assistant,
        "Chatbot": chatbot,
    }

    select_page = st.sidebar.radio("Select assistant", list(dict_pages.keys()))
    st.sidebar.write("---")

    dict_pages[select_page]()


if __name__ == "__main__":
    main()
