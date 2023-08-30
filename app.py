import streamlit as st

from app.ai_assistant import ai_assistant
from app.code_assistant import code_assistant

st.set_page_config(page_title="AI Assistant")


def main():
    dict_pages = {
        "AI Assistant": ai_assistant,
        "Code Assistant": code_assistant,
    }

    select_page = st.radio("Select", list(dict_pages.keys()))
    st.write("---")
    
    dict_pages[select_page]()


if __name__ == "__main__":
    main()
