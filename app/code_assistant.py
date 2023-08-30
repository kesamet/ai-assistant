import requests

import streamlit as st

from src import CFG
from src.codellama import get_prompt


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "past" not in st.session_state or "generated" not in st.session_state:
        st.session_state.past = []
        st.session_state.generated = []


def get_answer(inputs: str) -> str:
    api_url = f"http://{CFG.HOST}:8001"
    payload = {"inputs": get_prompt(inputs)}
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()["content"]


def code_assistant():
    st.sidebar.title("Code Assistant")
    init_messages()

    if st.session_state.generated:
        for user_msg, assistant_msg in zip(
            st.session_state.past, st.session_state.generated
        ):
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(assistant_msg)
            
    if user_input := st.chat_input("Your input"):
        st.session_state.past.append(user_input)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Responding ..."):
            answer = get_answer(user_input)
        st.session_state.generated.append(answer)
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    code_assistant()
