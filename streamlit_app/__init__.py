import streamlit as st
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError


def get_http_status(url):
    try:
        r = requests.get(url + "/docs")
        r.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

    except (ConnectionError, Timeout) as e:
        st.sidebar.error(f"ConnectionError: Model is not deployed - {e}")
    except HTTPError as e:
        st.sidebar.error(e)
    else:
        st.sidebar.info("Endpoint is OK")
