"""
Utility functions for Streamlit app.
"""

import base64
import json
import pickle
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


def uri_encode_path(path: str, mime: str = "image/png") -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def add_header(path: str) -> None:
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )


def get_pdf_display(pdfbytes: bytes) -> str:
    base64_pdf = base64.b64encode(pdfbytes).decode("utf-8")
    return (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        'width="100%" height="970" type="application/pdf"></iframe>'
    )


def download_button(
    data,
    filename: str,
    button_text: str,
    pickle_it: bool = False,
) -> str:
    """
    Generates a link to download the given data.

    Args:
        data: The object to be downloaded.
        filename (str): filename and extension of file to be downloaded
        button_text (str): Text to display on download button
        pickle_it (bool): If True, pickle file.

    Returns
        (str): the anchor tag to download data
    """
    if pickle_it:
        try:
            data = pickle.dumps(data)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(data, bytes):
            pass

        elif isinstance(data, pd.DataFrame):
            data = data.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            data = json.dumps(data)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(data.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(data).decode()

    custom_css, button_id = _custom_button_style()
    return (
        custom_css + f'<a download="{filename}" id="{button_id}" '
        f'href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    )


def logout_button(auth_domain: str) -> str:
    custom_css, button_id = _custom_button_style()
    return (
        custom_css
        + f'<a id="{button_id}" href="https://{auth_domain}/_oauth/logout" '
        'target="_self">Logout</a><br></br>'
    )


def _custom_button_style():
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
            <style>
            #{button_id} {{
                background-color: #FFFFFF;
                color: #262730;
                padding: 0.4em 0.74em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: #DDDDDD;
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: #F63366;
                color: #F63366;
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: #F63366;
                color: white;
            }}
            </style>
        """
    return custom_css, button_id


def adjust_container_width(width: int = 1000) -> None:
    st.markdown(
        (
            "<style>\n"
            ".reportview-container .main .block-container{{\n"
            f"    max-width: {width}px;\n"
            "}}\n"
            "</style>"
        ),
        unsafe_allow_html=True,
    )


def remove_menu() -> None:
    st.markdown(
        "<style>#MainMenu {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def remove_footer() -> None:
    st.markdown(
        "<style>footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def horizontal_radio() -> None:
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )


def colour_text(notes: str, color: str = "red") -> None:
    st.markdown(
        f"<span style='color: {color}'>{notes}</span>",
        unsafe_allow_html=True,
    )


def local_css(filename: str) -> None:
    with open(filename) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )
