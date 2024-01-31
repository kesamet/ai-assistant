import logging
import os
from typing import List

import fitz
import streamlit as st
from dotenv import load_dotenv
from glob import glob
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from streamlit_app.utils import get_pdf_display

logging.basicConfig(level=logging.INFO)

_ = load_dotenv()

LLM = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

METRICS = [
    "net property income",
    "distribution per unit (DPU)",
    "investment properties",
    "total assets",
    "total liabilities",
    "total debts",
    "total number of units",
    "net asset value (NAV)",
    "aggregate leverage",
    "cost of debt",
    "interest cover",
    "average term to maturity / weighted average tenor of debt",
    "weighted average lease expiry (WALE)",
]


@st.cache_data
def _get_content_db(filename):
    loader = PyMuPDFLoader(filename)
    pages = loader.load_and_split()
    return FAISS.from_documents(pages, EMBEDDINGS)


@st.cache_data
def _get_tables_db(filename):
    table_docs = extract_tables(filename)
    return FAISS.from_documents(table_docs, EMBEDDINGS)


def extract_tables(filename: str) -> List[Document]:
    """Extract tables from PDF."""
    pdf_file = fitz.open(filename)
    table_docs = list()
    for page in pdf_file:
        tabs = page.find_tables()
        logging.info(f"[+] Found {len(tabs.tables)} table(s) on page {page.number}")

        for tab in tabs:
            try:
                df = tab.to_pandas()
                if df.shape == (1, 1):
                    logging.info("  [!] dataframe shape is (1, 1)")
                    continue
                d = Document(
                    page_content=df.to_json(),
                    metadata={"source": filename, "page": page.number},
                )
                table_docs.append(d)
            except Exception:
                logging.info("  [!] unable to convert to dataframe")
    return table_docs


@st.cache_data
def extract_metrics(filename):
    content_db = _get_content_db(filename)
    content_retriever = content_db.as_retriever()

    content_template = """Based only on the following context, answer the question in a sentence:
Context: {context}
Question: {question}"""
    content_chain = (
        {"context": content_retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(content_template)
        | LLM
        | StrOutputParser()
    )

    tables_db = _get_tables_db(filename)
    tables_retriever = tables_db.as_retriever()

    table_prompt = """Using the following tables, answer the question in a sentence:
Tables: {context}
Question: {question}"""
    table_chain = (
        {"context": tables_retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(table_prompt)
        | LLM
        | StrOutputParser()
    )

    summarise_prompt = """From the answers below, summarise the final answer \
that contains information in a sentence:
Answer 1: {content_answer}
Answer 2: {table_answer}
Final answer:"""

    chain = (
        {"content_answer": content_chain, "table_answer": table_chain}
        | ChatPromptTemplate.from_template(summarise_prompt)
        | LLM
        | StrOutputParser()
    )

    results = list()
    for metric in METRICS:
        res = chain.invoke(f"What is the portfolio {metric}?")
        st.info(res)
        results.append(res)
    return results


def financial_assistant():
    st.title("Financial metrics")

    c0, c1 = st.columns(2)
    c0.info("This app extract predefined metrics from financial reports.")

    report_dir = st.sidebar.text_input("Input report directory")
    if report_dir == "":
        st.stop()

    filenames = sorted(glob(os.path.join(report_dir, "*.pdf")))
    with c0.form("metrics"):
        filename = st.selectbox("Select file", filenames)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.fc_messages = extract_metrics(filename)

    with c1:
        st.markdown(
            get_pdf_display(open(filename, "rb").read()),
            unsafe_allow_html=True,
        )
