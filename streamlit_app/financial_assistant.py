from typing import List

import fitz
import streamlit as st
from loguru import logger
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.schema import Document, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    GoogleGenerativeAI,
)

LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, max_retries=2)
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
def _get_content_db(pdf_filepath):
    loader = PyMuPDFLoader(pdf_filepath)
    pages = loader.load_and_split()
    return FAISS.from_documents(pages, EMBEDDINGS)


@st.cache_data
def _get_tables_db(pdf_filepath):
    table_docs = extract_tables(pdf_filepath)
    return FAISS.from_documents(table_docs, EMBEDDINGS)


def extract_tables(pdf_filepath: str) -> List[Document]:
    """Extract tables from PDF."""
    pdf_file = fitz.open(pdf_filepath)
    table_docs = list()
    for page in pdf_file:
        tabs = page.find_tables()
        logger.info(f"[+] Found {len(tabs.tables)} table(s) on page {page.number}")

        for tab in tabs:
            try:
                df = tab.to_pandas()
                if df.shape == (1, 1):
                    logger.info("  [!] dataframe shape is (1, 1)")
                    continue
                d = Document(
                    page_content=df.to_json(),
                    metadata={"source": pdf_filepath, "page": page.number},
                )
                table_docs.append(d)
            except Exception:
                logger.info("  [!] unable to convert to dataframe")
    return table_docs


def build_chain(content_retriever, tables_retriever):
    content_template = (
        "Based only on the following context, answer the question in a sentence:\n"
        "Context: {context}\n"
        "Question: {question}"
    )
    content_chain = (
        {"context": content_retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(content_template)
        | LLM
        | StrOutputParser()
    )

    table_template = (
        "Using the following tables, answer the question in a sentence:\n"
        "Tables: {context}\n"
        "Question: {question}"
    )
    table_chain = (
        {"context": tables_retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(table_template)
        | LLM
        | StrOutputParser()
    )

    summarise_template = (
        "From the answers below, summarise the final answer "
        "that contains information in a sentence:\n"
        "Answer 1: {content_answer}\n"
        "Answer 2: {table_answer}\n"
        "Final answer:"
    )
    chain = (
        {"content_answer": content_chain, "table_answer": table_chain}
        | PromptTemplate.from_template(summarise_template)
        | LLM
        | StrOutputParser()
    )
    return chain


def extract_metrics(chain):
    results = list()
    for metric in METRICS:
        res = chain.invoke(f"What is the portfolio {metric}?")
        st.info(res)
        results.append(res)
    return results


@st.cache_data
def _get_metrics(_chain):
    return extract_metrics(_chain)


def build_retrieval_chain(retriever):
    condense_question_template = """Given the following conversation and a follow up question, \
rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        condense_question_prompt=PromptTemplate.from_template(condense_question_template),
    )
    return retrieval_chain


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="chatbot")
    if clear_button or "fc_messages" not in st.session_state:
        st.session_state.fc_messages = []


def financial_assistant():
    st.title("Financial Assistant")
    # st.info("This app extracts predefined metrics from financial reports.")

    pdf_filepath = st.text_input("Input PDF filepath")
    if pdf_filepath == "":
        st.stop()

    content_db = _get_content_db(pdf_filepath)
    content_retriever = content_db.as_retriever()
    tables_db = _get_tables_db(pdf_filepath)
    tables_retriever = tables_db.as_retriever()
    metrics_chain = build_chain(content_retriever, tables_retriever)
    retrieval_chain = build_retrieval_chain(content_retriever)

    with st.sidebar.form("metrics"):
        st.write("Retrieve metrics")
        submitted = st.form_submit_button("Submit")
        if submitted:
            _ = _get_metrics(metrics_chain)

    init_messages()

    for message in st.session_state.fc_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if user_query := st.chat_input("Your query"):
        with st.chat_message("user"):
            st.markdown(user_query)

        response = retrieval_chain.invoke(
            {
                "question": user_query,
                "chat_history": st.session_state.fc_messages,
            },
        )
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        st.session_state.fc_messages.append(HumanMessage(content=user_query))
        st.session_state.fc_messages.append(AIMessage(content=response["answer"]))
