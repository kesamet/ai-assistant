import fitz
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

_ = load_dotenv()

LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
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
    return build_content_db(pdf_filepath)


@st.cache_data
def _get_tables_db(pdf_filepath):
    return build_tables_db(pdf_filepath)


@st.cache_data
def _get_metrics(_chain):
    return extract_metrics(_chain)


def build_content_db(pdf_filepath: str):
    loader = PyPDFLoader(pdf_filepath)
    pages = loader.load_and_split()
    return FAISS.from_documents(pages, EMBEDDINGS)


def build_tables_db(pdf_filepath):
    table_docs = extract_tables(pdf_filepath)
    return FAISS.from_documents(table_docs, EMBEDDINGS)


def extract_tables(pdf_filepath: str) -> list[Document]:
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


if __name__ == "__main__":
    st.title("Extract metrics")

    with st.form("metrics"):
        pdf_filepath = st.text_input("Input pdf filepath")

        submitted = st.form_submit_button("Submit")
        if submitted:
            content_db = _get_content_db(pdf_filepath)
            content_retriever = content_db.as_retriever()
            tables_db = _get_tables_db(pdf_filepath)
            tables_retriever = tables_db.as_retriever()
            chain = build_chain(content_retriever, tables_retriever)
            _ = _get_metrics(chain)
