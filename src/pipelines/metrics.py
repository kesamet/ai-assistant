import fitz
from dotenv import load_dotenv
from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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


def build_content_db(pdf_filepath: str):
    loader = PyPDFLoader(pdf_filepath)
    pages = loader.load_and_split()
    return FAISS.from_documents(pages, EMBEDDINGS)


def build_tables_db(pdf_filepath):
    table_docs = extract_tables(pdf_filepath)
    return FAISS.from_documents(table_docs, EMBEDDINGS)


def extract_tables(pdf_filepath: str) -> list[Document]:
    """Extract tables from PDF."""
    doc = fitz.open(pdf_filepath)
    table_docs = list()
    for page in doc:
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


# def extract_metrics(chain):
#     results = list()
#     for metric in METRICS:
#         res = chain.invoke(f"What is the portfolio {metric}?")
#         results.append(res)
#     return results


def format_query(metrics):
    query = """From the given image, extract the latest value for the following metrics:
{metrics}

If no information can be found for the metric, say "UNK". \
Output your response as a json like this:

{{
    "net property income": ...,
    "distribution per unit (DPU)": ...,
    ...
}}"""
    return query.format(metrics=metrics)


def extract_metrics(pdf_filepath: str) -> dict:
    from src.general import encode_page, sleep

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    chain = llm | JsonOutputParser()

    @sleep(2)
    def get_response(query, img_data):
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                },
            ],
        )

        try:
            return chain.invoke([message])
        except Exception as e:
            logger.error(e)

    doc = fitz.open(pdf_filepath)
    results = {m: "UNK" for m in METRICS}
    for i, page in enumerate(doc):
        metrics = [m for m, v in results.items() if v == "UNK"]
        if len(metrics) == 0:
            print("All found")
            break

        print(f"Page {i + 1}")
        query = format_query(metrics)
        img_data = encode_page(page)
        res = get_response(query, img_data)
        results.update(res)

        for k, v in res.items():
            if v != "UNK":
                print(f"  {k}: {v}")
