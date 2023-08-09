from typing import Any, Optional, Type

from pydantic import BaseModel, Field
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains import RetrievalQA
from langchain.chains.llm_math.prompt import PROMPT
from langchain.tools import BaseTool, Tool
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain.llms import GooglePalm
from langchain.chat_models import ChatGooglePalm

llm = GooglePalm(temperature=0.0)
# llm = ChatGooglePalm(temperature=0.0)

search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, \
        issues, etc. Worth using for general topics. Use precise questions.",
)

# Web Search Tool
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, \
        issues, etc. Worth using for general topics. Use precise questions.",
)


class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


# Calculator Tool
llm_math = LLMChain(llm=llm, prompt=PROMPT)


class CalculatorInput(BaseModel):
    question: str = Field()


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return llm_math.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")


# vectordb = ...


# # Custom Retrieval QA Tool
# QA_TEMPLATE = """Use the following pieces of information to answer the user's question. \
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# prompt = PromptTemplate(
#     template=QA_TEMPLATE,
#     input_variables=["context", "question"],
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
#     return_source_documents=False,
#     chain_type_kwargs={"prompt": prompt},
#     verbose=True,
# )


# class CustomRetrievalTool(BaseTool):
#     retriever: Any
#     return_direct = True

#     def _run(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         result = self.retriever({"query": query})
#         answer = result["result"]
#         # Convert the list of source_documents to a string.
#         sources = str(result["source_documents"])
#         # Concatenate the answer and the sources into a single string.
#         return f"Answer: {answer}\nSources:\n{sources}"

#     async def _arun(
#         self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("CustomRetrieval does not support async")


# custom_qa_tool = CustomRetrievalTool(
#     name="Information Extractor",
#     description="This tool is capable of extracting factual data related to a provided query.",
#     retriever=qa,
# )
