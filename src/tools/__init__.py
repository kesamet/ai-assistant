from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains import llm_math
from langchain.llms.google_palm import GooglePalm
from langchain.schema import StrOutputParser
from langchain.tools import BaseTool, Tool, WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from .newsapi import NewsAPITool
from .yahoo_finance import YahooFinanceTool

llm = GooglePalm(temperature=0.0)
output_parser = StrOutputParser()

# Wikipedia Tool
_wikipedia = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=_wikipedia)
# wikipedia_tool = Tool(
#     name="Wikipedia",
#     func=wikipedia.run,
#     description="A useful tool for searching wikipedia to find information.",
# )

# Web Search Tool
_search = SerpAPIWrapper()
search_tool = Tool(
    name="Web Search",
    func=_search.run,
    description="A useful tool for searching the Internet to find information on world events, \
        issues, etc. Worth using for general topics. Use precise questions.",
)

# Wolfram Tool
_wolfram = WolframAlphaAPIWrapper()
wolfram_tool = Tool(
    name="WolframAlpha",
    func=_wolfram.run,
    description="A useful tool for answering complex questions about math, such as solving equations, etc.",
)

# NewsAPI Tool
newsapi_tool = NewsAPITool()

# YahooFinance Tool
yahoo_finance_tool = YahooFinanceTool()

# Calculator Tool
_MATH_CHAIN = llm_math.prompt.PROMPT | llm | output_parser


class CalculatorInput(BaseModel):
    question: str = Field()


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "A useful tool for answering simple questions about math."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return _MATH_CHAIN.invoke({"question": query})

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise _MATH_CHAIN.ainvoke({"question": query})


calculator_tool = CustomCalculatorTool()
