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

output_parser = StrOutputParser()
llm = GooglePalm(temperature=0.0)

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

# Calculator Tool
_MATH_CHAIN = llm_math.prompt.PROMPT | llm | output_parser


class CalculatorInput(BaseModel):
    question: str = Field()


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "A useful tool for answering questions about math"
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
