from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_google_genai import GoogleGenerativeAI

from .newsapi_tool import NewsAPITool
from .calculator_tool import CalculatorTool

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.0)

# Web Search Tool
search = TavilySearchAPIWrapper()
description = (
    "A search engine optimized for comprehensive, accurate, "
    "and trusted results. Useful for when you need to answer questions "
    "about current events or about recent information. "
    "Input should be a search query. "
    "If the user is asking about something that you don't know about, "
    "you should probably use this tool to see if that can provide any information."
)
tavily_tool = TavilySearchResults(api_wrapper=search, description=description)

# Wikipedia Tool
_wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=_wikipedia.run,
    description=(
        "A wrapper around Wikipedia. Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects."
    ),
)

# Wolfram Tool
_wolfram = WolframAlphaAPIWrapper()
wolfram_tool = Tool(
    name="WolframAlpha",
    func=_wolfram.run,
    description=(
        "A useful tool for answering complex questions about math, "
        "such as solving equations."
    ),
)

# NewsAPI Tool
newsapi_tool = NewsAPITool()

# Calculator Tool
calculator_tool = CalculatorTool.from_llm(llm)
