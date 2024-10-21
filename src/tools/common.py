from langchain_community.tools import TavilySearchResults, WikipediaQueryRun, WolframAlphaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, WolframAlphaAPIWrapper
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.agent_toolkits import PolygonToolkit

from .newsapi_tool import NewsAPITool

# Web Search Tool
tavily_tool = TavilySearchResults(
    description=(
        "A search engine optimized for comprehensive, accurate, "
        "and trusted results. Useful for when you need to answer questions "
        "about current events or about recent information. "
        "Input should be a search query. "
        "If the user is asking about something that cannot be found in the document, "
        "you should probably use this tool."
    ),
    max_results=4,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Wikipedia Tool
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(),
    description=(
        "A wrapper around Wikipedia. Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    ),
)

# Wolfram Tool
wolfram_tool = WolframAlphaQueryRun(
    api_wrapper=WolframAlphaAPIWrapper(),
    description=(
        "A useful tool for answering complex questions about math, " "such as solving equations."
    ),
)

# Polygon Tools: PolygonAggregates, PolygonLastQuote, PolygonTickerNews, PolygonFinancials
polygon = PolygonAPIWrapper()
toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)
polygon_tools = toolkit.get_tools()

# NewsAPI Tool
newsapi_tool = NewsAPITool()
