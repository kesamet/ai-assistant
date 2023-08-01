
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper


search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

# Web Search Tool
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)
