from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.google_palm import GooglePalm
from langchain.schema import HumanMessage, AIMessage

from src.tools import search_tool, wikipedia_tool, calculator_tool

LLM = GooglePalm(temperature=0.0)
BUFFER = 5

def build_agent(messages):
    memory = _build_memory(messages)
    agent = initialize_agent(
        [search_tool, wikipedia_tool, calculator_tool],
        LLM,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        memory=memory,
    )
    return agent


def _build_memory(messages):
    memory = ConversationBufferWindowMemory(k=BUFFER, memory_key="chat_history", return_messages=True)
    for message in messages[-BUFFER * 2:]:
        if isinstance(message, AIMessage):
            memory.chat_memory.add_ai_message(message.content)
        elif isinstance(message, HumanMessage):
            memory.chat_memory.add_user_message(message.content)
    return memory
