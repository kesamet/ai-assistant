from langchain import PromptTemplate
from langchain.chat_models import ChatGooglePalm
from langchain.agents import AgentType, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from src.tools import search_tool, wikipedia_tool

llm = ChatGooglePalm(temperature=0.0)

prompt = PromptTemplate(
    template="""Plan: {input}

History: {chat_history}

Let's think about answer step by step.
If it's information retrieval task, solve it like a professor in particular field.""",
    input_variables=["input", "chat_history"],
)

plan_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""Prepare plan for task execution. (e.g. retrieve current date to find weather forecast)

    Tools to use: wikipedia, web search

    REMEMBER: Keep in mind that you don't have information about current date, temperature, informations after September 2021. Because of that you need to use tools to find them.

    Question: {input}

    History: {chat_history}

    Output look like this:
    '''
        Question: {input}

        Execution plan: [execution_plan]

        Rest of needed information: [rest_of_needed_information]
    '''

    IMPORTANT: if there is no question, or plan is not need (YOU HAVE TO DECIDE!), just populate {input} (pass it as a result). Then output should look like this:
    '''
        input: {input}
    '''
    """,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

plan_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[search_tool, wikipedia_tool],
    llm=llm,
    verbose=True,
    max_iterations=3,
    prompt=prompt,
    memory=memory,
)
