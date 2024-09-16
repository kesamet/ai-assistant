from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch


def condense_question_chain(llm: LLM, prompt: BasePromptTemplate | None = None) -> Runnable:
    """Builds a chain that condenses question and chat history to create a standalone question."""
    if prompt is None:
        template = (
            "Given the following chat history and a follow up question, "
            "rephrase the follow up question to be a standalone question, in its original language.\n\n"
            "Chat History:\n{chat_history}\n\nFollow Up Question: {question}\nStandalone question:"
        )
        prompt = PromptTemplate.from_template(template)

    chain = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input
            (lambda x: x["question"]),
        ),
        # If chat history, then we pass inputs to LLM chain
        prompt | llm | StrOutputParser(),
    )
    return chain
