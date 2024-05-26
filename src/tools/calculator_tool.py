from typing import Any, Optional, Type

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.llm_math.prompt import PROMPT
from langchain.tools import BaseTool


class CalculatorInput(BaseModel):
    question: str = Field()


class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "A useful tool for answering simple questions about math."
    args_schema: Type[BaseModel] = CalculatorInput
    llm_chain: Runnable

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self.llm_chain.invoke({"question": query})

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise self.llm_chain.ainvoke({"question": query})

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> Runnable:
        llm_chain = prompt | llm
        return cls(llm_chain=llm_chain, **kwargs)


# problem_chain = LLMMathChain.from_llm(llm=llm)
# math_tool = Tool.from_function(
#     name="Calculator",
#     func=problem_chain.run,
#     description="Useful for when you need to answer questions about math."
# )
