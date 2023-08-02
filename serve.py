from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


model = "../retrieval-augmented-generation/models/llama-2-7b-chat-ggml/llama-2-7b-chat.ggmlv3.q2_K.bin"
LLM = CTransformers(
    model=model,
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.01,
    },
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
)


class Request(BaseModel):
    inputs: str


app = FastAPI()


@app.post("/")
async def get_response(request: Request) -> Any:
    return {"content": LLM(request.dict()["inputs"])}


if __name__ == "__main__":
    uvicorn.run(app)
