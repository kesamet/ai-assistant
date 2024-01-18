import argparse
import time
import uuid
import asyncio
from typing import AsyncIterable, Awaitable, Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.llms.ctransformers import CTransformers
from langchain.callbacks import (
    AsyncIteratorCallbackHandler,
    StreamingStdOutCallbackHandler,
)

from src import CFG
from src.prompt_format import Llama2Format, MistralFormat, CodeLlamaFormat

ASYNC_CALLBACK = AsyncIteratorCallbackHandler()


class Request(BaseModel):
    messages: List[Dict[str, str]]


app = FastAPI()


# chat completion
@app.get("/v1/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completion(request: Request) -> dict:
    prompt = PROMPT_FORMAT.get_prompt(request.messages)
    print(prompt)
    output = LLM.invoke(prompt)

    return {
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": output,
                    "role": "assistant",
                },
            }
        ],
        "id": "chatcmpl-" + str(uuid.uuid4()),
        "created": time.time(),
        "model": MODEL_NAME,
        "usage": {"completion_tokens": -1, "prompt_tokens": -1, "total_tokens": -1},
    }


async def request_stream(prompt: str) -> AsyncIterable[str]:
    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(LLM.ainvoke(prompt), ASYNC_CALLBACK.done))

    async for token in ASYNC_CALLBACK.aiter():
        # Use server-sent-events to stream the response
        yield f"{token}"

    await task


@app.get("/stream")
@app.post("/stream")
async def chat_completion_stream(request: Request):
    prompt = PROMPT_FORMAT.get_prompt(request.messages)
    print(prompt)
    return StreamingResponse(request_stream(prompt), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str)
    args = parser.parse_args()

    if args.model == "llama2":
        MODEL_NAME = CFG.LLAMA2.MODEL_NAME
        model_path = f"{CFG.MODELS_DIR}/{CFG.LLAMA2.MODEL_PATH}"
        model_type = CFG.LLAMA2.MODEL_TYPE
        PROMPT_FORMAT = Llama2Format()
        port = CFG.PORT_LLAMA2
    elif args.model == "codellama":
        MODEL_NAME = CFG.CODELLAMA.MODEL_NAME
        model_path = f"{CFG.MODELS_DIR}/{CFG.CODELLAMA.MODEL_PATH}"
        model_type = CFG.CODELLAMA.MODEL_TYPE
        PROMPT_FORMAT = CodeLlamaFormat()
        port = CFG.PORT_CODELLAMA
    elif args.model == "mistral":
        MODEL_NAME = CFG.MISTRAL.MODEL_NAME
        model_path = f"{CFG.MODELS_DIR}/{CFG.MISTRAL.MODEL_PATH}"
        model_type = CFG.MISTRAL.MODEL_TYPE
        PROMPT_FORMAT = MistralFormat()
        port = CFG.PORT_MISTRAL
    else:
        raise NotImplementedError

    LLM = CTransformers(
        model=model_path,
        config={
            "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
            "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
            "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
        },
        callbacks=[ASYNC_CALLBACK, StreamingStdOutCallbackHandler()],
    )

    uvicorn.run(app, host=CFG.HOST, port=port)
