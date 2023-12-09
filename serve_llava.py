from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src import CFG
from src.llava import load_llava

LLM = load_llava()

app = FastAPI()


class Request(BaseModel):
    inputs: list


@app.post("/")
async def get_response(request: Request) -> Any:
    output = LLM.create_chat_completion(messages=request.inputs)
    return output


if __name__ == "__main__":
    uvicorn.run(app, host=CFG.HOST, port=CFG.PORT_LLAVA)
