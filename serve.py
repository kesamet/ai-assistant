from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src import CFG
from src.llama2 import load_llama2


class Request(BaseModel):
    inputs: str


LLM = load_llama2()

app = FastAPI()


@app.post("/")
async def get_response(request: Request) -> Any:
    return {"content": LLM(request.dict()["inputs"])}


if __name__ == "__main__":
    uvicorn.run(app, host=CFG.HOST, port=CFG.PORT)
