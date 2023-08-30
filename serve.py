import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src import CFG
from src.llama2chat import load_llama2chat
from src.codellama import load_codellama


app = FastAPI()


class Request(BaseModel):
    inputs: str


@app.post("/")
async def get_response(request: Request) -> Any:
    payload = request.model_dump()
    return {"content": LLM(payload["inputs"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str)
    args = parser.parse_args()

    host = CFG.HOST
    if args.model == "llama2chat":
        LLM = load_llama2chat()
        uvicorn.run(app, host=host, port=8000)
    elif args.model == "codellama":
        LLM = load_codellama()
        uvicorn.run(app, host=host, port=8001)
    else:
        raise NotImplementedError
