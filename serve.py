import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src import CFG
from src.llama2 import load_llama2
from src.codellama import load_codellama
from src.mistral import load_mistral


app = FastAPI()


class Request(BaseModel):
    inputs: str


@app.post("/")
async def get_response(request: Request) -> Any:
    return {"content": LLM(request.inputs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str)
    args = parser.parse_args()

    if args.model == "llama2":
        LLM = load_llama2()
        uvicorn.run(app, host=CFG.HOST, port=CFG.PORT_LLAMA2)
    elif args.model == "codellama":
        LLM = load_codellama()
        uvicorn.run(app, host=CFG.HOST, port=CFG.PORT_CODELLAMA)
    elif args.model == "mistral":
        LLM = load_mistral()
        uvicorn.run(app, host=CFG.HOST, port=CFG.PORT_MISTRAL)
    else:
        raise NotImplementedError
