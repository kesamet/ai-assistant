import logging

from langchain.llms.ctransformers import CTransformers
from langchain.callbacks import StreamingStdOutCallbackHandler

from src import CFG

logging.basicConfig(level=logging.INFO)


def build_ctransformers(*, model_path: str, model_type: str, debug: bool = False):
    """Builds ctransformers model."""
    logging.info(f"Loading {model_path} ...")
    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        config={
            "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
            "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
            "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
        },
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
    )
    logging.info("Model loaded")
    return llm
