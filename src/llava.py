import os

from loguru import logger
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from src import CFG


def load_llava() -> Llama:
    """Load llava model."""
    logger.info("Loading llava model ...")

    model = Llama(
        model_path=os.path.join(CFG.MODELS_DIR, CFG.LLAVA_MODEL_PATH),
        chat_handler=Llava15ChatHandler(
            clip_model_path=os.path.join(CFG.MODELS_DIR, CFG.CLIP_MODEL_PATH)
        ),
        n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
        logits_all=True,
    )
    logger.info("Model loaded")
    return model


def get_response(model: Llama, messages: list, **kwargs) -> dict:
    output = model.create_chat_completion(messages=messages, **kwargs)
    return output
