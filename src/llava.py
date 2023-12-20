import base64
import logging
import os

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from src import CFG

logging.basicConfig(level=logging.INFO)


def load_llava() -> Llama:
    """Load llava model."""
    logging.info("Loading llava model ...")

    model = Llama(
        model_path=os.path.join(CFG.MODELS_DIR, CFG.LLAVA_MODEL_PATH),
        chat_handler=Llava15ChatHandler(
            clip_model_path=os.path.join(CFG.MODELS_DIR, CFG.CLIP_MODEL_PATH)
        ),
        n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
        logits_all=True,
    )
    logging.info("Model loaded")
    return model


def encode_image(uri: str) -> str:
    """Get base64 string from image URI."""
    with open(uri, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response(model: Llama, messages: list) -> dict:
    output = model.create_chat_completion(messages=messages)
    return output
