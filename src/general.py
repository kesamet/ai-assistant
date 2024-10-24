import base64
import os
import tempfile
import time
from io import BytesIO

import fitz
import numpy as np
from loguru import logger
from PIL import Image


def perform(func, filebytes, **kwargs):
    """Wrapper function to perform func for bytes file."""
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(f.name, **kwargs)
    finally:
        os.close(fh)
        os.remove(temp_filename)


def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return
                except Exception:
                    logger.info(f"Sleeping for {timeout} seconds")
                    time.sleep(timeout)
                    retries += 1

        return wrapper

    return the_real_decorator


def encode_uri_image(uri: str) -> str:
    """Get base64 encoded string from image URI."""
    with open(uri, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image(image: Image.Image) -> str:
    """Encode a PIL image to base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


def encode_fitz_page(page: fitz.Page) -> str:
    """Encode a fitz page to base64 encoded string."""
    pix = page.get_pixmap()
    return base64.b64encode(pix.pil_tobytes("png")).decode("utf-8")


def decode_image(field: str) -> np.ndarray:
    """Decode a base64 encoded image to a numpy array of floats."""
    import cv2

    array = np.frombuffer(base64.b64decode(field), dtype=np.uint8)
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)  # BGR
    return image_array
