import requests


def setup_tracing(url: str = "http://localhost:6006/"):
    """Tracing with arize-pheonix."""
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:
        return False, e
    else:
        from phoenix.trace.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
        return True, None
