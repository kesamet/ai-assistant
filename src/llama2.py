from typing import List, Union

from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage, HumanMessage, AIMessage


def build_llama2() -> CTransformers:
    model = "../retrieval-augmented-generation/models/llama-2-7b-chat-ggml/llama-2-7b-chat.ggmlv3.q2_K.bin"
    return CTransformers(
        model=model,
        model_type="llama",
        config={
            "max_new_tokens": 512,
            "temperature": 0.01,
        },
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )


def llama2_prompt(messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> str:
    """
    Convert the messages to Llama2 compliant format.
    """
    messages = convert_langchainschema_to_dict(messages)

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    BOS = "<s>"
    EOS = "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe. Please ensure that your responses \
are socially unbiased and positive in nature. If a question does not make any sense, \
or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def convert_langchainschema_to_dict(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
) -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [
        {"role": find_role(message), "content": message.content} for message in messages
    ]


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")
