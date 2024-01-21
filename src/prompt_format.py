from typing import List, Dict


class PromptFormat:
    B_INST = ""
    E_INST = ""
    B_SYS = ""
    E_SYS = ""
    BOS = ""
    EOS = ""
    SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe."
    )

    def get_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Converts messages to compliant prompt format."""
        # if messages are in langchain.schema, convert to dict
        if messages[0]["role"] != "system":
            system_prompt = self.SYSTEM_PROMPT
        else:
            system_prompt = messages[0]["content"].strip()
            messages = messages[1:]

        chat_history = [
            (prompt["content"], answer["content"])
            for prompt, answer in zip(messages[::2], messages[1::2])
        ]

        user_prompt = messages[-1]["content"]
        return self._format(user_prompt, chat_history, system_prompt)

    def _format(
        self, message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> str:
        """Copied from https://huggingface.co/spaces/codellama/codellama-13b-chat/blob/main/model.py#L25-L36."""
        texts = [f"{self.BOS}{self.B_INST} {self.B_SYS}{system_prompt}{self.E_SYS}"]
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(
                f"{user_input} {self.E_INST} {response.strip()} {self.EOS}{self.BOS}{self.B_INST} "
            )
        message = message.strip() if do_strip else message
        texts.append(f"{message} {self.E_INST}")
        return "".join(texts)


class Llama2Format(PromptFormat):
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    BOS = "<s>"
    EOS = "</s>"


class CodeLlamaFormat(PromptFormat):
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    BOS = "<s>"
    EOS = "</s>"
    SYSTEM_PROMPT = "You are a helpful, respectful and honest code assistant."


class MistralFormat(PromptFormat):
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = ""
    E_SYS = "\n"
    BOS = "<s>"
    EOS = "</s>"
