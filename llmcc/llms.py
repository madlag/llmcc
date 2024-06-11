import os
from contextlib import asynccontextmanager

import openai
from anthropic import AsyncAnthropic


class LLM:
    def __init__(self):
        pass

    @asynccontextmanager
    async def get_stream(self, prompt, max_tokens=1024):
        raise NotImplementedError

    async def get_full_message(self, *args, **kwargs):
        async with self.get_stream(*args, **kwargs) as stream:
            final_message = await stream.get_final_message()
            ret = ""
            for content in final_message.content:
                ret += content.text
            return ret

    SYNONYMS = {"gpt4": "gpt-4", "gpt4o": "gpt-4o"}

    @staticmethod
    def llm_by_name(model_name, *args, **kwargs):
        model_name = LLM.SYNONYMS.get(model_name, model_name)

        if "/" in model_name:
            llm_provider, model_name = model_name.split("/")
        else:
            if model_name in AnthropicLLM.MODEL_NAMES:
                llm_provider = "anthropic"
            elif model_name in OpenAILLM.MODEL_NAMES:
                llm_provider = "openai"
            else:
                raise ValueError(f"LLM type {model_name} not found")

        if llm_provider == "anthropic":
            return AnthropicLLM(model_name=model_name, *args, **kwargs)
        elif llm_provider == "openai":
            return OpenAILLM(model_name=model_name, *args, **kwargs)
        else:
            raise ValueError(f"LLM type {model_name} not found")


class AnthropicLLM(LLM):
    MODEL_NAMES = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
    ]

    def __init__(self, model_name, api_key=None, temperature=1.0):
        super().__init__()
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Model name {model_name} not found")

        self.model_name = self.model_name
        api_key = (
            api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise Exception("Missing Anthropic API key")
        self.anthropic = AsyncAnthropic(api_key=api_key)
        self.temperature = temperature

    @asynccontextmanager
    async def get_stream(self, messages=None, prompt=None, max_tokens=1024):
        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided")
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        async with self.anthropic.messages.stream(
            max_tokens=max_tokens,
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
        ) as stream:
            yield stream


class OpenAILLM(LLM):
    MODEL_NAMES = ["gpt-4", "gpt-4o"]

    def __init__(self, model_name, api_key=None, temperature=1.0):
        super().__init__()
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Model name {model_name} not found")
        self.model_name = model_name
        api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Missing OpenAI API key")
        self.temperature = temperature
        self.openai_client = openai.OpenAI()

    def get_stream(self, messages=None, prompt=None, max_tokens=1024):
        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided")
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        return self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            stream=True,
        )

    async def get_full_message(self, *args, **kwargs):
        full_text = ""
        stream = self.get_stream(*args, **kwargs)
        for t in stream:
            content = t.choices[0].delta.content
            if content is not None:
                full_text += content
        return full_text
