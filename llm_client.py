import json
import os
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

# API Keys loaded strictly from environment variables or .env file
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")



@dataclass
class Message:
    role: str    
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: object = None

# rate-limit errors

MAX_RETRIES   = 6      
BACKOFF_BASE  = 3      
BACKOFF_MAX   = 120     


def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = (
                "429" in msg
                or "resource exhausted" in msg.lower()
                or "rate limit" in msg.lower()
                or exc.__class__.__name__ in ("RateLimitError", "ResourceExhausted")
            )
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise
            wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 1), BACKOFF_MAX)
            print(f"Rate-limited (attempt {attempt}/{MAX_RETRIES}). Retrying in {wait:.1f}s")
            time.sleep(wait)

class LLMProvider(ABC):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def call(self, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> LLMResponse: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"

# Providers

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-5.5", api_key: Optional[str] = None):
        super().__init__(model, api_key or OPENAI_API_KEY)

    def _raw_call(self, messages, temperature, max_tokens, kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, timeout=120.0)
        
        # Modern models (like gpt-5.5 and o-series) deprecate max_tokens in favor of max_completion_tokens
        # and only support the default temperature (1.0)
        params = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        }
        if "gpt-5" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
            params["max_completion_tokens"] = max_tokens
        else:
            params["temperature"] = temperature
            params["max_tokens"] = max_tokens
            
        return client.chat.completions.create(**params)

    def call(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse:
        raw = _call_with_retry(self._raw_call, messages, temperature, max_tokens, kwargs)
        return LLMResponse(
            content=raw.choices[0].message.content,
            model=raw.model,
            usage={"input_tokens": raw.usage.prompt_tokens, "output_tokens": raw.usage.completion_tokens},
            raw=raw,
        )


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        super().__init__(model, GEMINI_API_KEY)

    def _raw_call(self, messages, temperature, max_tokens, kwargs):
        genai.configure(api_key=self.api_key)
        
        system_instruction = None
        contents = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            elif m.role == "user":
                contents.append({"role": "user", "parts": [m.content]})
            elif m.role in ("assistant", "model"):
                contents.append({"role": "model", "parts": [m.content]})

        # If it is a single-turn request, we can optimize using direct generate_content call
        if len(contents) <= 1:
            generation_config = kwargs.pop("generation_config", None)
            if generation_config is None:
                config_kwargs = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                for key in ["response_mime_type", "response_schema", "candidate_count", "stop_sequences", "top_p", "top_k"]:
                    if key in kwargs:
                        config_kwargs[key] = kwargs.pop(key)
                generation_config = genai.GenerationConfig(**config_kwargs)

            client = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
                system_instruction=system_instruction,
            )
            return client.generate_content(contents, request_options={"timeout": 120.0}, **kwargs)

        # Fallback to multi-turn start_chat to avoid breaking multi-turn pipelines
        client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens),
        )

        history: list[dict] = []
        prepend = ""
        for m in messages:
            if m.role == "system":
                prepend = m.content + "\n\n"
            elif m.role == "user":
                history.append({"role": "user", "parts": [prepend + m.content]})
                prepend = ""
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        *prior, last = history
        config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        chat = client.start_chat(history=prior)
        return chat.send_message(last["parts"][0], generation_config=config, request_options={"timeout": 120.0}, **kwargs)

    def call(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse:
        raw = _call_with_retry(self._raw_call, messages, temperature, max_tokens, kwargs)

        meta = getattr(raw, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(meta, "prompt_token_count", 0),
            "output_tokens": getattr(meta, "candidates_token_count", 0),
        } if meta else {}

        return LLMResponse(content=raw.text, model=self.model, usage=usage, raw=raw)


class TogetherAIProvider(LLMProvider):
    "https://api.together.xyz/models"
    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key: Optional[str] = None, timeout: float = 120.0):
        super().__init__(model, api_key or TOGETHER_API_KEY)
        self.timeout = timeout

    def _raw_call(self, messages, temperature, max_tokens, kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL, timeout=self.timeout)
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def call(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse:
        raw = _call_with_retry(self._raw_call, messages, temperature, max_tokens, kwargs)
        return LLMResponse(
            content=raw.choices[0].message.content,
            model=raw.model,
            usage={"input_tokens": raw.usage.prompt_tokens, "output_tokens": raw.usage.completion_tokens},
            raw=raw,
        )


class LlamaProvider(TogetherAIProvider):
    """Llama via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)


class DeepSeekProvider(TogetherAIProvider):
    """DeepSeek via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "deepseek-ai/DeepSeek-V4-Pro", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)


class QwenProvider(TogetherAIProvider):
    """Qwen via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "Qwen/Qwen3.6-Plus", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)

    def call(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL, timeout=120.0)
        
        def _stream_call():
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            full_text = ""
            input_tokens = 0
            output_tokens = 0
            for chunk in response:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    full_text += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
            
            return full_text, input_tokens, output_tokens
            
        full_text, input_tokens, output_tokens = _call_with_retry(_stream_call)
        
        return LLMResponse(
            content=full_text,
            model=self.model,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            raw=full_text,
        )


class GptOssProvider(TogetherAIProvider):
    """GPT-OSS via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)


class GemmaProvider(TogetherAIProvider):
    """Gemma via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "google/gemma-4-31B-it", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key, timeout=30.0)


class KimiProvider(TogetherAIProvider):
    """Kimi via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "moonshotai/Kimi-K2.6", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key, timeout=120.0)


class LfmProvider(TogetherAIProvider):
    """LFM via TogetherAI (https://api.together.xyz/models)."""

    def __init__(self, model: str = "LiquidAI/LFM2-24B-A2B", api_key: Optional[str] = None):
        super().__init__(model=model, api_key=api_key)


class AnthropicProvider(LLMProvider):
    """Anthropic Provider (e.g. Claude models)."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        super().__init__(model, api_key or (os.environ.get("ANTHROPIC_API_KEY") if os.environ.get("ANTHROPIC_API_KEY") else ANTHROPIC_API_KEY))

    def _raw_call(self, messages, temperature, max_tokens, kwargs):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("The 'anthropic' package is required to use AnthropicProvider. Run 'pip install anthropic' to install it.")

        client = Anthropic(api_key=self.api_key)
        
        system_content = ""
        anthropic_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})
                
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_content:
            params["system"] = system_content
        # Omit temperature for reasoning models like claude-opus-4-8
        if temperature is not None and "opus-4" not in self.model.lower():
            params["temperature"] = temperature
            
        return client.messages.create(**params)

    def call(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse:
        raw = _call_with_retry(self._raw_call, messages, temperature, max_tokens, kwargs)
        text = raw.content[0].text
        usage = {
            "input_tokens": raw.usage.input_tokens,
            "output_tokens": raw.usage.output_tokens,
        }
        return LLMResponse(content=text, model=self.model, usage=usage, raw=raw)





def ask_and_save(
    prompt: str,
    provider: LLMProvider,
    *,
    system_prompt: str = "You are a helpful assistant.",
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Send *prompt* to the LLM and save the response to a JSON file.

    Returns the response text.
    The saved file contains the prompt, response, model, usage, and timestamp.
    """
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt),
    ]

    response = provider.call(messages, temperature=temperature, max_tokens=max_tokens)

    record = {
        "timestamp": datetime.now().isoformat(),
        "model": response.model,
        "prompt": prompt,
        "response": response.content,
        "usage": response.usage,
    }

    if output_file is None:
        Path("responses").mkdir(exist_ok=True)
        output_file = f"responses/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{response.model.replace('/', '-')}.json"

    Path(output_file).write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"✓ Response saved to: {output_file}")

    return response.content



if __name__ == "__main__":
    provider = GeminiProvider()
    # provider = OpenAIProvider()
    # provider = LlamaProvider()
    # provider = QwenProvider()
    # provider = DeepSeekProvider()
    # provider = GptOssProvider()
    # provider = KimiProvider()

    prompt = "Explain the difference between supervised and unsupervised learning in 3 sentences."

    reply = ask_and_save(prompt, provider)
    print(f"\n{reply}")
