import os
import time
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openrouter":
        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "stepfun/step-3.5-flash"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost:8000"),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "MTG Deck Builder"),
            },
        )
    else:  # openai
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature,
        )


def structured(schema: type[BaseModel], temperature: float = 0, retries: int = 3):
    """Invoke LLM with structured output, retrying on None."""
    chain = get_llm(temperature).with_structured_output(schema, method="function_calling")

    class _Wrapper:
        def invoke(self, messages):
            for attempt in range(retries):
                result = chain.invoke(messages)
                if result is not None:
                    return result
                wait = 2 ** attempt
                print(f"[llm] None from {schema.__name__}, retry {attempt+1}/{retries} in {wait}s", flush=True)
                time.sleep(wait)
            raise ValueError(f"structured() returned None {retries}x for {schema.__name__}")

    return _Wrapper()
