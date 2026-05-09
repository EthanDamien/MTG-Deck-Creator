import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """OpenRouter via the OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENROUTER_BASE_URL"],
        temperature=temperature,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", ""),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "MTG Deck Builder"),
        },
    )


def structured(schema: type[BaseModel], temperature: float = 0):
    """get_llm bound to a Pydantic schema via function_calling."""
    return get_llm(temperature).with_structured_output(
        schema,
        method="function_calling",
    )
