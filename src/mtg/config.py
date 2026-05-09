import os
from dotenv import load_dotenv

load_dotenv()


def require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


OPENROUTER_API_KEY = require("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "stepfun/step-3.5-flash:free")
OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER", "http://localhost:8000")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "MTG Deck Builder")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "mtgdb")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "mtg")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "mtg")

DATABASE_URL = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
