"""High-level RAG search API used by the picker node."""
from langchain_ollama import OllamaEmbeddings
from mtg.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from mtg.rag.db import get_conn, similarity_search

_embedder: OllamaEmbeddings | None = None


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return _embedder


def search(
    query: str,
    limit: int = 10,
    color_identity: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[dict]:
    """Embed query and return top-k similar cards."""
    emb = _get_embedder().embed_query(query)
    with get_conn() as conn:
        results = similarity_search(conn, emb, limit=limit * 2, color_identity=color_identity)

    if exclude_names:
        excluded = set(n.lower() for n in exclude_names)
        results = [r for r in results if r["name"].lower() not in excluded]

    return results[:limit]
