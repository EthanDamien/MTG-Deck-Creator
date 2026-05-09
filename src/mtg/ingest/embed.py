"""Embed card reasonings and insert into pgvector."""
from langchain_ollama import OllamaEmbeddings
from mtg.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
from mtg.rag.db import init_db, get_conn, upsert_card


def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def embed_and_insert(cards: list[dict], batch_size: int = 50) -> int:
    """Embed each card's reasoning and upsert into Postgres. Returns count inserted."""
    init_db()
    embedder = get_embedder()

    inserted = 0
    with get_conn() as conn:
        for i in range(0, len(cards), batch_size):
            batch = cards[i: i + batch_size]
            texts = [c["reasoning"] or c["oracle_text"] for c in batch]
            embeddings = embedder.embed_documents(texts)
            for card, emb in zip(batch, embeddings):
                upsert_card(conn, card, emb)
                inserted += 1
            conn.commit()
            print(f"  Inserted {inserted}/{len(cards)} cards...")

    return inserted
