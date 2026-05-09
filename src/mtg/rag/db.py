import json
from typing import Optional
import psycopg
from psycopg.rows import dict_row
from mtg.config import DATABASE_URL, EMBEDDING_DIM

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cards (
    id BIGSERIAL PRIMARY KEY,
    oracle_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    mana_cost TEXT,
    cmc REAL,
    colors TEXT[],
    color_identity TEXT[],
    type_line TEXT,
    oracle_text TEXT,
    keywords TEXT[],
    power TEXT,
    toughness TEXT,
    edhrec_rank INT,
    legalities JSONB,
    image_uri TEXT,
    reasoning TEXT,
    embedding VECTOR({EMBEDDING_DIM})
);

CREATE INDEX IF NOT EXISTS cards_embedding_idx
    ON cards USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS cards_name_idx ON cards (name);
CREATE INDEX IF NOT EXISTS cards_color_identity_idx ON cards USING gin (color_identity);
"""

_dsn = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


def get_conn():
    return psycopg.connect(_dsn, row_factory=dict_row)


def init_db():
    with get_conn() as conn:
        conn.execute(DDL)
        conn.commit()


def upsert_card(conn, card: dict, embedding: list[float]):
    conn.execute(
        """
        INSERT INTO cards (
            oracle_id, name, mana_cost, cmc, colors, color_identity,
            type_line, oracle_text, keywords, power, toughness,
            edhrec_rank, legalities, image_uri, reasoning, embedding
        ) VALUES (
            %(oracle_id)s, %(name)s, %(mana_cost)s, %(cmc)s, %(colors)s,
            %(color_identity)s, %(type_line)s, %(oracle_text)s, %(keywords)s,
            %(power)s, %(toughness)s, %(edhrec_rank)s, %(legalities)s,
            %(image_uri)s, %(reasoning)s, %(embedding)s
        )
        ON CONFLICT (oracle_id) DO UPDATE SET
            reasoning = EXCLUDED.reasoning,
            embedding = EXCLUDED.embedding,
            edhrec_rank = EXCLUDED.edhrec_rank
        """,
        {**card, "legalities": json.dumps(card["legalities"]), "embedding": embedding},
    )


def similarity_search(
    conn,
    query_embedding: list[float],
    limit: int = 10,
    color_identity: Optional[list[str]] = None,
) -> list[dict]:
    if color_identity:
        rows = conn.execute(
            """
            SELECT name, oracle_id, mana_cost, cmc, type_line, oracle_text,
                   color_identity, reasoning, image_uri, edhrec_rank,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM cards
            WHERE color_identity <@ %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, color_identity, query_embedding, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT name, oracle_id, mana_cost, cmc, type_line, oracle_text,
                   color_identity, reasoning, image_uri, edhrec_rank,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM cards
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit),
        ).fetchall()
    return [dict(r) for r in rows]
