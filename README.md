# MTG Deck Builder

```
  ___  ___  _____ ___   ____            _      ____        _ _     _
 |   \/ _ \|_   _/ __| |  _ \  ___  ___| | __ | __ ) _   _(_) | __| | ___ _ __
 | |) \  _/  | || (_ | | | | |/ _ \/ __| |/ / |  _ \| | | | | |/ _` |/ _ \ '__|
 |___/ |_|   |_| \___| |_| |_|\___/\___|_\_\  |_|_) | |_| | | | (_| |  __/ |
                                                |____/ \__,_|_|_|\__,_|\___|_|
```

LangGraph-powered Magic: The Gathering deck builder. Give it a free-text prompt,
get back a full 100-card Commander deck — cards sourced from a local pgvector RAG
store, selected by parallel LLM workers, validated, and repaired automatically.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ parse_query │  LLM → ParsedQuery (format, commander, colors, style, theme)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  plan_deck  │  LLM → DeckPlan (8 slots: commander, theme, ramp, draw,
└──────┬──────┘                   removal, wipes, protection, lands)
       │
       │  fan_out_wave()  ── sends one task per slot in current wave
       ▼
┌─────────────────────────────────────────────┐
│  worker  │  worker  │  worker  │  worker  … │  (parallel, one per slot)
│  RAG sub-queries for role                   │
└─────────────────────┬───────────────────────┘
                      │  (all merge back)
                      ▼
               ┌────────────┐
               │   picker   │  RAG search → LLM picks N cards per slot
               └─────┬──────┘  (fallback: top RAG match if LLM returns 0)
                     │
                     ▼
              ┌──────────────┐
              │ advance_wave │  wave counter +1
              └──────┬───────┘
                     │
          ┌──────────┴──────────┐
          │ more waves?         │ no slots left in next wave
          ▼                     ▼
     (fan out again)     ┌──────────────┐
                         │ hard_validate│  LLM → critical issues (legality, color)
                         └──────┬───────┘
                                │
                   ┌────────────┴────────────┐
              critical?                  no critical
                   ▼                         ▼
            ┌────────────┐          ┌──────────────┐
            │   repair   │◄─────────│ soft_validate│  LLM → high/medium issues
            │ RAG + LLM  │  issues  └──────┬───────┘
            └─────┬──────┘   (max 3x)      │ clean
                  │                        ▼
                  └──────────────►  ┌────────────┐
                                    │  finalize  │  → final_deck
                                    └────────────┘
```

---

## Wave System

Slots are grouped into waves so high-dependency picks happen first:

```
Wave 1 ── commander, theme
Wave 2 ── ramp, draw, protection
Wave 3 ── removal, wipes
Wave 4 ── lands
```

Workers for a wave run in parallel. The picker sees all candidates before
choosing. Waves with no assigned slots are skipped automatically.

---

## Stack

```
┌─────────────────────────────────────────────────────────┐
│                     Browser UI                          │
│   HTML + vanilla JS · dark theme · card image popups   │
│   sidebar of past decks · click to reload any deck      │
└────────────────────┬────────────────────────────────────┘
                     │  POST /api/build
                     │  GET  /api/decks
┌────────────────────▼────────────────────────────────────┐
│              FastAPI  (uvicorn, port 8000)               │
│   mtg.server   serves static + API                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           LangGraph StateGraph  (mtg.graph)             │
│   8 nodes · Send-based fan-out · conditional routing    │
└──────┬─────────────────────────────────────┬────────────┘
       │  structured_output (function call)  │  pgvector similarity search
┌──────▼──────┐                    ┌─────────▼────────────┐
│  OpenAI API │                    │  PostgreSQL + pgvector│
│  gpt-4.1-   │                    │  Docker, port 5433    │
│  nano        │                    │  294 cards, 768-dim   │
└─────────────┘                    │  embeddings           │
                                   └──────────┬────────────┘
                                              │  embed_query
                                   ┌──────────▼────────────┐
                                   │  Ollama               │
                                   │  nomic-embed-text     │
                                   │  (local, free)        │
                                   └───────────────────────┘
```

---

## Project Layout

```
AngelosMTGProject/
├── src/mtg/
│   ├── config.py            env vars
│   ├── llm.py               LLM factory + structured() retry wrapper
│   ├── schemas.py           Pydantic models (ParsedQuery, DeckPlan, Pick, Issue …)
│   ├── graph/
│   │   ├── build.py         StateGraph wiring, routing functions
│   │   ├── state.py         DeckBuildState TypedDict
│   │   ├── templates.py     deck archetype templates
│   │   └── nodes/
│   │       ├── parse_query.py
│   │       ├── plan_deck.py
│   │       ├── worker.py
│   │       ├── picker.py
│   │       ├── hard_validate.py
│   │       ├── soft_validate.py
│   │       └── repair.py
│   ├── rag/
│   │   ├── db.py            pgvector DDL, upsert, similarity_search
│   │   └── search.py        embed + search API
│   ├── ingest/
│   │   ├── fetch_scryfall.py  pull card data from Scryfall API
│   │   ├── reasoning.py       LLM-generated reasoning docs per card
│   │   └── embed.py           embed reasoning + upsert to DB
│   └── server/
│       ├── __init__.py      FastAPI app (build + decks endpoints)
│       └── static/
│           └── index.html   single-page UI
├── docker-compose.yml       pgvector container (port 5433)
├── langgraph.json           langgraph dev config
├── pyproject.toml
└── .env                     API keys, DB config
```

---

## Setup

**Prerequisites:** Docker Desktop, Python 3.11+, Ollama

```bash
# 1. Start the database
docker compose up -d

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Install dependencies
pip install -e .

# 4. Copy and fill in .env
cp .env.example .env   # set OPENAI_API_KEY

# 5. Ingest cards (run once)
python -m mtg.ingest.fetch_scryfall   # fetch from Scryfall
python -m mtg.ingest.reasoning        # generate reasoning docs via LLM
python -m mtg.ingest.embed            # embed + store in pgvector

# 6. Start the web server
uvicorn mtg.server:api --port 8000 --reload

# 7. (Optional) LangGraph Studio
langgraph dev
```

Open http://localhost:8000 — type a prompt, hit **Build Deck**.

---

## Environment Variables

```
LLM_PROVIDER=openai          # "openai" or "openrouter"
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-nano

OPENROUTER_API_KEY=...
OPENROUTER_MODEL=stepfun/step-3.5-flash

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=mtgdb
POSTGRES_USER=mtg
POSTGRES_PASSWORD=mtg

POSTGRES_URI=postgresql://mtg:mtg@localhost:5433/langgraph  # langgraph dev
LANGSMITH_API_KEY=...
```

---

## API

```
POST /api/build
  Body:    { "query": "Mono-green stompy commander with Ghalta" }
  Returns: { "id", "created_at", "deck": [{ slot, card, reason, image_uri }],
             "plan", "issues" }

GET /api/decks
  Returns: last 50 saved decks (newest first)
```
