# MTG Deck Builder — Executable Build Plan

> A step-by-step plan for an autonomous LLM (Claude Opus or similar) to implement the system end-to-end on a local machine, with a quick path from zero to a working baseline before any optimization.

---

## Architecture Recap (one paragraph)

A LangGraph orchestrator takes a free-text query, picks a hardcoded archetype template, and runs **wave-based parallel deck building**: theme cards → support (ramp/draw/protection) → answers (removal/wipes) → lands. Each wave fans out parallel "worker" LLMs that generate dense-phrase RAG queries, hits a local pgvector store of card reasonings, and a "picker" LLM chooses winners with cross-slot synergy awareness. A deterministic hard validator + an LLM soft validator drive a surgical repair loop. A 300-card subset is used for the local baseline; full 30k Scryfall is the production target.

---

## Tech Stack (locked in)

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph 1.x** (`langgraph` pip pkg 0.3.x+) | Subgraphs, parallel `Send` API, structured outputs |
| LLM (workers/picker/validator) | **`stepfun/step-3.5-flash:free`** via OpenRouter | Free tier, 256k context, reasoning MoE, OpenAI-compatible API |
| Embeddings | **`nomic-embed-text` via Ollama** (free, local) | No API cost, 768-dim, runs on CPU |
| Vector DB | **PostgreSQL 16+ with pgvector** (Docker) | Single store for vectors + card metadata |
| Card data | **Scryfall bulk API** | Free, complete, no scraping |
| Backend | **FastAPI** | Quick local server |
| Frontend | **Single React/HTML page** served by FastAPI | Zero build setup |
| Schemas | **Pydantic v2** | Structured outputs for every node |

---

## Phase 0 — Local Setup

### 0.1 Prerequisites

```bash
# Required
python --version    # 3.11+
docker --version    # for Postgres
node --version      # optional, only if separate frontend
```

Install Ollama (for free local embeddings):

```bash
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.com/install.sh | sh

ollama pull nomic-embed-text
ollama serve  # runs on http://localhost:11434
```

### 0.2 Project skeleton

```
mtg-deck-builder/
├── docker-compose.yml          # Postgres + pgvector
├── pyproject.toml              # uv or poetry
├── .env.example
├── README.md
├── src/
│   └── mtg/
│       ├── __init__.py
│       ├── config.py           # env vars, model names
│       ├── schemas.py          # ALL Pydantic models — single source of truth
│       ├── ingest/
│       │   ├── fetch_scryfall.py
│       │   ├── reasoning.py    # batched LLM reasoning generation
│       │   └── embed.py        # embed + insert into pgvector
│       ├── rag/
│       │   ├── db.py           # pgvector connection + similarity search
│       │   └── search.py       # high-level RAG search API
│       ├── graph/
│       │   ├── state.py        # DeckBuildState TypedDict
│       │   ├── nodes/
│       │   │   ├── parse_query.py
│       │   │   ├── plan_deck.py
│       │   │   ├── worker.py
│       │   │   ├── picker.py
│       │   │   ├── hard_validate.py
│       │   │   ├── soft_validate.py
│       │   │   └── repair.py
│       │   ├── templates.py    # archetype skeletons
│       │   └── build.py        # graph wiring
│       ├── prompts/
│       │   ├── parse_query.txt
│       │   ├── reasoning.txt
│       │   ├── worker.txt
│       │   ├── picker.txt
│       │   └── soft_validator.txt
│       └── server/
│           ├── api.py          # FastAPI routes
│           └── static/
│               └── index.html  # baseline UI
├── scripts/
│   ├── ingest_300.py           # 300-card baseline
│   ├── ingest_full.py          # full 30k
│   └── eval_node.py            # node-level testability harness
└── tests/
    └── nodes/
        ├── test_parse_query.py
        ├── test_worker.py
        └── test_picker.py
```

### 0.3 Environment

`.env.example`:
```bash
# OpenRouter (free LLM tier)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=stepfun/step-3.5-flash:free

# Optional headers OpenRouter rewards (better routing + leaderboard credit)
OPENROUTER_REFERER=http://localhost:8000
OPENROUTER_TITLE=MTG Deck Builder

# Embeddings (local, free)
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# Postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mtgdb
POSTGRES_USER=mtg
POSTGRES_PASSWORD=mtg
```

### 0.4 docker-compose.yml

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    container_name: mtg-pgvector
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: mtg
      POSTGRES_PASSWORD: mtg
      POSTGRES_DB: mtgdb
    volumes:
      - mtg_pgdata:/var/lib/postgresql/data
volumes:
  mtg_pgdata:
```

```bash
docker compose up -d
```

### 0.6 LLM client helper

Centralize the OpenRouter client so every node uses the same factory. `src/mtg/llm.py`:

```python
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """OpenRouter via the OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],                       # stepfun/step-3.5-flash:free
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENROUTER_BASE_URL"],          # https://openrouter.ai/api/v1
        temperature=temperature,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", ""),
            "X-Title":      os.environ.get("OPENROUTER_TITLE", "MTG Deck Builder"),
        },
        # Step 3.5 Flash is a reasoning model — extended_thinking is automatic;
        # if responses are slow, you can pass `model_kwargs={"reasoning": {"enabled": False}}`
    )

def structured(schema: type[BaseModel], temperature: float = 0):
    """Shortcut: get_llm bound to a Pydantic schema.

    Uses method='function_calling' explicitly. Step 3.5 Flash was trained
    primarily on XML tool-call templates (per StepFun's paper); function
    calling is its strongest structured-output path. The json_schema method
    can be flaky on complex nested schemas — avoid it for this model.
    """
    return get_llm(temperature).with_structured_output(
        schema,
        method="function_calling",
    )
```

Every node imports `from mtg.llm import structured` and calls `structured(ParsedQuery).invoke(...)`. Swapping models later is a one-line change.

> ⚠️ **Free-tier caveats.** OpenRouter's `:free` endpoints are rate-limited (per-minute and per-day) and shared across users — expect occasional `429`s. Wrap every LLM call in a `tenacity` retry with exponential backoff. If you hit rate limits during ingest, drop concurrency from 10 → 3, or fall back to a paid tier (`stepfun/step-3.5-flash` without `:free` is still cheap).

### 0.5 pyproject.toml (uv preferred)

Core deps:
```
langgraph>=0.3.34
langchain-openai>=0.3.12      # used as OpenAI-compatible client pointed at OpenRouter
langchain-ollama
langchain-postgres
pgvector
psycopg[binary]
sqlalchemy
pydantic>=2.6
fastapi
uvicorn[standard]
httpx
python-dotenv
tenacity
```

> We're using `ChatOpenAI` from `langchain-openai` but configured to talk to OpenRouter's OpenAI-compatible endpoint. This is the simplest path and gives us `with_structured_output` for free.

---

## Phase 1 — Schemas First (the contract)

> **Everything else follows from these.** Build them first. Every node has typed input + typed output.

`src/mtg/schemas.py`:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

# ─── Parsed query ────────────────────────────────────────────
class ParsedQuery(BaseModel):
    format: Literal["commander", "60card"]
    commander: Optional[str] = None
    colors: Optional[list[Literal["W","U","B","R","G"]]] = None
    theme: str
    style: Literal["aggro","midrange","control","combo","tribal"]
    constraints: list[str] = []
    pinned_cards: list[str] = []
    banned_cards: list[str] = []

# ─── Slot manifest ───────────────────────────────────────────
class Slot(BaseModel):
    role: str
    count: int
    hint: str
    wave: int   # 1, 2, 3, 4

class DeckPlan(BaseModel):
    template: str
    slots: list[Slot]

# ─── Worker output ───────────────────────────────────────────
class SubQuery(BaseModel):
    role: str
    query: str  # dense-phrase RAG query

class WorkerOutput(BaseModel):
    role: str
    queries: list[SubQuery]

# ─── Card record (in DB) ─────────────────────────────────────
class CardRecord(BaseModel):
    name: str
    oracle_id: str
    mana_cost: str
    cmc: float
    colors: list[str]
    color_identity: list[str]
    type_line: str
    oracle_text: str
    keywords: list[str]
    power: Optional[str] = None
    toughness: Optional[str] = None
    edhrec_rank: Optional[int] = None
    legalities: dict[str, str]
    image_uri: Optional[str] = None
    reasoning: str   # the dense-phrase RAG document

# ─── Picker ──────────────────────────────────────────────────
class Pick(BaseModel):
    slot: str
    card: str
    reason: str

class PickerOutput(BaseModel):
    picks: list[Pick]

# ─── Validation ──────────────────────────────────────────────
class Issue(BaseModel):
    severity: Literal["critical","high","medium"]
    category: Literal["mana","curve","strategy","synergy","redundancy","legality"]
    issue: str
    detail: str
    fix_action: Literal["REPLACE","ADD","REMOVE"]
    target_card: Optional[str] = None
    slot: Optional[str] = None
    fix_hint: str

class ValidationReport(BaseModel):
    passed: bool
    issues: list[Issue]
```

Every LLM call uses `llm.with_structured_output(SomeSchema)`. **No string parsing anywhere.**

---

## Phase 2 — Ingestion: 300-Card Baseline

> Goal: a working RAG over 300 hand-picked staple cards before doing the full 30k. Faster iteration, easier debugging.

### 2.1 Fetch from Scryfall

`src/mtg/ingest/fetch_scryfall.py`:
- Hit `https://api.scryfall.com/bulk-data` to get the latest `default_cards` URL.
- Stream it to disk as `data/default_cards.json`.

For the 300-card baseline, `scripts/ingest_300.py`:
- Use Scryfall's search API with curated queries:
  - `"is:commander legal:commander cmc>=3"` top 100 by edhrec_rank
  - Top 50 ramp staples: `"o:'add' (t:artifact OR t:creature) cmc<=2"`
  - Top 50 removal staples
  - Top 50 dragons (for dragon-tribal eval)
  - 50 lands
- Dedup. Write to `data/cards_300.json`.

### 2.2 Generate reasoning (batched LLM)

`src/mtg/ingest/reasoning.py`:

```python
async def generate_reasoning_batch(cards: list[CardRecord]) -> list[str]:
    """200 cards per call. Returns dense-phrase reasoning per card."""
```

System prompt → `src/mtg/prompts/reasoning.txt`:
```
You generate dense-phrase RAG documents for Magic: The Gathering cards.

Each output is comma-separated phrases that a deck builder would search for.
Cover: mechanical role, archetype fit, game-state fit, synergy concepts,
format relevance.

Write in deck-builder language, not rules-lawyer language.

Examples:
- Sol Ring: "colorless ramp, 1 mana, 2 mana out, commander staple,
  any deck, fast mana, turn 1 acceleration, pairs with high cmc"
- Utvara Hellkite: "dragon tribal payoff, exponential token generation,
  6/6 flying tokens, attack trigger, late game finisher, ramp target,
  rewards wide dragon boards, commander staple"

Return JSON: {"reasonings": [{"name": "...", "reasoning": "..."}, ...]}
Order MUST match input order. Match by name on the receiving side.
```

For 300 cards: 2 calls of 150 each. ~5 seconds total.

### 2.3 Embed and insert

`src/mtg/ingest/embed.py`:

```python
# pgvector schema
"""
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
    embedding VECTOR(768)   -- nomic-embed-text dim
);

CREATE INDEX IF NOT EXISTS cards_embedding_idx
    ON cards USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS cards_name_idx ON cards (name);
CREATE INDEX IF NOT EXISTS cards_color_identity_idx ON cards USING gin (color_identity);
"""
```

Embed each card's `reasoning` via `OllamaEmbeddings(model="nomic-embed-text")`. Batched insert.

### 2.4 Verify

```bash
python scripts/ingest_300.py
psql -U mtg -d mtgdb -c "SELECT count(*) FROM cards;"
# Expect: 300
```

Sample similarity check:
```sql
SELECT name, reasoning
FROM cards
ORDER BY embedding <=> (
  SELECT embedding FROM cards WHERE name = 'Sol Ring'
)
LIMIT 5;
```
Expect: Arcane Signet, Mind Stone, Mana Vault — other cheap colorless rocks.

---

## Phase 3 — LangGraph Build (the orchestrator)

### 3.1 State

`src/mtg/graph/state.py`:

```python
from typing import TypedDict, Annotated, Optional
from operator import add
from .schemas import ParsedQuery, DeckPlan, Pick, Issue

class DeckBuildState(TypedDict):
    user_query: str
    parsed: Optional[ParsedQuery]
    plan: Optional[DeckPlan]
    current_wave: int
    picks: Annotated[list[Pick], add]      # accumulates across waves
    used_cards: list[str]                   # dedup set (as list for JSON)
    issues: list[Issue]
    repair_attempts: int
    final_deck: Optional[list[Pick]]
```

### 3.2 Templates

`src/mtg/graph/templates.py`:

```python
TEMPLATES = {
    "tribal": {
        "ratios": {
            "commander": 1, "theme": 26, "ramp": 14, "draw": 9,
            "removal": 8, "wipes": 4, "protection": 4, "lands": 34,
        },
        "waves": {
            1: ["commander", "theme"],
            2: ["ramp", "draw", "protection"],
            3: ["removal", "wipes"],
            4: ["lands"],
        },
    },
    "midrange": {  # ... },
    "aggro":    {  # ... },
    "control":  {  # ... },
    "combo":    {  # ... },
}

def get_template(style: str, format: str) -> dict:
    """Returns ratios + waves for the chosen archetype + format."""
```

For the 60-card formats, halve land counts and trim theme.

### 3.3 Nodes

Every node is a pure function `state -> partial state update`. Every LLM call uses `with_structured_output`.

`parse_query.py`:
```python
from mtg.llm import structured
from mtg.schemas import ParsedQuery

def parse_query(state: DeckBuildState) -> dict:
    parsed = structured(ParsedQuery).invoke(
        load_prompt("parse_query") + state["user_query"]
    )
    return {"parsed": parsed}
```

`plan_deck.py`:
```python
from mtg.llm import structured
from mtg.schemas import DeckPlan

def plan_deck(state: DeckBuildState) -> dict:
    parsed = state["parsed"]
    template = get_template(parsed.style, parsed.format)
    # Hardcoded ratios; LLM only writes the slot hints
    plan = structured(DeckPlan).invoke(
        f"Template ratios: {template['ratios']}\n"
        f"User intent: {parsed.model_dump_json()}\n"
        "Write a 1-sentence deck-builder hint for each slot. Keep counts EXACT."
    )
    return {"plan": plan, "current_wave": 1, "picks": [], "used_cards": []}
```

`worker.py` — runs once per slot in the current wave, in parallel via `Send`:

```python
from mtg.llm import structured
from mtg.schemas import WorkerOutput

def worker(state: dict) -> dict:
    """state has: slot, deck_context (from main state)"""
    output = structured(WorkerOutput).invoke(...)
    return {"worker_outputs": [output]}  # collected via reducer
```

`picker.py` — once per wave, sees all candidates from all workers in that wave:

```python
def picker(state: DeckBuildState) -> dict:
    # 1. For each sub-query in this wave, run RAG search → top 5 candidates
    # 2. Build candidate map { slot_id: [card1..card5] }
    # 3. Filter against state["used_cards"]
    # 4. Send to picker LLM with structured output
    # 5. Return picks
```

`hard_validate.py` — pure Python, no LLM:
```python
def hard_validate(state: DeckBuildState) -> dict:
    issues = []
    deck = state["picks"]
    parsed = state["parsed"]
    # count, color identity, singleton, banned list, pinned present
    return {"issues": issues}
```

`soft_validate.py` — LLM critic:
```python
from mtg.llm import structured
from mtg.schemas import ValidationReport

def soft_validate(state: DeckBuildState) -> dict:
    report = structured(ValidationReport).invoke(...)
    return {"issues": report.issues}
```

`repair.py` — for each issue, one targeted RAG search → pick → swap.

### 3.4 Wiring

`src/mtg/graph/build.py`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

def fan_out_wave(state: DeckBuildState) -> list[Send]:
    """Returns one Send per slot in the current wave."""
    plan = state["plan"]
    wave = state["current_wave"]
    return [
        Send("worker", {"slot": s, "deck_context": state})
        for s in plan.slots if s.wave == wave
    ]

def advance_wave(state: DeckBuildState) -> dict:
    return {"current_wave": state["current_wave"] + 1}

def is_build_complete(state: DeckBuildState) -> str:
    if state["current_wave"] > 4:
        return "hard_validate"
    return "fan_out"

graph = StateGraph(DeckBuildState)
graph.add_node("parse_query", parse_query)
graph.add_node("plan_deck", plan_deck)
graph.add_node("worker", worker)
graph.add_node("picker", picker)
graph.add_node("advance_wave", advance_wave)
graph.add_node("hard_validate", hard_validate)
graph.add_node("soft_validate", soft_validate)
graph.add_node("repair", repair)

graph.add_edge(START, "parse_query")
graph.add_edge("parse_query", "plan_deck")
graph.add_conditional_edges("plan_deck", fan_out_wave, ["worker"])
graph.add_edge("worker", "picker")
graph.add_edge("picker", "advance_wave")
graph.add_conditional_edges("advance_wave", is_build_complete,
                            {"fan_out": ..., "hard_validate": "hard_validate"})
graph.add_conditional_edges("hard_validate",
    lambda s: "repair" if s["issues"] else "soft_validate")
graph.add_conditional_edges("soft_validate",
    lambda s: "repair" if s["issues"] and s["repair_attempts"] < 3 else END)
graph.add_edge("repair", "soft_validate")

app = graph.compile()
```

---

## Phase 4 — Per-Node Testability

> Every node is independently testable. This is the entire point of structured outputs.

`scripts/eval_node.py`:

```python
"""
Usage:
  python scripts/eval_node.py parse_query "Build me an aggressive Krenko deck"
  python scripts/eval_node.py plan_deck --input fixtures/parsed_dragon.json
  python scripts/eval_node.py worker --slot ramp --context fixtures/dragon_context.json
"""
```

`tests/nodes/`:

```python
# test_parse_query.py
def test_parse_dragon_tribal():
    out = parse_query({"user_query": "dragon tribal commander, ur-dragon"})
    assert out["parsed"].format == "commander"
    assert out["parsed"].style == "tribal"
    assert "Ur-Dragon" in out["parsed"].pinned_cards
```

Save fixtures from real runs. Replay them.

> ⚠️ **Reasoning-model gotchas with Step 3.5 Flash.** Two things to know:
>
> 1. **Use `method="function_calling"`, not `json_schema`.** StepFun's own paper notes the model was trained primarily on XML tool-call templates and that JSON's escape sequences induce parsing errors in smaller models. The `structured()` helper above already does this — just don't override it.
> 2. **It emits `<think>...</think>` reasoning traces before the final answer.** With `function_calling` mode the trace stays out of the tool-call payload, but if you ever fall back to `json_schema` or raw text, you'll see leakage. Defenses: keep `temperature=0` for structured calls (already the helper default), and wrap the call in `OutputFixingParser` from `langchain-classic` for an automatic re-parse on failure. Add both retries to your test fixtures so regressions surface immediately.

### 4.1 Prompt tuning workflow

```
1. Run: python scripts/eval_node.py worker --slot theme --context fixtures/dragons.json
2. Inspect: outputs/worker_theme_run_$timestamp.json
3. Tweak: src/mtg/prompts/worker.txt
4. Re-run. Diff outputs.
5. Once happy, save as a test fixture.
```

Each prompt lives in its own `.txt` file. **Never inline prompts in Python.** This makes diffs trivial.

---

## Phase 5 — FastAPI Server + Local UI

### 5.1 API

`src/mtg/server/api.py`:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .schemas import BuildRequest, BuildResponse

app = FastAPI()

@app.post("/api/build")
async def build(req: BuildRequest) -> BuildResponse:
    final_state = await graph_app.ainvoke({"user_query": req.query})
    return BuildResponse(
        deck=final_state["final_deck"],
        plan=final_state["plan"],
        issues=final_state["issues"],
    )

@app.get("/api/card/{name}")
async def card_detail(name: str):
    """Fetch full Scryfall card data + image for UI."""

app.mount("/", StaticFiles(directory="src/mtg/server/static", html=True))
```

```bash
uvicorn mtg.server.api:app --reload
# Open http://localhost:8000
```

### 5.2 Baseline UI

`src/mtg/server/static/index.html` — single file, no build step.

Layout:
- Top: `<textarea>` + "Build deck" button
- Below: streaming progress (wave 1 → wave 2 → ...) via SSE or polling
- Right pane: card grid with image, role badge, click → modal with reasoning

Use plain `fetch()`, no React build chain. The point is to ship the baseline fast. (Upgrade to a real React app once the langgraph is stable.)

---

## Phase 6 — Output Rendering

When the graph completes:

1. Server returns `BuildResponse` containing the deck.
2. UI fetches images from `/api/card/{name}` (which proxies Scryfall, with a small in-memory cache).
3. Each card tile shows: image, role pill, count badge.
4. Click a card → modal with `Pick.reason` from the picker, `oracle_text`, mana cost, plus any `Issue.detail` if validator flagged it.

Optional but high-value: render the wave-by-wave progress while the graph runs (LangGraph supports `astream_events`). Each wave's picks pop into the grid as they're decided.

---

## Phase 7 — Going from 300 to 30k

Same code, different ingest script:

```bash
python scripts/ingest_full.py
```

- Streams the Scryfall bulk file
- Filters to cards legal in at least one of: commander, modern, standard, pioneer
- Batches reasoning generation 200/call
- **Concurrency: start with 3 (free-tier safe). Bump if no rate-limit errors.**
- Cost: **$0** on `:free` (subject to OpenRouter daily limits). On the paid Step 3.5 Flash tier, the full 30k run is still pennies.
- Wall time on free tier: expect 30–90 minutes (free models are slower + rate-limited). On a paid tier, ~5–10 minutes.
- Idempotent: keyed by `oracle_id`, upsert on conflict

For the periodic refresh after each new MTG set (~4×/year), the same script with a `--since <date>` flag pulls only the delta.

---

## Phase 8 — Acceptance Checklist

The system is "done baseline" when all of these pass:

- [ ] `docker compose up -d` brings up pgvector cleanly
- [ ] `python scripts/ingest_300.py` populates 300 cards with reasonings + embeddings
- [ ] Similarity query for "Sol Ring" returns other colorless rocks in top 5
- [ ] `pytest tests/nodes/` is green
- [ ] `eval_node.py worker --slot ramp` produces sensible dense-phrase queries on a sample context
- [ ] `POST /api/build` with `"dragon tribal commander, focus on Ur-Dragon"` returns a 100-card deck in under 30s
- [ ] The deck includes Ur-Dragon (pinned), 20+ dragons, recognizable ramp staples, ~36 lands
- [ ] Hard validate catches an injected illegal card (test by manually adding a blue card to a mono-red deck context)
- [ ] Soft validate flags a deck with 50 lands and only 5 creatures
- [ ] UI renders the deck with images and clickable card detail

---

## Phase 9 — Tuning Surfaces (in priority order)

When the baseline runs, tune in this order:

1. **`reasoning.txt`** — quality of every embedded document. Bad reasonings → bad retrieval forever.
2. **`worker.txt`** — quality of RAG queries. Drives what gets retrieved per slot.
3. **`soft_validator.txt`** — what counts as a "good deck." This is where MTG expertise lives.
4. **`templates.py` ratios** — adjust per archetype based on player feedback.
5. **`picker.txt`** — cross-slot synergy reasoning.

Build an eval set of 20 known-good queries (Atraxa superfriends, Krenko goblins, Yuriko ninjas, mono-red burn, esper control, etc.). After every prompt change, run the eval set and compare outputs side-by-side.

---

## What Not To Build Yet

Resist these until the baseline works:

- ❌ Streaming progress in UI (do it after build is stable)
- ❌ User-editable picks / regenerate-this-card buttons
- ❌ Multi-format support beyond commander + 60-card
- ❌ Authentication or persistence across sessions
- ❌ Production deployment

The point of this plan is a **single-user local tool that proves the architecture**. Once it works on 300 cards locally, scaling out is mechanical.

---

## File-by-File Order of Implementation

For an LLM agent executing this plan, build in this order to minimize blocked work:

1. `docker-compose.yml`, `pyproject.toml`, `.env.example`
2. `schemas.py` — all Pydantic models, no logic
3. `config.py` — env loading
4. `rag/db.py` — pgvector connection + raw SQL helpers
5. `ingest/fetch_scryfall.py` — fetch the 300-card subset
6. `ingest/reasoning.py` — batched LLM reasoning
7. `ingest/embed.py` — embed + insert
8. `scripts/ingest_300.py` — wires above three together
9. **CHECKPOINT**: similarity search by hand in psql
10. `graph/templates.py`
11. `graph/state.py`
12. `graph/nodes/parse_query.py` + `prompts/parse_query.txt`
13. `graph/nodes/plan_deck.py`
14. **CHECKPOINT**: eval_node parse_query and plan_deck on 5 sample queries
15. `rag/search.py` — RAG helper used by picker
16. `graph/nodes/worker.py` + `prompts/worker.txt`
17. `graph/nodes/picker.py` + `prompts/picker.txt`
18. `graph/nodes/hard_validate.py`
19. `graph/build.py` — wire the wave loop
20. **CHECKPOINT**: end-to-end run on "dragon tribal" prints picks to console
21. `graph/nodes/soft_validate.py` + `prompts/soft_validator.txt`
22. `graph/nodes/repair.py`
23. `server/api.py` + `static/index.html`
24. `tests/nodes/*` — backfill tests against fixtures from earlier runs

Each checkpoint is a hard stop: don't proceed until the current layer is observably correct. The structured-output contracts make every checkpoint a clean integration test.