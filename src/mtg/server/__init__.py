import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from mtg.schemas import BuildRequest
from mtg.rag.db import get_conn

api = FastAPI(title="MTG Deck Builder")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enrich_picks(picks, conn):
    names = list({p.card for p in picks})
    if not names:
        return []
    rows = conn.execute(
        "SELECT name, image_uri FROM cards WHERE name = ANY(%s)", (names,)
    ).fetchall()
    images = {r["name"]: r["image_uri"] for r in rows}
    return [
        {"slot": p.slot, "card": p.card, "reason": p.reason, "image_uri": images.get(p.card)}
        for p in picks
    ]


@api.get("/api/decks")
async def list_decks():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, query, created_at, picks, plan, issues FROM decks ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return JSONResponse(content=[
        {
            "id": r["id"],
            "query": r["query"],
            "created_at": r["created_at"].isoformat(),
            "picks": r["picks"],
            "plan": r["plan"],
            "issues": r["issues"],
        }
        for r in rows
    ])


@api.post("/api/build")
async def build(req: BuildRequest):
    from mtg.graph.build import app as graph_app

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph_app.invoke({"user_query": req.query}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    picks = result.get("final_deck") or result.get("picks") or []
    plan = result.get("plan")
    issues = result.get("issues") or []

    with get_conn() as conn:
        enriched = _enrich_picks(picks, conn)
        plan_dict = plan.model_dump() if plan else None
        issues_list = [i.model_dump() for i in issues]

        row = conn.execute(
            "INSERT INTO decks (query, picks, plan, issues) VALUES (%s, %s, %s, %s) RETURNING id, created_at",
            (req.query, json.dumps(enriched), json.dumps(plan_dict), json.dumps(issues_list)),
        ).fetchone()
        conn.commit()

    return JSONResponse(content={
        "id": row["id"],
        "created_at": row["created_at"].isoformat(),
        "deck": enriched,
        "plan": plan_dict,
        "issues": issues_list,
    })


static_dir = Path(__file__).parent / "static"
api.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
