import httpx
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from mtg.schemas import BuildRequest, BuildResponse
from mtg.graph.build import app as graph_app

api = FastAPI(title="MTG Deck Builder")

_scryfall_cache: dict[str, dict] = {}


@api.post("/api/build")
async def build(req: BuildRequest) -> BuildResponse:
    final_state = await graph_app.ainvoke({"user_query": req.query})
    return BuildResponse(
        deck=final_state.get("final_deck") or final_state.get("picks", []),
        plan=final_state.get("plan"),
        issues=final_state.get("issues", []),
    )


@api.get("/api/card/{name}")
async def card_detail(name: str) -> dict:
    if name in _scryfall_cache:
        return _scryfall_cache[name]

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://api.scryfall.com/cards/named",
            params={"fuzzy": name},
        )
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Card not found")
        r.raise_for_status()
        data = r.json()

    _scryfall_cache[name] = data
    return data


static_dir = Path(__file__).parent / "static"
api.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
