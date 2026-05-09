"""Fetch card data from Scryfall. Supports bulk download and search API."""
import json
import time
from pathlib import Path
from typing import Optional
import httpx

SCRYFALL_API = "https://api.scryfall.com"
DATA_DIR = Path(__file__).parents[4] / "data"


def _get(url: str, params: Optional[dict] = None) -> dict:
    time.sleep(0.1)  # Scryfall asks for 50-100ms between requests
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_cards(query: str, order: str = "edhrec") -> list[dict]:
    """Return all cards matching a Scryfall search query."""
    cards = []
    url = f"{SCRYFALL_API}/cards/search"
    params = {"q": query, "order": order, "unique": "cards"}
    while url:
        data = _get(url, params)
        cards.extend(data.get("data", []))
        url = data.get("next_page")
        params = None  # next_page URL already has params
    return cards


def normalize_card(raw: dict) -> dict:
    """Extract the fields we care about from a raw Scryfall card object."""
    # Handle double-faced cards — use front face for oracle text
    if "card_faces" in raw and "oracle_text" not in raw:
        face = raw["card_faces"][0]
        oracle_text = face.get("oracle_text", "")
        mana_cost = face.get("mana_cost", raw.get("mana_cost", ""))
    else:
        oracle_text = raw.get("oracle_text", "")
        mana_cost = raw.get("mana_cost", "")

    image_uri = None
    if "image_uris" in raw:
        image_uri = raw["image_uris"].get("normal")
    elif "card_faces" in raw:
        image_uri = raw["card_faces"][0].get("image_uris", {}).get("normal")

    return {
        "oracle_id": raw["oracle_id"],
        "name": raw["name"],
        "mana_cost": mana_cost,
        "cmc": raw.get("cmc", 0.0),
        "colors": raw.get("colors", []),
        "color_identity": raw.get("color_identity", []),
        "type_line": raw.get("type_line", ""),
        "oracle_text": oracle_text,
        "keywords": raw.get("keywords", []),
        "power": raw.get("power"),
        "toughness": raw.get("toughness"),
        "edhrec_rank": raw.get("edhrec_rank"),
        "legalities": raw.get("legalities", {}),
        "image_uri": image_uri,
        "reasoning": "",  # filled by ingest/reasoning.py
    }


def fetch_bulk_data(dest: Path = DATA_DIR / "default_cards.json") -> Path:
    """Download the full Scryfall default_cards bulk file to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = _get(f"{SCRYFALL_API}/bulk-data")
    bulk_url = next(
        item["download_uri"]
        for item in meta["data"]
        if item["type"] == "default_cards"
    )
    print(f"Downloading bulk data from {bulk_url} ...")
    with httpx.stream("GET", bulk_url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    print(f"Saved to {dest}")
    return dest


def fetch_300_cards() -> list[dict]:
    """Fetch ~300 staple cards across several curated queries."""
    queries = [
        ("is:commander legal:commander cmc>=3", 100),
        ("o:'add' (t:artifact OR t:creature) cmc<=2 legal:commander", 50),
        ("(o:destroy OR o:exile) t:instant legal:commander", 50),
        ("t:dragon legal:commander", 50),
        ("t:land legal:commander", 50),
    ]

    seen: set[str] = set()
    cards: list[dict] = []

    for query, limit in queries:
        results = search_cards(query)
        for raw in results[:limit]:
            oid = raw.get("oracle_id", "")
            if oid and oid not in seen:
                seen.add(oid)
                cards.append(normalize_card(raw))
        if len(cards) >= 300:
            break

    return cards[:300]
