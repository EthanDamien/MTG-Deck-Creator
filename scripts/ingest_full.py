#!/usr/bin/env python
"""Ingest the full 30k Scryfall card set."""
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from mtg.ingest.fetch_scryfall import fetch_bulk_data, normalize_card
from mtg.ingest.reasoning import generate_reasoning_all
from mtg.ingest.embed import embed_and_insert

DATA_DIR = Path(__file__).parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

LEGAL_FORMATS = {"commander", "modern", "standard", "pioneer"}


def load_bulk(path: Path) -> list[dict]:
    print("Loading bulk card data...")
    with open(path) as f:
        raw_cards = json.load(f)

    filtered = []
    for raw in raw_cards:
        if raw.get("object") != "card":
            continue
        # Only non-digital, non-token cards
        if raw.get("digital") or "Token" in raw.get("type_line", ""):
            continue
        # Legal in at least one target format
        legalities = raw.get("legalities", {})
        if not any(legalities.get(fmt) == "legal" for fmt in LEGAL_FORMATS):
            continue
        filtered.append(normalize_card(raw))

    print(f"Filtered to {len(filtered)} cards legal in target formats.")
    return filtered


def main(since: Optional[str] = None):
    bulk_path = DATA_DIR / "default_cards.json"
    if not bulk_path.exists():
        fetch_bulk_data(bulk_path)

    cards = load_bulk(bulk_path)

    # Deduplicate by oracle_id
    seen: set[str] = set()
    unique = []
    for c in cards:
        if c["oracle_id"] not in seen:
            seen.add(c["oracle_id"])
            unique.append(c)

    print(f"{len(unique)} unique cards after dedup.")

    # Reasoning generation — start with concurrency=3 for free tier safety
    print("Generating reasoning (this may take 30–90 min on free tier)...")
    cards_with_reasoning = generate_reasoning_all(unique, batch_size=150)

    print("Embedding and inserting...")
    count = embed_and_insert(cards_with_reasoning, batch_size=100)
    print(f"\nDone. {count} cards inserted/updated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="Only process cards updated after this date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(since=args.since)
