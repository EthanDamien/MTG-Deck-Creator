#!/usr/bin/env python
"""Ingest 300-card baseline: fetch → reasoning → embed → insert."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

from mtg.ingest.fetch_scryfall import fetch_300_cards
from mtg.ingest.reasoning import generate_reasoning_batch
from mtg.ingest.embed import embed_and_insert

DATA_DIR = Path(__file__).parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
CARDS_FILE = DATA_DIR / "cards_300.json"

BATCH_SIZE = 30  # cards per LLM call


def main():
    # ── Step 1: Fetch ────────────────────────────────────────────────────────
    if CARDS_FILE.exists():
        print(f"[fetch] Loading cached cards from {CARDS_FILE}")
        cards = json.loads(CARDS_FILE.read_text())
    else:
        print("[fetch] Fetching cards from Scryfall...")
        t0 = time.time()
        cards = fetch_300_cards()
        CARDS_FILE.write_text(json.dumps(cards, indent=2))
        print(f"[fetch] Saved {len(cards)} cards in {time.time()-t0:.1f}s")

    print(f"[fetch] {len(cards)} cards total")

    # ── Step 2: Generate reasoning in batches, save after each ───────────────
    needs_reasoning = [c for c in cards if not c.get("reasoning")]
    print(f"[reasoning] {len(needs_reasoning)} cards need reasoning, {len(cards)-len(needs_reasoning)} already done")

    if needs_reasoning:
        batches = [needs_reasoning[i: i + BATCH_SIZE] for i in range(0, len(needs_reasoning), BATCH_SIZE)]
        print(f"[reasoning] {len(batches)} batches of ~{BATCH_SIZE} cards each")

        name_map = {c["name"]: c for c in cards}

        for i, batch in enumerate(batches):
            names_preview = [c["name"] for c in batch[:3]]
            print(f"\n[reasoning] Batch {i+1}/{len(batches)}: {names_preview}{'...' if len(batch)>3 else ''}")
            t0 = time.time()
            try:
                reasonings = generate_reasoning_batch(batch, batch_index=i, total_batches=len(batches))
                for card, reasoning in zip(batch, reasonings):
                    name_map[card["name"]]["reasoning"] = reasoning
                elapsed = time.time() - t0
                print(f"[reasoning] Batch {i+1} done in {elapsed:.1f}s — saving progress...")
                cards = list(name_map.values())
                CARDS_FILE.write_text(json.dumps(cards, indent=2))
                done = sum(1 for c in cards if c.get("reasoning"))
                print(f"[reasoning] {done}/{len(cards)} cards have reasoning so far")
            except Exception as e:
                print(f"[reasoning] ERROR on batch {i+1}: {e}")
                print("[reasoning] Skipping batch, will use oracle_text as fallback")
                for card in batch:
                    name_map[card["name"]]["reasoning"] = card.get("oracle_text", "")
                cards = list(name_map.values())
                CARDS_FILE.write_text(json.dumps(cards, indent=2))
    else:
        print("[reasoning] All cards already have reasoning — skipping")

    # ── Step 3: Embed + insert ───────────────────────────────────────────────
    print(f"\n[embed] Embedding and inserting {len(cards)} cards into pgvector...")
    t0 = time.time()
    count = embed_and_insert(cards)
    print(f"[embed] Done in {time.time()-t0:.1f}s — inserted/updated {count} cards")

    # ── Step 4: Verify ───────────────────────────────────────────────────────
    print("\n[verify] Checking DB count...")
    import psycopg
    from mtg.config import DATABASE_URL
    dsn = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT count(*) FROM cards").fetchone()
        print(f"[verify] Cards in DB: {row[0]}")


if __name__ == "__main__":
    main()
