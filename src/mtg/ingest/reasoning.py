"""Batch LLM reasoning generation for card RAG documents."""
import json
import time
from pathlib import Path
from mtg.llm import get_llm

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "reasoning.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def generate_reasoning_batch(cards: list[dict], batch_index: int = 0, total_batches: int = 1) -> list[str]:
    """Generate dense-phrase RAG reasoning for a batch of cards."""
    llm = get_llm(temperature=0.3)
    prompt = _load_prompt()

    card_list = [
        {
            "name": c["name"],
            "mana_cost": c["mana_cost"],
            "cmc": c["cmc"],
            "type_line": c["type_line"],
            "oracle_text": c["oracle_text"][:200],  # truncate long oracle texts
            "keywords": c["keywords"],
            "colors": c["colors"],
        }
        for c in cards
    ]

    user_msg = f"Cards:\n{json.dumps(card_list)}"

    print(f"  [batch {batch_index+1}/{total_batches}] Sending {len(cards)} cards to LLM... ", end="", flush=True)
    t0 = time.time()

    response = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg},
    ])

    elapsed = time.time() - t0
    raw = response.content
    print(f"got response in {elapsed:.1f}s ({len(raw)} chars)")

    # Strip <think>...</think> reasoning traces
    if "<think>" in raw:
        think_end = raw.rfind("</think>")
        if think_end != -1:
            raw = raw[think_end + len("</think>"):].strip()

    # Extract JSON from markdown fences if present
    if "```" in raw:
        start = raw.find("```")
        end = raw.rfind("```")
        raw = raw[start + 3:end].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    # Find the first { to handle any leading whitespace/text
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [batch {batch_index+1}] JSON parse error: {e}")
        print(f"  Raw response (first 500 chars): {raw[:500]}")
        # Fall back to oracle text for all cards in batch
        return [c["oracle_text"] for c in cards]

    name_to_reasoning: dict[str, str] = {
        r["name"]: r["reasoning"] for r in data.get("reasonings", [])
    }

    missing = [c["name"] for c in cards if c["name"] not in name_to_reasoning]
    if missing:
        print(f"  [batch {batch_index+1}] WARNING: missing reasoning for {len(missing)} cards: {missing[:5]}")

    return [name_to_reasoning.get(c["name"], c["oracle_text"]) for c in cards]


def generate_reasoning_all(cards: list[dict], batch_size: int = 30) -> list[dict]:
    """Generate reasoning for all cards in batches, return updated card dicts."""
    results = []
    batches = [cards[i: i + batch_size] for i in range(0, len(cards), batch_size)]
    total = len(batches)

    for i, batch in enumerate(batches):
        start_num = i * batch_size + 1
        end_num = start_num + len(batch) - 1
        print(f"  Batch {i+1}/{total}: cards {start_num}–{end_num} ({[c['name'] for c in batch[:3]]}...)")
        reasonings = generate_reasoning_batch(batch, batch_index=i, total_batches=total)
        for card, reasoning in zip(batch, reasonings):
            results.append({**card, "reasoning": reasoning})
        # Save progress after each batch
        print(f"  Progress: {len(results)}/{len(cards)} cards have reasoning")

    return results
