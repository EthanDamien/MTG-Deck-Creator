import time
from pathlib import Path
from mtg.llm import structured
from mtg.schemas import PickerOutput, Pick
from mtg.graph.state import DeckBuildState
from mtg.rag.search import search

_PROMPT = (Path(__file__).parents[2] / "prompts" / "picker.txt").read_text()


def picker(state: DeckBuildState) -> dict:
    plan = state["plan"]
    parsed = state["parsed"]
    current_wave = state["current_wave"]
    used_cards = set(state.get("used_cards", []))
    worker_outputs = state.get("worker_outputs", [])

    print(f"[picker] wave={current_wave} worker_outputs={len(worker_outputs)}", flush=True)
    color_identity = parsed.colors if parsed.colors else None
    wave_slots = {s.role: s for s in plan.slots if s.wave == current_wave}

    # RAG search per slot
    candidate_map: dict[str, list[dict]] = {}
    for wo in worker_outputs:
        role = wo.role
        if role not in wave_slots:
            continue
        all_candidates: list[dict] = []
        seen_names: set[str] = set()
        for subq in wo.queries:
            results = search(subq.query, limit=8, color_identity=color_identity, exclude_names=list(used_cards))
            for r in results:
                if r["name"] not in seen_names and r["name"] not in used_cards:
                    seen_names.add(r["name"])
                    all_candidates.append(r)
        candidate_map[role] = all_candidates[:20]
        print(f"[picker]   {role}: {len(all_candidates)} candidates", flush=True)

    slots_text = ""
    for role, slot in wave_slots.items():
        candidates = candidate_map.get(role, [])
        card_lines = "\n".join(
            f"  - {c['name']} [{c['type_line']}] {c['mana_cost']}: {c['reasoning']}"
            for c in candidates[:15]
        )
        slots_text += f"\n### Slot: {role} (need exactly {slot.count} cards)\n{card_lines}\n"

    already_picked = [p.card for p in state.get("picks", [])]
    user_msg = (
        f"Deck context: {parsed.model_dump_json()}\n"
        f"Already picked (do not repeat): {already_picked}\n"
        f"\nCandidate cards per slot:\n{slots_text}"
    )

    print(f"[picker] Calling LLM to pick cards for wave {current_wave}...", flush=True)
    t0 = time.time()
    output = structured(PickerOutput).invoke(
        [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )

    if output is None or not hasattr(output, "picks"):
        print(f"[picker] WARNING: LLM returned None/invalid for wave {current_wave}, using empty picks", flush=True)
        new_picks = []
    else:
        new_picks = output.picks or []

    new_used = list(used_cards | {p.card for p in new_picks})
    print(f"[picker] wave={current_wave} done in {time.time()-t0:.1f}s → {len(new_picks)} picks", flush=True)
    for p in new_picks:
        print(f"  [{p.slot}] {p.card}", flush=True)

    return {
        "picks": new_picks,
        "used_cards": new_used,
        "worker_outputs": [],
    }
