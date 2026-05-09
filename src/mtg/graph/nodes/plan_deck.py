import time
from mtg.llm import structured
from mtg.schemas import DeckPlan
from mtg.graph.state import DeckBuildState
from mtg.graph.templates import get_template


def plan_deck(state: DeckBuildState) -> dict:
    parsed = state["parsed"]
    template = get_template(parsed.style, parsed.format)
    print(f"[plan_deck] Building plan for style={parsed.style} format={parsed.format}", flush=True)
    t0 = time.time()

    plan = structured(DeckPlan).invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert MTG deck builder. Given a template with exact slot counts "
                    "and a user's deck intent, write a 1-sentence deck-builder hint for each slot. "
                    "Keep counts EXACTLY as specified — do not change them."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Template name: {parsed.style}\n"
                    f"Template ratios: {template['ratios']}\n"
                    f"Wave assignments: {template['waves']}\n"
                    f"User intent: {parsed.model_dump_json()}\n\n"
                    "Return a DeckPlan with one Slot per entry in the ratios, "
                    "each with the exact count from the template and a brief hint."
                ),
            },
        ]
    )

    print(f"[plan_deck] Done in {time.time()-t0:.1f}s → {len(plan.slots)} slots", flush=True)
    for s in plan.slots:
        print(f"  wave{s.wave} [{s.role}] x{s.count}", flush=True)

    return {
        "plan": plan,
        "current_wave": 1,
        "picks": [],
        "used_cards": [],
        "issues": [],
        "repair_attempts": 0,
    }
