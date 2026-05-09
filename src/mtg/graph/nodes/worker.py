import time
from pathlib import Path
from mtg.llm import structured
from mtg.schemas import WorkerOutput

_PROMPT = (Path(__file__).parents[2] / "prompts" / "worker.txt").read_text()


def worker(state: dict) -> dict:
    slot = state["slot"]
    deck_context = state["deck_context"]
    parsed = deck_context.get("parsed", {})
    parsed_str = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else str(parsed)

    print(f"[worker] slot={slot.role} count={slot.count}", flush=True)
    t0 = time.time()

    output = structured(WorkerOutput).invoke(
        [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": (
                    f"Slot: {slot.role} (need {slot.count} cards)\n"
                    f"Hint: {slot.hint}\n"
                    f"Deck context: {parsed_str}\n\n"
                    "Generate 3-5 dense-phrase RAG queries for this slot."
                ),
            },
        ]
    )

    output.role = slot.role
    print(f"[worker] slot={slot.role} done in {time.time()-t0:.1f}s → {len(output.queries)} queries", flush=True)
    return {"worker_outputs": [output]}
