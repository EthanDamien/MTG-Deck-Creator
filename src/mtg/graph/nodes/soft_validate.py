from pathlib import Path
from mtg.llm import structured
from mtg.schemas import ValidationReport
from mtg.graph.state import DeckBuildState

_PROMPT = (Path(__file__).parents[2] / "prompts" / "soft_validator.txt").read_text()


def soft_validate(state: DeckBuildState) -> dict:
    picks = state.get("picks", [])
    parsed = state["parsed"]

    deck_lines = "\n".join(f"  [{p.slot}] {p.card}" for p in picks)
    user_msg = (
        f"Deck intent: {parsed.model_dump_json()}\n\n"
        f"Current deck ({len(picks)} cards):\n{deck_lines}"
    )

    report: ValidationReport = structured(ValidationReport).invoke(
        [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )

    return {"issues": report.issues}
