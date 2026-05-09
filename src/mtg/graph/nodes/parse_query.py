import time
from pathlib import Path
from mtg.llm import structured
from mtg.schemas import ParsedQuery
from mtg.graph.state import DeckBuildState

_PROMPT = (Path(__file__).parents[2] / "prompts" / "parse_query.txt").read_text()


def parse_query(state: DeckBuildState) -> dict:
    print(f"[parse_query] Parsing: {state['user_query']!r}", flush=True)
    t0 = time.time()
    parsed = structured(ParsedQuery).invoke(
        [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    print(f"[parse_query] Done in {time.time()-t0:.1f}s → format={parsed.format} style={parsed.style} commander={parsed.commander}", flush=True)
    return {"parsed": parsed}
