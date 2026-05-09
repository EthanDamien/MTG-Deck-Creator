"""Repair node: resolve issues one at a time via targeted RAG + LLM pick."""
from mtg.schemas import Issue, Pick
from mtg.graph.state import DeckBuildState
from mtg.rag.search import search
from mtg.llm import structured
from mtg.schemas import PickerOutput


def repair(state: DeckBuildState) -> dict:
    issues = state.get("issues", [])
    picks = list(state.get("picks", []))
    used_cards = set(state.get("used_cards", []))
    parsed = state["parsed"]
    repair_attempts = state.get("repair_attempts", 0)

    if not issues:
        return {"repair_attempts": repair_attempts + 1}

    # Process critical issues first, then high
    sorted_issues = sorted(issues, key=lambda i: {"critical": 0, "high": 1, "medium": 2}[i.severity])

    for issue in sorted_issues[:3]:  # fix at most 3 per repair pass
        if issue.fix_action == "REMOVE" and issue.target_card:
            picks = [p for p in picks if p.card != issue.target_card]
            used_cards.discard(issue.target_card)

        elif issue.fix_action == "REPLACE" and issue.target_card:
            # Remove the offending card, find a replacement via RAG
            picks = [p for p in picks if p.card != issue.target_card]
            used_cards.discard(issue.target_card)
            replacement = _find_replacement(issue, parsed, used_cards)
            if replacement:
                picks.append(replacement)
                used_cards.add(replacement.card)

        elif issue.fix_action == "ADD":
            # Find a card to add for this slot
            if issue.target_card:
                # Specific card requested (e.g. commander or pinned)
                picks.append(Pick(
                    slot=issue.slot or "unknown",
                    card=issue.target_card,
                    reason=f"Added to fix: {issue.issue}",
                ))
                used_cards.add(issue.target_card)
            else:
                replacement = _find_replacement(issue, parsed, used_cards)
                if replacement:
                    picks.append(replacement)
                    used_cards.add(replacement.card)

    return {
        "picks": picks,
        "used_cards": list(used_cards),
        "issues": [],  # cleared; soft_validate will re-check
        "repair_attempts": repair_attempts + 1,
    }


def _find_replacement(issue: Issue, parsed, used_cards: set) -> Pick | None:
    color_identity = parsed.colors if parsed.colors else None
    query = issue.fix_hint or f"{issue.slot or 'utility'} card for {parsed.theme} deck"

    results = search(query, limit=10, color_identity=color_identity, exclude_names=list(used_cards))
    if not results:
        return None

    best = results[0]
    output: PickerOutput = structured(PickerOutput).invoke(
        [
            {
                "role": "system",
                "content": "Pick the single best card from the candidates to fix the described issue. Return exactly one pick.",
            },
            {
                "role": "user",
                "content": (
                    f"Issue: {issue.issue} — {issue.detail}\n"
                    f"Slot: {issue.slot or 'any'}\n"
                    f"Deck: {parsed.model_dump_json()}\n"
                    f"Candidates:\n"
                    + "\n".join(
                        f"  - {r['name']}: {r['reasoning']}" for r in results[:8]
                    )
                ),
            },
        ]
    )

    if output and output.picks:
        p = output.picks[0]
        if not p.slot:
            p.slot = issue.slot or "utility"
        return p

    return Pick(slot=issue.slot or "utility", card=best["name"], reason=issue.fix_hint)
