"""Deterministic hard validation — no LLM."""
from collections import Counter
from mtg.schemas import Issue
from mtg.graph.state import DeckBuildState

# Comprehensive commander ban list (key examples — extend as needed)
BANNED_COMMANDER = {
    "ancestral recall", "balance", "biorhythm", "black lotus", "braids cabal minion",
    "channel", "chaos orb", "coalition victory", "contract from below",
    "emrakul the aeons torn", "fastbond", "falling star", "gifts ungiven",
    "griselbrand", "karakas", "leovold emissary of trest", "library of alexandria",
    "limited resources", "lion's eye diamond", "mox emerald", "mox jet", "mox pearl",
    "mox ruby", "mox sapphire", "panoptic mirror", "paradox engine",
    "primeval titan", "prophet of kruphix", "recurring nightmare", "rofellos llanowar emissary",
    "sundering titan", "sway of the stars", "sylvan primordial", "time stretch",
    "time vault", "time walk", "tinker", "tolarian academy", "trade secrets",
    "upheaval", "worldfire", "yawgmoth's bargain",
}


def hard_validate(state: DeckBuildState) -> dict:
    issues: list[Issue] = []
    picks = state.get("picks", [])
    parsed = state["parsed"]

    if not picks:
        return {"issues": []}

    # ── Count check ──────────────────────────────────────────────────────────
    expected = 100 if parsed.format == "commander" else 60
    actual = len(picks)
    if actual != expected:
        issues.append(Issue(
            severity="critical",
            category="mana",
            issue="Wrong deck size",
            detail=f"Deck has {actual} cards, expected {expected}.",
            fix_action="ADD" if actual < expected else "REMOVE",
            fix_hint=f"{'Add' if actual < expected else 'Remove'} {abs(expected - actual)} cards.",
        ))

    # ── Singleton check (commander) ──────────────────────────────────────────
    if parsed.format == "commander":
        name_counts = Counter(p.card for p in picks)
        for name, count in name_counts.items():
            if count > 1 and name.lower() not in {"plains", "island", "swamp", "mountain", "forest"}:
                issues.append(Issue(
                    severity="critical",
                    category="legality",
                    issue="Non-basic duplicate",
                    detail=f"{name} appears {count} times (singleton format).",
                    fix_action="REMOVE",
                    target_card=name,
                    fix_hint=f"Remove {count - 1} copies of {name}.",
                ))

    # ── Commander present ────────────────────────────────────────────────────
    if parsed.format == "commander" and parsed.commander:
        card_names_lower = {p.card.lower() for p in picks}
        if parsed.commander.lower() not in card_names_lower:
            issues.append(Issue(
                severity="critical",
                category="legality",
                issue="Commander missing",
                detail=f"Commander {parsed.commander!r} is not in the deck.",
                fix_action="ADD",
                target_card=parsed.commander,
                slot="commander",
                fix_hint=f"Add {parsed.commander} to the deck.",
            ))

    # ── Pinned cards present ─────────────────────────────────────────────────
    if parsed.pinned_cards:
        card_names_lower = {p.card.lower() for p in picks}
        for pinned in parsed.pinned_cards:
            if pinned.lower() not in card_names_lower:
                issues.append(Issue(
                    severity="high",
                    category="strategy",
                    issue="Pinned card missing",
                    detail=f"{pinned!r} was requested but not included.",
                    fix_action="ADD",
                    target_card=pinned,
                    fix_hint=f"Add {pinned} to the deck.",
                ))

    # ── Banned cards ─────────────────────────────────────────────────────────
    if parsed.format == "commander":
        for pick in picks:
            if pick.card.lower() in BANNED_COMMANDER:
                issues.append(Issue(
                    severity="critical",
                    category="legality",
                    issue="Banned card",
                    detail=f"{pick.card} is banned in Commander.",
                    fix_action="REMOVE",
                    target_card=pick.card,
                    slot=pick.slot,
                    fix_hint=f"Replace {pick.card} with a legal alternative.",
                ))

    # ── Color identity check ─────────────────────────────────────────────────
    if parsed.format == "commander" and parsed.colors:
        allowed = set(parsed.colors)
        for pick in picks:
            # We'd need DB access to check color_identity per card; defer to soft validator
            pass

    # ── Land count sanity ────────────────────────────────────────────────────
    land_picks = [p for p in picks if p.slot == "lands"]
    min_lands = 30 if parsed.format == "commander" else 18
    max_lands = 45 if parsed.format == "commander" else 28
    if len(land_picks) < min_lands:
        issues.append(Issue(
            severity="high",
            category="mana",
            issue="Too few lands",
            detail=f"Only {len(land_picks)} lands (minimum {min_lands} recommended).",
            fix_action="ADD",
            slot="lands",
            fix_hint=f"Add {min_lands - len(land_picks)} more lands.",
        ))
    elif len(land_picks) > max_lands:
        issues.append(Issue(
            severity="medium",
            category="mana",
            issue="Too many lands",
            detail=f"{len(land_picks)} lands (maximum {max_lands} recommended).",
            fix_action="REMOVE",
            slot="lands",
            fix_hint=f"Remove {len(land_picks) - max_lands} lands.",
        ))

    passed = all(i.severity != "critical" for i in issues)
    return {"issues": issues, "final_deck": picks if passed else None}
