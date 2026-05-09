"""Tests for hard_validate (deterministic, no LLM needed)."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def _dragon_parsed():
    from mtg.schemas import ParsedQuery
    return ParsedQuery(
        format="commander",
        commander="The Ur-Dragon",
        colors=["W", "U", "B", "R", "G"],
        theme="dragon tribal",
        style="tribal",
        pinned_cards=["The Ur-Dragon"],
    )


def test_hard_validate_missing_commander():
    from mtg.schemas import Pick
    from mtg.graph.nodes.hard_validate import hard_validate

    # Build a near-complete deck without the commander
    picks = [Pick(slot="theme", card=f"Dragon {i}", reason="test") for i in range(99)]
    state = {
        "picks": picks,
        "parsed": _dragon_parsed(),
    }
    result = hard_validate(state)
    issues = result["issues"]
    assert any(i.issue == "Commander missing" for i in issues)


def test_hard_validate_duplicate():
    from mtg.schemas import Pick
    from mtg.graph.nodes.hard_validate import hard_validate

    picks = [Pick(slot="theme", card="Sol Ring", reason="test")] * 2
    picks += [Pick(slot="theme", card=f"Card {i}", reason="test") for i in range(97)]
    picks.append(Pick(slot="commander", card="The Ur-Dragon", reason="test"))

    state = {"picks": picks, "parsed": _dragon_parsed()}
    result = hard_validate(state)
    issues = result["issues"]
    assert any(i.issue == "Non-basic duplicate" and i.target_card == "Sol Ring" for i in issues)


def test_hard_validate_banned_card():
    from mtg.schemas import Pick
    from mtg.graph.nodes.hard_validate import hard_validate

    picks = [Pick(slot="commander", card="The Ur-Dragon", reason="test")]
    picks.append(Pick(slot="theme", card="Griselbrand", reason="test"))
    picks += [Pick(slot="theme", card=f"Card {i}", reason="test") for i in range(98)]

    state = {"picks": picks, "parsed": _dragon_parsed()}
    result = hard_validate(state)
    issues = result["issues"]
    assert any(i.issue == "Banned card" for i in issues)
