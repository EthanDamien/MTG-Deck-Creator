"""Tests for the worker node."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def _make_slot(role: str, wave: int = 2, count: int = 14):
    from mtg.schemas import Slot
    return Slot(role=role, count=count, hint=f"{role} cards for the deck", wave=wave)


def _dragon_context():
    from mtg.schemas import ParsedQuery
    return {
        "parsed": ParsedQuery(
            format="commander",
            commander="The Ur-Dragon",
            colors=["W", "U", "B", "R", "G"],
            theme="dragon tribal",
            style="tribal",
        )
    }


def test_worker_generates_queries():
    from mtg.graph.nodes.worker import worker

    slot = _make_slot("ramp")
    result = worker({"slot": slot, "deck_context": _dragon_context()})

    outputs = result["worker_outputs"]
    assert len(outputs) == 1
    wo = outputs[0]
    assert wo.role == "ramp"
    assert len(wo.queries) >= 3


def test_worker_theme_slot():
    from mtg.graph.nodes.worker import worker

    slot = _make_slot("theme", wave=1, count=26)
    result = worker({"slot": slot, "deck_context": _dragon_context()})

    outputs = result["worker_outputs"]
    wo = outputs[0]
    # Theme queries for dragon deck should mention dragons
    all_text = " ".join(q.query for q in wo.queries).lower()
    assert "dragon" in all_text
