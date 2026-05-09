"""Tests for the parse_query node. Requires OPENROUTER_API_KEY env var."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def test_parse_dragon_tribal():
    from mtg.graph.nodes.parse_query import parse_query

    result = parse_query({"user_query": "dragon tribal commander, ur-dragon"})
    parsed = result["parsed"]

    assert parsed.format == "commander"
    assert parsed.style == "tribal"
    assert any("Dragon" in c or "dragon" in c.lower() for c in parsed.pinned_cards)


def test_parse_aggro_60card():
    from mtg.graph.nodes.parse_query import parse_query

    result = parse_query({"user_query": "mono red burn deck for modern, budget"})
    parsed = result["parsed"]

    assert parsed.format == "60card"
    assert parsed.style in ("aggro", "combo")
    assert "R" in (parsed.colors or [])


def test_parse_control():
    from mtg.graph.nodes.parse_query import parse_query

    result = parse_query({"user_query": "esper control commander, Zur the Enchanter"})
    parsed = result["parsed"]

    assert parsed.format == "commander"
    assert parsed.style == "control"
