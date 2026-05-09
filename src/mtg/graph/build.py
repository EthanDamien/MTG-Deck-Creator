"""LangGraph orchestrator wiring."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from mtg.graph.state import DeckBuildState
from mtg.graph.nodes.parse_query import parse_query
from mtg.graph.nodes.plan_deck import plan_deck
from mtg.graph.nodes.worker import worker
from mtg.graph.nodes.picker import picker
from mtg.graph.nodes.hard_validate import hard_validate
from mtg.graph.nodes.soft_validate import soft_validate
from mtg.graph.nodes.repair import repair


def fan_out_wave(state: DeckBuildState) -> list[Send]:
    """Fan out one Send per slot in the current wave."""
    plan = state["plan"]
    wave = state["current_wave"]
    return [
        Send("worker", {"slot": s, "deck_context": state})
        for s in plan.slots
        if s.wave == wave
    ]


def advance_wave(state: DeckBuildState) -> dict:
    return {"current_wave": state["current_wave"] + 1}


def route_after_advance(state: DeckBuildState) -> list[Send] | str:
    """After advancing the wave counter: fan out again or move to validation."""
    if state["current_wave"] > 4:
        return "hard_validate"
    plan = state["plan"]
    wave = state["current_wave"]
    return [
        Send("worker", {"slot": s, "deck_context": state})
        for s in plan.slots
        if s.wave == wave
    ]


def route_after_hard(state: DeckBuildState) -> str:
    critical = [i for i in state.get("issues", []) if i.severity == "critical"]
    return "repair" if critical else "soft_validate"


def route_after_soft(state: DeckBuildState) -> str:
    issues = state.get("issues", [])
    attempts = state.get("repair_attempts", 0)
    if issues and attempts < 3:
        return "repair"
    return "finalize"


def finalize(state: DeckBuildState) -> dict:
    return {"final_deck": state["picks"]}


builder = StateGraph(DeckBuildState)

builder.add_node("parse_query", parse_query)
builder.add_node("plan_deck", plan_deck)
builder.add_node("worker", worker)
builder.add_node("picker", picker)
builder.add_node("advance_wave", advance_wave)
builder.add_node("hard_validate", hard_validate)
builder.add_node("soft_validate", soft_validate)
builder.add_node("repair", repair)
builder.add_node("finalize", finalize)

builder.add_edge(START, "parse_query")
builder.add_edge("parse_query", "plan_deck")
builder.add_conditional_edges("plan_deck", fan_out_wave, ["worker"])
builder.add_edge("worker", "picker")
builder.add_edge("picker", "advance_wave")
builder.add_conditional_edges(
    "advance_wave",
    route_after_advance,
    ["worker", "hard_validate"],
)
builder.add_conditional_edges(
    "hard_validate",
    route_after_hard,
    {"repair": "repair", "soft_validate": "soft_validate"},
)
builder.add_conditional_edges(
    "soft_validate",
    route_after_soft,
    {"repair": "repair", "finalize": "finalize"},
)
builder.add_edge("repair", "soft_validate")
builder.add_edge("finalize", END)

app = builder.compile(checkpointer=MemorySaver())
